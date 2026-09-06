import json
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from ids_pipeline import cli, preprocessing as prep
from ids_pipeline.clustering import _fit_cpu_minibatch_kmeans
from ids_pipeline.config import PipelineConfig
from ids_pipeline.dataset_schema import model_feature_types, normalize_dataset_schema
from ids_pipeline.feature_policy import BEHAVIORAL_STRICT_FEATURES, BEHAVIORAL_STRICT_VERSION, FeaturePolicy
from ids_pipeline.leakage_checks import LeakageError, check_for_leakage_columns
from ids_pipeline.nominal_encoding import ConnStateOneHot
from ids_pipeline.schema import TON_IOT_SCHEMA

EXPECTED = ['duration','src_bytes','dst_bytes','conn_state','missed_bytes','src_pkts','src_ip_bytes','dst_pkts','dst_ip_bytes']
NUMERIC = [c for c in EXPECTED if c != 'conn_state']
FORBIDDEN = ['ssl_resumed','attack_count','future_sum','future_min','future_max','tcp_window',
             'packets_future','src_ip','dst_ip','src_port','dst_port','service','proto','ts','label','type']


def fixture_frame(n=12):
    rng = np.random.default_rng(24)
    frame = pd.DataFrame({c:rng.integers(1,1000,n).astype(str) for c in NUMERIC})
    frame['conn_state'] = (['SF','S0','REJ']*((n+2)//3))[:n]
    frame['ts'] = 1554220429 + np.arange(n)
    frame['label'] = np.arange(n)%2
    frame['type'] = np.where(frame.label==0,'normal','dos')
    for c in FORBIDDEN:
        if c not in frame:
            frame[c] = 'not_selected'
    return frame


def splits(n=12):
    return {'train':np.arange(n//2), 'val':np.arange(n//2,3*n//4), 'test':np.arange(3*n//4,n)}


def dense(X):
    return X.toarray() if hasattr(X,'toarray') else np.asarray(X)


def test_policy_exact_order_version_and_no_substring_selection():
    frame=fixture_frame()
    policy=FeaturePolicy.from_name('behavioral_strict')
    assert policy.version == 'behavioral_strict/2.0.0'
    assert policy.select_features(frame)==EXPECTED
    assert policy.select_features(frame[frame.columns[::-1]])==EXPECTED
    assert set(FORBIDDEN).isdisjoint(EXPECTED)


@pytest.mark.parametrize('missing',EXPECTED)
def test_every_required_feature_is_mandatory(missing):
    with pytest.raises(ValueError,match=missing):
        FeaturePolicy.from_name('behavioral_strict').select_features(fixture_frame().drop(columns=missing))


@pytest.mark.parametrize('extra',FORBIDDEN)
def test_direct_preprocessing_cannot_bypass_strict_allowlist(extra,tmp_path):
    with pytest.raises(LeakageError):
        prep.fit_transform_preprocessing(fixture_frame(),splits(),EXPECTED+[extra],
                                         PipelineConfig(feature_policy='behavioral_strict'),tmp_path)


def test_reorder_and_custom_target_conflicts_cannot_bypass_contract():
    frame=fixture_frame()
    with pytest.raises(LeakageError):
        check_for_leakage_columns(frame,EXPECTED[::-1],feature_policy='behavioral_strict')
    with pytest.raises(ValueError,match='target conflicts'):
        FeaturePolicy.from_name('behavioral_strict').select_features(frame,label_col='src_bytes')


def test_semantic_roles_ignore_numeric_storage_in_partitions():
    first=fixture_frame()[EXPECTED]
    second=first.copy()
    for col in NUMERIC:
        second[col]=pd.to_numeric(second[col])
    for frame in [first,second]:
        normalized,_=normalize_dataset_schema(frame)
        assert model_feature_types(normalized,EXPECTED,strict=True)==(NUMERIC,['conn_state'])
    assert TON_IOT_SCHEMA['conn_state'].semantic_type=='nominal'
    assert TON_IOT_SCHEMA['conn_state'].model_treatment=='categorical'


def test_nominal_encoder_train_only_missing_and_unknown_and_serialization(tmp_path):
    train=pd.DataFrame({'conn_state':['SF','S0','SF',None]})
    encoder=ConnStateOneHot().fit(train)
    assert encoder.categories_.tolist()==['S0','SF']
    val=pd.DataFrame({'conn_state':['NEW_VAL',None,'S0']})
    expected=np.array([[0,0],[0,1],[1,0]],dtype=np.float32)
    np.testing.assert_array_equal(encoder.transform(val).toarray(),expected)
    before=encoder.metadata()
    assert encoder.transform(pd.DataFrame({'conn_state':['NEW_TEST']})).shape==(1,2)
    assert encoder.metadata()==before
    joblib.dump(encoder,tmp_path/'encoder.joblib')
    restored=joblib.load(tmp_path/'encoder.joblib')
    np.testing.assert_array_equal(restored.transform(val).toarray(),expected)


def test_all_missing_training_keeps_one_stable_nominal_column():
    encoder=ConnStateOneHot().fit(pd.DataFrame({'conn_state':[None,None]}))
    assert encoder.categories_.tolist()==['__MISSING__']
    np.testing.assert_array_equal(encoder.transform(pd.DataFrame({'conn_state':[None,'SF']})).toarray(),[[1],[0]])


def test_one_hot_has_equal_distances_between_known_states_cpu_gpu(monkeypatch):
    monkeypatch.setitem(sys.modules,'cupy',np)
    frame=pd.DataFrame({'conn_state':['S0','SF','REJ']})
    enc=ConnStateOneHot().fit(frame)
    cpu=enc.transform(frame).toarray()
    gpu=enc.transform_gpu(frame)
    np.testing.assert_array_equal(cpu,gpu)
    assert cpu.shape==(3,3)
    assert np.all(cpu.sum(axis=1)==1)
    distances=((gpu[:,None]-gpu[None,:])**2).sum(axis=2)
    np.testing.assert_array_equal(distances[np.triu_indices(3,1)],[2,2,2])
    with pytest.raises(ValueError,match='never ordinal'):
        prep._gpu_categorical_code_arrays(frame,frame,frame,['conn_state'],1,64)


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_full_preprocessing_train_vocabulary_stable_geometry(backend,monkeypatch,tmp_path):
    if backend=='gpu':
        monkeypatch.setitem(sys.modules,'cupy',np)
        monkeypatch.setattr(prep,'resolve_backend',lambda _: 'gpu')
    frame=fixture_frame()
    frame.loc[6:,'conn_state']='ONLY_HOLDOUT'
    config=PipelineConfig(feature_policy='behavioral_strict',svd_components=0,
                          onehot_min_frequency=100,gpu_max_categories_per_col=1)
    X,bundle=prep.fit_transform_preprocessing(frame,splits(),EXPECTED,config,tmp_path/'one')
    assert bundle.numeric_cols==NUMERIC and bundle.categorical_cols==['conn_state']
    assert bundle.log_cols==NUMERIC
    assert {v.shape[1] for v in X.values()}=={11}
    np.testing.assert_array_equal(dense(X['val'])[:,-3:],np.zeros((3,3)))
    train_block=dense(X['train'])[:,-3:]
    assert set(np.unique(train_block))=={0,1} and np.all(train_block.sum(axis=1)==1)
    meta=json.loads((tmp_path/'one/artifacts/preprocessing_metadata.json').read_text())
    assert meta['conn_state_encoding']['vocabulary']==['REJ','S0','SF']
    changed=frame.copy()
    changed.loc[6:,'conn_state']='ANOTHER_UNKNOWN'
    changed.loc[6:,'src_bytes']='9999999'
    Y,other=prep.fit_transform_preprocessing(changed,splits(),EXPECTED,config,tmp_path/'two')
    np.testing.assert_allclose(dense(X['train']),dense(Y['train']))
    assert {v.shape[1] for v in Y.values()}=={11}
    second=json.loads((tmp_path/'two/artifacts/preprocessing_metadata.json').read_text())
    assert meta['conn_state_encoding']==second['conn_state_encoding']
    if backend=='gpu':
        assert not meta['legacy_ordinal_columns']
        assert 'conn_state_encoder' in other.preprocessor


def test_cpu_gpu_same_category_columns_in_actual_preprocessing_branch(monkeypatch,tmp_path):
    frame=fixture_frame()
    config=PipelineConfig(feature_policy='behavioral_strict',svd_components=0)
    cpu,a=prep.fit_transform_preprocessing(frame,splits(),EXPECTED,config,tmp_path/'cpu')
    monkeypatch.setitem(sys.modules,'cupy',np)
    monkeypatch.setattr(prep,'resolve_backend',lambda _: 'gpu')
    gpu,b=prep.fit_transform_preprocessing(frame,splits(),EXPECTED,config,tmp_path/'gpu')
    assert a.numeric_cols==b.numeric_cols==NUMERIC
    assert a.categorical_cols==b.categorical_cols==['conn_state']
    for split in splits():
        np.testing.assert_array_equal(dense(cpu[split])[:,-3:],dense(gpu[split])[:,-3:])


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_reduction_keeps_training_fit_independent_of_holdout(backend,monkeypatch,tmp_path):
    if backend=='gpu':
        from sklearn.decomposition import TruncatedSVD
        monkeypatch.setitem(sys.modules,'cupy',np)
        monkeypatch.setattr(prep,'resolve_backend',lambda _: 'gpu')
        monkeypatch.setattr(prep,'_make_gpu_reducer',lambda n,seed:(TruncatedSVD(n_components=n,random_state=seed),'TruncatedSVD'))
    frame=fixture_frame(120)
    config=PipelineConfig(feature_policy='behavioral_strict',svd_components=8)
    X,a=prep.fit_transform_preprocessing(frame,splits(120),EXPECTED,config,tmp_path/'first')
    frame.loc[60:,'conn_state']='EXTERNAL_ONLY'
    Y,b=prep.fit_transform_preprocessing(frame,splits(120),EXPECTED,config,tmp_path/'second')
    assert {v.shape[1] for v in X.values()}=={8}
    assert {v.shape[1] for v in Y.values()}=={8}
    np.testing.assert_allclose(a.svd.components_,b.svd.components_)
    np.testing.assert_allclose(X['train'],Y['train'])


def test_small_clustering_consumes_nominal_coordinates_with_k30(tmp_path):
    frame=fixture_frame(120)
    config=PipelineConfig(feature_policy='behavioral_strict',svd_components=0,selected_k=30,
                          cluster_n_init=1,cluster_max_iter=4)
    X,bundle=prep.fit_transform_preprocessing(frame,splits(120),EXPECTED,config,tmp_path)
    clusterer,labels,distances=_fit_cpu_minibatch_kmeans(X,config)
    assert clusterer.n_features_in_==11 and clusterer.n_clusters==30
    assert all(len(labels[k])==len(v) for k,v in splits(120).items())
    assert all(np.isfinite(v).all() for v in distances.values())


@pytest.mark.parametrize('stage',['prepare','cluster'])
def test_selected_features_records_declared_and_fitted_representation(stage,monkeypatch,tmp_path):
    frame=fixture_frame(120)
    monkeypatch.setattr(cli,'load_dataset',lambda *args,**kwargs: frame.copy())
    monkeypatch.setattr(cli,'create_splits',lambda *args: splits(120))
    monkeypatch.setattr(cli,'save_splits',lambda *args,**kwargs: None)
    monkeypatch.setattr(cli,'fit_predict_clustering',lambda *args: {'metrics':{}})
    config=PipelineConfig(feature_policy='behavioral_strict',svd_components=0,selected_k=30)
    cli.run_experiment(config,tmp_path,stage=stage)
    payload=json.loads((tmp_path/'selected_features.json').read_text())
    assert payload['feature_policy_version']==BEHAVIORAL_STRICT_VERSION
    assert payload['requested_features']==payload['found_features']==payload['selected_features']==EXPECTED
    assert payload['semantic_types']['conn_state']=='nominal'
    assert payload['feature_transformations']['src_bytes']['model_treatment']=='numeric'
    assert 'log1p' in payload['feature_transformations']['src_bytes']['transformations']
    if stage=='cluster':
        assert payload['transformation_status']=='fitted_on_train'
        assert payload['conn_state_encoding']['encoding']=='one_hot'
        assert payload['conn_state_encoding']['unknown']=='all_zero'
    else:
        assert payload['transformation_status']=='configured_not_fitted'
