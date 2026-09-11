"""Phase 2.1 fixtures only. GPU doubles below are not CUDA validation."""
import copy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans as CPUKMeans

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import phase_2_1_common as C
import prepare_phase_2_1_clustering as P
import run_phase_2_1_clustering as R
import report_phase_2_1_clustering as S
import evaluate_phase_2_1_external as E
from test_phase_1e_real_gpu_smoke import gpu_doubles, gpu_array
from ids_pipeline.preprocessing import PreprocessingBundle, _gpu_numeric_arrays, apply_log1p
from ids_pipeline.nominal_encoding import ConnStateOneHot


class FakeSVD:
    fit_calls=[]
    def __init__(self,n_components,random_state=42):
        self.n_components=n_components
        self.random_state=random_state
    def get_params(self,deep=False):
        return dict(n_components=self.n_components,random_state=self.random_state)
    def fit(self,X):
        type(self).fit_calls.append((self.n_components,len(X)))
        self.components_=gpu_array(np.eye(X.shape[1],dtype=np.float32)[:self.n_components])
        return self
    def transform(self,X):
        return gpu_array(X@self.components_.T,dtype=np.float32)


class FakeKMeans:
    fit_calls=[]
    def __init__(self,n_clusters,random_state,**params):
        self.params=dict(n_clusters=n_clusters,random_state=random_state,**params)
    def get_params(self,deep=False):
        return self.params.copy()
    def fit(self,X):
        p=self.params
        type(self).fit_calls.append((p['n_clusters'],p['random_state'],len(X)))
        model=CPUKMeans(n_clusters=p['n_clusters'],random_state=p['random_state'],n_init=1,max_iter=p['max_iter']).fit(np.asarray(X))
        self.cluster_centers_=gpu_array(model.cluster_centers_,dtype=np.float32)
        self.labels_=gpu_array(model.labels_,dtype=np.int32)
        self.inertia_,self.n_iter_=float(model.inertia_),int(model.n_iter_)
        return self
    def predict(self,X):
        distances=((np.asarray(X)[:,None,:]-self.cluster_centers_[None,:,:])**2).sum(axis=2)
        return gpu_array(distances.argmin(axis=1),dtype=np.int32)


def config():
    return C.load_config()


def test_frozen_config_and_flags():
    c=config()
    assert len(c['representations'])*len(c['k_values'])*len(c['clustering_seeds'])==96
    assert c['degeneracy_flags']['automatically_exclude_from_pareto'] is False
    assert c['degeneracy_flags']['thresholds']['max_cluster_fraction']==0.9


@pytest.mark.parametrize('key,value',[
    ('clustering_seeds',[42]),('k_values',[30]),('compute_backend','cpu'),
    ('representations',{'R20':20}),('reducer_seed',123),('export_cluster_assignments',True),
    ('expected_eligible_rows',5),('selection_budget',1000),('label','type')])
def test_config_mutations_rejected(key,value):
    c=config(); c[key]=value
    with pytest.raises(ValueError,match='Frozen'):
        C.validate_config(c)


def tiny_sample_config():
    c=config()
    c['development_sample']['size']=60
    c['metric_sample_a']['size']=20
    c['metric_sample_b']['size']=20
    return c


def test_sample_hashes_identity_and_reproducibility(tmp_path):
    c=tiny_sample_config()
    train=np.arange(120,dtype=np.int64)[::-1]
    original=np.arange(170,dtype=np.int64)*3+100
    a=C.make_samples(train,original,c)
    changed=copy.deepcopy(c); changed['clustering_seeds']=[7,8,9]
    b=C.make_samples(train,original,changed)
    for key in a:
        np.testing.assert_array_equal(a[key],b[key])
    assert len(np.unique(a['development_indices']))==60
    assert set(a['metric_sample_indices'])<=set(a['development_indices'])<=set(train)
    np.testing.assert_array_equal(a['development_original_ids'],original[a['development_indices']])
    np.testing.assert_array_equal(a['metric_sample_indices'],a['development_indices'][a['metric_offsets']])
    assert (np.diff(a['development_train_offsets'])>0).all()
    hashes={key:C.array_hash(a[key]) for key in ('development_indices','metric_sample_indices')}
    assert hashes['development_indices']!=hashes['metric_sample_indices']
    # Exact known hashes provide stable review evidence, not merely equality to self.
    expected=json.loads((Path(__file__).parent/'phase_2_1_sample_hashes.json').read_text())
    assert hashes==expected


@pytest.mark.parametrize('counts',[[0,10,20],[1,1,1],[99,1,0],[1000]*10])
def test_concentration_definitions(counts):
    s=C.concentration(counts)
    p=np.asarray(counts)/sum(counts)
    H=-sum(x*np.log(x) for x in p if x)
    assert s['entropy']==pytest.approx(H)
    assert s['normalized_entropy']==pytest.approx(H/np.log(len(counts)))
    assert s['effective_number_of_clusters']==pytest.approx(np.exp(H))
    assert s['std_size']==pytest.approx(np.std(counts,ddof=0))
    assert s['clusters_lt10']==sum(n<10 for n in counts)


def test_flags_do_not_invalidate_or_filter_and_top10_context():
    c=config()
    stats=C.concentration([990001]+[1]*9)
    flags=C.degeneracy_flags(stats,c['degeneracy_flags'])
    assert flags['dominant_cluster']['triggered'] is True
    assert flags['microclusters_lt100']['triggered'] is True
    assert flags['top10_concentration']['state']=='not_applicable'
    row=dict(configuration='R0:k10',mean_silhouette=.8,mean_davies_bouldin=.3,
        mean_calinski_harabasz=100,min_pairwise_stability_ami=.9,worst_max_cluster_fraction=.999)
    assert C.pareto_front([row])==['R0:k10']


def test_missing_cluster_in_metric_sample_is_allowed():
    X=np.array([[0,0],[0,.1],[4,4],[4,4.1],[9,9],[9,9.1]],dtype=np.float32)
    labels=np.array([0,0,1,1,2,2])
    C.cluster_counts(labels,6,3,require_all=True)
    metrics=C.internal_metrics(X,labels,np.arange(4),full_geometry=False)
    assert metrics['sample_observed_clusters']==2
    assert metrics['silhouette'] is not None
    one=C.internal_metrics(X,labels,np.arange(2),full_geometry=False)
    assert one['silhouette'] is None  # undefined metric, not incompatible fit q
    with pytest.raises(ValueError,match='Observed fit'):
        C.cluster_counts(np.array([0,0,1,1]),4,3,require_all=True)


@pytest.mark.parametrize('labels,n,k',[(np.array([0,1]),3,2),(np.array([0,2]),2,2),
                                     (np.array([0.,1.]),2,2),(np.array([-1,0]),2,2)])
def test_invalid_assignments(labels,n,k):
    with pytest.raises(ValueError): C.cluster_counts(labels,n,k)


def test_stability_invariant_to_cluster_id_permutation():
    a=np.array([0,0,1,1,2,2])
    s=C.stability({42:a,123:(a+1)%3,2026:2-a})
    assert s['min_pairwise_stability_ami']==pytest.approx(1)
    with pytest.raises(ValueError): C.stability({42:a,123:a})


def test_pareto_tradeoff_ties_and_external_column_rejection():
    def row(name,sil,db):
        return dict(configuration=name,mean_silhouette=sil,mean_davies_bouldin=db,
            mean_calinski_harabasz=100,min_pairwise_stability_ami=.8,worst_max_cluster_fraction=.5)
    rows=[row('a',.8,.5),row('b',.7,.4),row('c',.7,.7),row('tie',.8,.5)]
    assert C.pareto_front(rows)==['a','b','tie']
    for key in ('label','type','ami_type','v_measure','purity'):
        with pytest.raises(ValueError,match='registered internal'):
            C.pareto_front([{**rows[0],key:1}])
    with pytest.raises(ValueError,match='Nonfinite'):
        C.pareto_front([row('nan',float('nan'),.4)])


@pytest.fixture
def tiny_pipeline(tmp_path,monkeypatch,gpu_doubles):
    c=config()
    c.update(output_dir=str(tmp_path/'output'),parent_baseline_dir=str(tmp_path/'parent'),
        data_path=str(tmp_path/'data'),frozen_splits_dir=str(tmp_path/'frozen'))
    c['expected_eligible_rows']=260
    c['expected_split_rows']={'train':180,'val':40,'test':40}
    c['development_sample']['size']=90
    c['metric_sample_a']['size']=30
    c['metric_sample_b']['size']=30
    c['k_values']=[10]  # 12 tiny A runs, never the real 96-run matrix
    monkeypatch.setattr(C,'validate_config',lambda _:None)
    monkeypatch.setattr(C,'verify_provenance',lambda _:dict(digest='fixture_source',git_commit_sha='a'*40))
    env=dict(python_version='test',cuml_version='test',cupy_version='test',cudf_version='test',
             software={'numpy':np.__version__,'scikit-learn':'test'})
    monkeypatch.setattr(C,'gpu_environment',lambda _:env)
    original_class=C.gpu_class
    def fake_class(model,kind):
        if isinstance(model,FakeKMeans) and kind=='KMeans': return 'cuml.cluster.fixture.KMeans'
        if isinstance(model,FakeSVD) and kind=='TruncatedSVD': return 'cuml.decomposition.fixture.TruncatedSVD'
        return original_class(model,kind)
    monkeypatch.setattr(C,'gpu_class',fake_class)
    monkeypatch.setitem(sys.modules,'cuml.decomposition',SimpleNamespace(TruncatedSVD=FakeSVD))
    monkeypatch.setitem(sys.modules,'cuml.cluster',SimpleNamespace(KMeans=FakeKMeans))
    rng=np.random.default_rng(123)
    frame=pd.DataFrame({col:rng.integers(0,1000,260).astype(float) for col in C.QUANTITATIVE})
    states=['OTH','REJ','RSTO','RSTOS0','RSTR','RSTRH','S0','S1','S2','S3','SF','SH','SHR']
    frame['conn_state']=[states[i%13] for i in range(260)]
    frame['label']=np.arange(260)%2
    frame['type']=np.where(frame.label,'attack','normal')
    frame.index=np.arange(260)+1000
    splits={'train':np.arange(180),'val':np.arange(180,220),'test':np.arange(220,260)}
    train=apply_log1p(frame.iloc[:180][list(C.BEHAVIORAL_STRICT_FEATURES)],C.QUANTITATIVE)
    numeric,_,_,stats=_gpu_numeric_arrays(train,train.iloc[:1],train.iloc[:1],C.QUANTITATIVE)
    encoder=ConnStateOneHot().fit(train[['conn_state']])
    matrix=gpu_array(np.concatenate([numeric,encoder.transform_gpu(train[['conn_state']])],axis=1))
    svd=FakeSVD(20).fit(matrix)
    bundle=PreprocessingBundle(list(C.BEHAVIORAL_STRICT_FEATURES),C.QUANTITATIVE,['conn_state'],C.QUANTITATIVE,
        dict(numeric_stats=stats,conn_state_encoder=encoder,conn_state_encoding=encoder.metadata()),svd,'gpu')
    parent=Path(c['parent_baseline_dir']); (parent/'artifacts').mkdir(parents=True)
    joblib.dump(bundle,parent/'artifacts/preprocessing_bundle.joblib')
    baseline=FakeKMeans(n_clusters=30,random_state=42,**c['kmeans']).fit(svd.transform(matrix))
    joblib.dump(baseline,parent/'artifacts/cuml_kmeans.joblib')
    p=dict(status='APPROVED',experiment_id=c['parent_experiment_id'],code_snapshot_hash='parent_fixture',
        data={'population_id':c['expected_population_id']},split={'observed_hash':c['expected_split_hash']},
        features={'selected_features':list(C.BEHAVIORAL_STRICT_FEATURES),'feature_policy_version':C.BEHAVIORAL_STRICT_VERSION},
        clustering={'parameters_effective':{**c['kmeans'],'n_clusters':30,'random_state':42}},
        software={**env,'numpy_version':np.__version__,'packages':{'scikit-learn':'test'}},
        preprocessing=dict(conn_state_encoding=encoder.metadata(),encoded_dimension_before_reduction=21,svd_components_effective=20))
    C.atomic_json(parent/'experiment_manifest.json',p)
    C.atomic_json(parent/'metrics_clustering.json',dict(silhouette_train=.5,davies_bouldin_train=1,
        calinski_harabasz_train=10,inertia=100))
    def loader(cfg,out,report,profiler):
        report['data']={'population_id':c['expected_population_id']}
        return frame,splits
    import run_phase_2_full_gpu_baseline as parent_runner
    monkeypatch.setattr(parent_runner,'load_population',loader)
    FakeSVD.fit_calls.clear(); FakeKMeans.fit_calls.clear()
    return c,frame,splits,bundle,matrix


def test_replay_same_training_statistics_and_unknowns(tiny_pipeline):
    c,frame,splits,bundle,matrix=tiny_pipeline
    before=copy.deepcopy(bundle.preprocessor['numeric_stats'])
    actual=C.replay_preprocessing(frame.iloc[:180][bundle.feature_cols],bundle)
    np.testing.assert_array_equal(actual,matrix)
    holdout=frame.iloc[180:][bundle.feature_cols].copy()
    holdout['src_bytes']=1e9
    holdout['conn_state']='new_state'
    out=C.replay_preprocessing(holdout,bundle)
    assert (out[:,8:]==0).all()
    assert before==bundle.preprocessor['numeric_stats']
    with pytest.raises(ValueError,match='only ordered feature'):
        C.replay_preprocessing(frame,bundle)


def test_end_to_end_preparation_stages_decisions_and_external_gate(tiny_pipeline,monkeypatch):
    c,frame,splits,bundle,matrix=tiny_pipeline
    import ids_pipeline.supervised as supervised
    monkeypatch.setattr(supervised,'train_random_forest',lambda *a,**k:pytest.fail('RF forbidden'))
    prepared=P.prepare(c)
    assert FakeSVD.fit_calls==[(10,180),(15,180)]
    assert FakeKMeans.fit_calls==[]
    root=Path(c['output_dir'])
    assert C.load_prepared(c)['digest']==prepared['digest']
    with pytest.raises(FileNotFoundError): E.external_gate(c)
    for representation in c['representations']:
        R.run(c,'A',representation)
    assert len(FakeKMeans.fit_calls)==12
    evidence=S.summarize(c,'A',prepared)
    assert evidence['recorded_runs']==12 and evidence['successful_runs']==12
    assert all(row['flagged_seeds'] for row in evidence['configurations'])
    pick=evidence['pareto'][0]
    with pytest.raises(ValueError,match='Explicit'):
        S.freeze(c,'A',[pick],'reviewer','internal tradeoffs',False)
    a=S.freeze(c,'A',[pick],'reviewer','internal tradeoffs',True)
    with pytest.raises(FileNotFoundError): E.external_gate(c)
    with pytest.raises(ValueError,match='frozen'):
        S.freeze(c,'A',[pick],'reviewer','changed',True)
    R.run(c,'B',finalist_index=0)
    assert len(FakeKMeans.fit_calls)==15
    assert FakeSVD.fit_calls==[(10,180),(15,180)]
    ev=S.summarize(c,'B',prepared)
    b=S.freeze(c,'B',[ev['pareto'][0]],'reviewer','B internal evidence only',True)
    assert E.external_gate(c)[2]['digest']==b['digest']
    # Post-freeze reporting must not mutate immutable evidence or decisions.
    assert S.summarize(c,'A',prepared)['digest']==evidence['digest']
    assert C.verify_decision(c,'A')['digest']==a['digest']
    with pytest.raises(ValueError,match='frozen'):
        R.run(c,'B',finalist_index=0)
    E.evaluate(c)
    assert (root/'stage_b/external_metrics.csv').is_file()
    assert len(pd.read_csv(root/'stage_b/external_metrics.csv'))==4*3*2
    assert C.read_json(root/C.MANIFEST)['status']=='COMPLETE'
    assert not list(root.rglob('cluster_assignments.csv'))
    assert len(FakeKMeans.fit_calls)==15  # external reporting has zero fits
    record=root/f"stage_b/runs/{C.run_id(b['configurations'][0]['representation'],10,42)}.json"
    value=C.read_json(record); value['seed']=999
    C.atomic_json(record,value)
    with pytest.raises(ValueError,match='digest'):
        E.external_gate(c)


def test_corrupt_sample_and_nan_wrong_backend_guards(tiny_pipeline):
    c,*_=tiny_pipeline
    prepared=P.prepare(c)
    with pytest.raises(ValueError,match='CuPy'):
        C.gpu_matrix(np.zeros((3,21),dtype=np.float32))
    with pytest.raises(ValueError,match='NaN/Inf'):
        C.gpu_matrix(gpu_array([[float('inf')]],dtype=np.float32))
    with pytest.raises(ValueError,match='Wrong GPU'):
        C.gpu_class(CPUKMeans(),'KMeans')
    path=Path(c['output_dir'])/'metric_sample_indices.npy'
    with path.open('ab') as stream: stream.write(b'corruption')
    with pytest.raises(ValueError,match='truncated'):
        C.load_prepared(c)


@pytest.mark.parametrize('failure',['execution','centers','rows','clusters','inertia'])
def test_invalid_runs_are_recorded_and_not_silently_retried(tiny_pipeline,monkeypatch,failure):
    c,*_=tiny_pipeline
    prepared=P.prepare(c)
    original=FakeKMeans.fit
    def broken(self,X):
        if failure=='execution': raise RuntimeError('GPU failure fixture')
        original(self,X)
        if failure=='centers': self.cluster_centers_[0,0]=np.nan
        if failure=='rows': self.labels_=self.labels_[:-1]
        if failure=='clusters': self.labels_=gpu_array(np.zeros(len(X),dtype=np.int32))
        if failure=='inertia': self.inertia_=float('inf')
        return self
    monkeypatch.setattr(FakeKMeans,'fit',broken)
    with pytest.raises((ValueError,RuntimeError)):
        R.run(c,'A','R0')
    record=C.check_seal(C.read_json(Path(c['output_dir'])/'stage_a/runs/R0_k10_s42.json'))
    assert record['validity']['status']=='INVALID'
    with pytest.raises(ValueError,match='retry-invalid'):
        R.run(c,'A','R0')
    summary=S.summarize(c,'A',prepared)
    assert summary['successful_runs']==0 and summary['pareto']==[]
    with pytest.raises(ValueError,match='All planned'):
        S.freeze(c,'A',['R0:k10'],'reviewer','internal',True)
    if failure=='execution':
        monkeypatch.setattr(FakeKMeans,'fit',original)
        R.run(c,'A','R0',retry_invalid=True)
        archived=list((Path(c['output_dir'])/'stage_a/attempts').rglob('previous_attempt_record.json'))
        assert len(archived)==1 and C.read_json(archived[0])['validity']['status']=='INVALID'


def test_external_evaluate_cannot_open_dataset_before_decisions(tiny_pipeline,monkeypatch):
    c,*_=tiny_pipeline
    P.prepare(c)
    import run_phase_2_full_gpu_baseline as original
    monkeypatch.setattr(original,'load_population',lambda *a:pytest.fail('Premature label access'))
    with pytest.raises(FileNotFoundError): E.evaluate(c)
    assert not (Path(c['output_dir'])/'stage_b/external_evaluation').exists()


def test_flags_threshold_boundaries():
    policy=config()['degeneracy_flags']
    stats=C.concentration([900]+[10]*10)
    flags=C.degeneracy_flags(stats,policy)
    assert flags['dominant_cluster']['value']==.9 and flags['dominant_cluster']['triggered'] is True
    stats['normalized_entropy']=.5
    stats['effective_clusters_ratio']=.25
    flags=C.degeneracy_flags(stats,policy)
    assert flags['low_entropy']['triggered'] is False
    assert flags['low_effective_ratio']['triggered'] is False


def test_promotion_limits_and_unknown_candidates(tiny_pipeline):
    c,*_=tiny_pipeline
    prepared=P.prepare(c)
    # Shape of a reviewed complete evidence manifest; candidate guards must fail
    # before any decision is written. No artificial scientific run is executed.
    evidence=C.sealed(dict(preparation_digest=prepared['digest'],run_digests={},
        recorded_runs=0,expected_runs=0,configurations=[],pareto=[]))
    C.atomic_json(Path(c['output_dir'])/'stage_a/internal_evidence.json',evidence)
    for candidates in ([],['R0:k10']*2,['a','b','c','d'],['unknown']):
        with pytest.raises(ValueError):
            S.freeze(c,'A',candidates,'reviewer','internal',True)
    assert not C.decision_path(c['output_dir'],'A').exists()


def test_source_provenance_reproducible_and_tampering_rejected(tmp_path,monkeypatch):
    c=config()
    c['source_provenance_path']=str(tmp_path/'source.json')
    monkeypatch.setattr(C.subprocess,'check_output',lambda args,**kwargs: 'a'*40 if 'rev-parse' in args else '')
    files={'code.py':'sha256:original'}
    monkeypatch.setattr(C,'source_hashes',lambda:files.copy())
    first=C.write_provenance(c)
    assert first['digest']==C.write_provenance(c)['digest']
    assert C.verify_provenance(c)==first
    files['code.py']='sha256:changed'
    with pytest.raises(ValueError,match='snapshot'):
        C.verify_provenance(c)


def test_flags_are_bound_by_scientific_config_hash():
    c=config()
    c['degeneracy_flags']['thresholds']['max_cluster_fraction']=0.99
    with pytest.raises(ValueError,match='Frozen'):
        C.validate_config(c)


def test_pca_reducer_is_rejected_even_if_gpu_namespace():
    PCA=type('PCA',(),{'__module__':'cuml.decomposition.pca'})
    with pytest.raises(ValueError,match='Wrong GPU implementation'):
        C.gpu_class(PCA(),'TruncatedSVD')
