import numpy as np
import pandas as pd
import pytest
from ids_pipeline.config import PipelineConfig
from ids_pipeline.splitting import create_splits,save_splits
from ids_pipeline.selection_budget import FrozenTrainingPopulation
from ids_pipeline.selection_methods import (METHODS,TrainStrata,allocate_budget,select_random,
    select_stratified_label,select_stratified_type,select_cluster_uniform,select_from_training_frame)


@pytest.fixture
def population(tmp_path):
    frame=pd.DataFrame(dict(ts=np.arange(240)+1700000000,label=np.arange(240)%2,
                            type=np.where(np.arange(240)%7==0,'rare','common')))
    config=PipelineConfig(split_strategy='temporal',timestamp_col='ts',timestamp_unit='s')
    splits=create_splits(frame,config);save_splits(splits,tmp_path,frame,config)
    with FrozenTrainingPopulation.load(tmp_path) as pop:
        yield pop,frame


@pytest.mark.parametrize('cap,w,b,expected',[
    ([10,10,10],[1,1,1],12,[4,4,4]),
    ([10,10,10],[1,1,1],5,[2,2,1]),
    ([10,10,10],[1,1,1],2,[1,1,0]),
    ([1,1,1000],[1,1,1],100,[1,1,98]),
    ([10000,1,1],[10000,1,1],3,[1,1,1]),
    ([0,1,8],[0,1,1],8,[0,1,7]),
    ([3,7,10],[1,1,1],19,[3,7,9]),
    ([3,7,10],[1,1,1],20,[3,7,10]),
    ([990,10],[990,10],100,[98,2]),
])
def test_known_quotas(cap,w,b,expected):
    np.testing.assert_array_equal(allocate_budget(cap,w,b),expected)


def test_allocator_randomized_capacity_conservation():
    rng=np.random.default_rng(4)
    for _ in range(300):
        capacities=rng.integers(0,10000,size=30)
        b=int(rng.integers(1,capacities.sum()+1))
        for weights in [np.ones(30),capacities,np.sqrt(capacities)]:
            q=allocate_budget(capacities,weights,b)
            assert q.sum()==b and (q>=0).all() and (q<=capacities).all()
            if b>=(capacities>0).sum():assert (q[capacities>0]>=1).all()


@pytest.mark.parametrize('cap,w,b',[
    ([1,2],[1,1],0),([1,2],[1,1],4),([1.2,3],[1,1],1),
    ([-1,3],[1,1],1),([1,2],[0,1],1),([1,2],[np.inf,1],1),
    ([1,2],[1,np.nan],1),([1,2],[1],1),([0,0],[0,0],1),
])
def test_allocator_invalid_inputs(cap,w,b):
    with pytest.raises(ValueError):allocate_budget(cap,w,b)


@pytest.mark.parametrize('method',METHODS)
def test_all_selectors_exact_real_unique_reproducible(population,method):
    pop,frame=population
    clusters=TrainStrata.from_values(pop,'cluster',pop.train_indices,np.arange(len(pop.train_indices))%30)
    for b in [1,30,100,len(pop.train_indices)-1,len(pop.train_indices),'full']:
        a=select_from_training_frame(pop,b,42,method,frame,clusters=clusters)
        again=select_from_training_frame(pop,b,42,method,frame,clusters=clusters)
        assert a.selection.selection_hash==again.selection.selection_hash
        ids=a.selection.indices
        assert len(ids)==(len(pop.train_indices) if b=='full' else b)
        assert len(ids)==len(np.unique(ids)) and np.isin(ids,pop.train_indices).all()
        if a.diagnostics['quotas']:assert sum(r['quota'] for r in a.diagnostics['quotas'])==len(ids)


@pytest.mark.parametrize('method',METHODS)
def test_information_isolation(population,method):
    pop,frame=population;n=len(pop.train_indices)
    clusters=TrainStrata.from_values(pop,'cluster',pop.train_indices,np.arange(n)%30)
    baseline=select_from_training_frame(pop,60,42,method,frame,clusters=clusters)
    mutated=frame.copy()
    mutated['label']=mutated['label'].astype(object)
    for column in ('label','type'):
        if method!='stratified_'+column:
            mutated[column]=mutated[column].sample(frac=1,random_state=99).to_numpy()
    mutated.loc[n:,['label','type']]='HOLDOUT_ONLY_NEW_CLASS'
    mutated['future_column']='do not read'
    after=select_from_training_frame(pop,60,42,method,mutated,clusters=clusters)
    assert baseline.selection.selection_hash==after.selection.selection_hash


def test_random_signature_cannot_receive_labels(population):
    pop,_=population
    with pytest.raises(TypeError):select_random(pop,20,42,labels=np.ones(len(pop.train_indices)))


def test_stratification_minimum_rare_class_and_not_balanced(population):
    pop,_=population;n=len(pop.train_indices)
    values=np.r_[np.zeros(n-1,dtype=int),1]
    strata=TrainStrata.from_values(pop,'label',pop.train_indices,values)
    result=select_stratified_label(pop,50,42,strata)
    assert [q['quota'] for q in result.diagnostics['quotas']]==[49,1]
    assert pop.train_indices[-1] in result.selection.indices
    with pytest.raises(ValueError):select_stratified_type(pop,50,42,strata)
    with pytest.raises(ValueError):select_cluster_uniform(pop,50,42,strata)


@pytest.mark.parametrize('method',METHODS)
def test_seed_can_change_selection_without_mutating_inputs(population,method):
    pop,frame=population
    clusters=TrainStrata.from_values(pop,'cluster',pop.train_indices,np.arange(len(pop.train_indices))%10)
    before=clusters.codes.copy();ids=pop.train_indices.copy();split_hash=pop.split_hash
    a=select_from_training_frame(pop,40,42,method,frame,clusters=clusters)
    b=select_from_training_frame(pop,40,99,method,frame,clusters=clusters)
    assert a.selection.selection_hash!=b.selection.selection_hash
    np.testing.assert_array_equal(clusters.codes,before);np.testing.assert_array_equal(pop.train_indices,ids)
    assert pop.split_hash==split_hash


def test_training_alignment_is_required(population):
    pop,_=population
    with pytest.raises(ValueError,match='exactly'):
        TrainStrata.from_values(pop,'label',pop.train_indices[::-1],np.ones(len(pop.train_indices)))
    with pytest.raises(ValueError,match='nonmissing'):
        TrainStrata.from_values(pop,'type',pop.train_indices,np.repeat(None,len(pop.train_indices)))


def test_sqrt_reduces_concentration_but_cannot_override_capacity():
    capacities=np.array([100000,1000,1000,1000])
    proportional=allocate_budget(capacities,capacities,1000)
    square_root=allocate_budget(capacities,np.sqrt(capacities),1000)
    assert square_root[0]<proportional[0]
    np.testing.assert_array_equal(allocate_budget([100000,1,1],np.sqrt([100000,1,1]),1000),[998,1,1])


def test_train_only_checkpoint_matches_frozen_preprocessing_and_clustering(tmp_path):
    import importlib.util
    from pathlib import Path
    from ids_pipeline.feature_policy import BEHAVIORAL_STRICT_FEATURES
    from ids_pipeline.preprocessing import fit_transform_preprocessing,apply_log1p,_select_log_cols
    from ids_pipeline.clustering import _fit_cpu_minibatch_kmeans
    spec=importlib.util.spec_from_file_location('selection_checkpoint',Path('scripts/validate_selection_methods.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rng=np.random.default_rng(9)
    frame=pd.DataFrame({c:rng.integers(0,10000,120) for c in BEHAVIORAL_STRICT_FEATURES if c!='conn_state'})
    frame['conn_state']=rng.choice(['SF','S0','REJ'],120)
    splits={'train':np.arange(72),'val':np.arange(72,90),'test':np.arange(90,120)}
    config=PipelineConfig(feature_policy='behavioral_strict',selected_k=30,svd_components=5,cluster_n_init=1,cluster_max_iter=3)
    expected,bundle=fit_transform_preprocessing(frame,splits,list(BEHAVIORAL_STRICT_FEATURES),config,tmp_path/'original')
    root=tmp_path/'checkpoint';root.mkdir()
    train=apply_log1p(frame.iloc[splits['train']],_select_log_cols(list(BEHAVIORAL_STRICT_FEATURES)))
    encoded,processor,_,_=module.fit_current_preprocessor(train,config,root)
    result,reducer=module.fit_current_reducer(encoded,config,root)
    np.testing.assert_allclose(result,expected['train'],atol=1e-10)
    a,labels,_=_fit_cpu_minibatch_kmeans(expected,config)
    b,actual=module.fit_current_clusterer(result,config)
    np.testing.assert_array_equal(labels['train'],actual)
    np.testing.assert_allclose(a.cluster_centers_,b.cluster_centers_)
    encoded._mmap.close();result._mmap.close()


@pytest.mark.parametrize('method',METHODS)
def test_cli_dispatches_budget_methods_and_never_calls_rf(population,method,tmp_path,monkeypatch):
    from ids_pipeline import cli
    from ids_pipeline.feature_policy import BEHAVIORAL_STRICT_FEATURES
    import json
    pop,df=population;df=df.copy()
    rng=np.random.default_rng(4)
    for c in BEHAVIORAL_STRICT_FEATURES:
        df[c]=rng.integers(0,10000,len(df)) if c!='conn_state' else rng.choice(['S0','SF'],len(df))
    config=PipelineConfig(split_strategy='temporal',timestamp_col='ts',timestamp_unit='s',
                          feature_policy='behavioral_strict',selected_k=30,svd_components=0,
                          selection_method=method,selection_budget=30,
                          extra={'frozen_splits_dir':str(pop.directory)})
    monkeypatch.setattr(cli,'load_dataset',lambda *a,**k:df.copy())
    def fake_clustering(X,frame,splits,features,c,out):
        return {'labels':{s:np.arange(len(idx))%30 for s,idx in splits.items()},'metrics':{}}
    monkeypatch.setattr(cli,'fit_predict_clustering',fake_clustering)
    monkeypatch.setattr(cli,'train_random_forest',lambda *a,**k:pytest.fail('RF is out of this validation'))
    out=tmp_path/'result'
    cli.run_experiment(config,out,stage='cluster')
    summary=json.loads((out/'selection_summary.json').read_text())
    diag=json.loads((out/'method_summary.json').read_text())
    assert summary['realized_budget']==30 and diag['method']==method
    assert not (out/'splits').exists()


@pytest.mark.parametrize('kwargs',[{'selected_k':29},{'feature_policy':'no_raw_ip_port'}])
def test_cluster_method_requires_frozen_feature_policy_and_k30(kwargs,tmp_path):
    from ids_pipeline.cli import run_experiment
    config=PipelineConfig(feature_policy='behavioral_strict',selected_k=30,selection_budget=10,
                          selection_method='cluster_sqrt',extra={'frozen_splits_dir':'unused'}).with_updates(**kwargs)
    with pytest.raises(ValueError,match='k=30'):run_experiment(config,tmp_path,stage='prepare')
