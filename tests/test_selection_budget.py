import json
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest

from ids_pipeline.config import PipelineConfig, load_config
from ids_pipeline.selection_budget import (FrozenTrainingPopulation, Selection, resolve_budget,
    validate_budget_config, hash_indices, prepare_fit_selection)
from ids_pipeline.splitting import create_splits, save_splits
from ids_pipeline.split_protocols import manifest
from ids_pipeline import supervised, cli
from ids_pipeline.representatives import build_cluster_representatives


@pytest.fixture
def case(tmp_path):
    df = pd.DataFrame(dict(ts=np.arange(40)+1700000000, label=np.arange(40)%2,
                           type=np.where(np.arange(40)%2,'attack','normal'), src_bytes=np.arange(40)+100),
                      index=np.arange(40)*3+10)
    config = PipelineConfig(split_strategy='temporal', timestamp_col='ts', timestamp_unit='s',
                            targets=['label','type'])
    splits = create_splits(df, config)
    directory = tmp_path/'frozen'
    save_splits(splits,directory,df,config)
    population = FrozenTrainingPopulation.load(directory)
    yield df, config.with_updates(extra={'frozen_splits_dir':str(directory)}), splits, population
    population.close()


@pytest.mark.parametrize('b',[1,24,'full'])
def test_budget_endpoints(case,b):
    _,_,_,pop=case
    ids=None if b in (24,'full') else pop.train_indices[:1]
    selection=Selection.accept(pop,b,42,ids)
    assert len(selection.indices)==resolve_budget(b,len(pop.train_indices))
    assert len(np.unique(selection.indices))==len(selection.indices)
    assert selection.report(pop)['realized_budget']==len(selection.indices)


@pytest.mark.parametrize('b',[0,-1,25,True,False,1.5,'1000','1k','FULL',None])
def test_invalid_budgets(case,b):
    with pytest.raises(ValueError): resolve_budget(b,len(case[3].train_indices))


def test_no_selector_no_silent_truncation(case):
    with pytest.raises(ValueError,match='no selector'): Selection.accept(case[3],3,42)


@pytest.mark.parametrize('ids',[[0,0],[0,24],[0,39],[-1,0],[0.0,1.0],[0,1,2]])
def test_invalid_ids_duplicates_holdout_and_size(case,ids):
    with pytest.raises(ValueError): Selection.accept(case[3],2,42,ids)


def test_equivalence_hash_order_and_immutable_ids(case,tmp_path):
    pop=case[3]
    a=Selection.accept(pop,len(pop.train_indices),42,pop.train_indices[::-1])
    b=Selection.accept(pop,'full',42)
    assert a.selection_hash==b.selection_hash
    assert a.selection_hash==Selection.accept(pop,'full',99).selection_hash
    with pytest.raises(ValueError): a.indices[0]=123
    a.save_summary(pop,tmp_path/'small.json')
    report=json.loads((tmp_path/'small.json').read_text())
    assert report['train_population_hash']==pop.train_population_hash
    assert 'indices' not in report and (tmp_path/'small.json').stat().st_size < 2048
    assert hash_indices(np.arange(10,dtype='int32'))==hash_indices(np.arange(10,dtype='int64'))


@pytest.mark.parametrize('update',[{'selection_seed':99},{'clustering_seed':99},{'model_seed':99}])
def test_seed_isolation_split_and_hash(case,update):
    df,config,splits,pop=case
    original=manifest(df,splits,config.for_stage('split'))['split_hash']
    changed=config.with_updates(**update)
    reused=create_splits(df,changed.for_stage('split'))
    assert manifest(df,reused,changed.for_stage('split'))['split_hash']==original==pop.split_hash
    for s in splits: np.testing.assert_array_equal(splits[s],reused[s])


def test_explicit_seeds_and_legacy_fallback():
    config=PipelineConfig(random_state=13,split_seed=21,clustering_seed=22,selection_seed=23,model_seed=24)
    for stage,value in [('split',21),('clustering',22),('selection',23),('model',24)]:
        assert config.for_stage(stage).random_state==value
        assert config.for_stage(stage).seed_for('model')==24
    old=PipelineConfig(random_state=17)
    assert all(old.seed_for(s)==17 for s in ['split','clustering','selection','model'])


@pytest.mark.parametrize('kwargs',[{'representative_strategy':'centroid'}, {'representative_strategy':'mixed'},
    {'use_representatives_for_supervised':True}, {'representatives_per_cluster':100}, {'boundary_per_cluster':1}])
def test_legacy_incompatibilities(kwargs):
    with pytest.raises(ValueError,match='legacy'): validate_budget_config(PipelineConfig(selection_budget=10,**kwargs))
    validate_budget_config(PipelineConfig(selection_budget=None,**kwargs))


def test_legacy_builder_rejects_budget_before_accessing_data(tmp_path):
    with pytest.raises(ValueError,match='Legacy representatives'):
        build_cluster_representatives(None,None,None,None,PipelineConfig(selection_budget='full'),'cpu',tmp_path)


def test_yaml_budget_and_seeds(tmp_path):
    path=tmp_path/'c.yaml'
    path.write_text('selection_budget: 100000\nselection_seed: 7\nsplit_seed: 42\nclustering_seed: 8\nmodel_seed: 9\n')
    config=load_config(path)
    assert config.selection_budget==100000 and config.seed_for('selection')==7
    validate_budget_config(config)


def test_same_rows_for_label_and_type_at_rf_fit(case,tmp_path,monkeypatch):
    df,config,splits,pop=case
    config=config.with_updates(selection_budget=3,model_seed=73)
    chosen=Selection.accept(pop,3,42,[9,2,17])
    X={s:np.column_stack([idx,idx+100]) for s,idx in splits.items()}
    fits=[]
    class FakeRF:
        def __init__(self,**kwargs): assert kwargs['random_state']==73
        def fit(self,x,y): fits.append((x.copy(),y.copy()));return self
    monkeypatch.setattr(supervised,'RandomForestClassifier',FakeRF)
    monkeypatch.setattr(supervised.joblib,'dump',lambda *a: None)
    monkeypatch.setattr(supervised,'_evaluate_split',lambda *a,**kw:{})
    monkeypatch.setattr(supervised,'_predict_decoded',lambda *a,**kw:np.repeat('normal',len(splits['test'])))
    monkeypatch.setattr(supervised,'_write_report_and_confusion',lambda *a:None)
    monkeypatch.setattr(supervised,'classification_report',lambda *a,**kw:'stub')
    monkeypatch.setattr(supervised,'_confusion_dataframe',lambda *a:pd.DataFrame())
    report=supervised.train_random_forest(X,df,splits,['src_bytes'],config,tmp_path,selection=chosen)
    assert len(fits)==2
    for x,y in fits: np.testing.assert_array_equal(x[:,0],chosen.indices);assert len(y)==3
    assert report['targets']['label']['selection_hash']==report['targets']['type']['selection_hash']==chosen.selection_hash


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_direct_fit_cannot_bypass_contract(backend):
    config=PipelineConfig(selection_budget=1)
    with pytest.raises(ValueError,match='verified real-row'):
        supervised._fit_model(np.ones((1,2)),np.array(['normal']),{},config,backend)
    with pytest.raises(ValueError,match='verified real-row'):
        supervised._fit_gpu_random_forest(np.ones((1,2)),np.array([0]),{},config)


def test_tampered_selection_fails_before_fit(case):
    _,config,splits,pop=case;config=config.with_updates(selection_budget=2)
    selected=Selection.accept(pop,2,42,[0,2])
    bad=replace(selected,indices=np.array([0,0]))
    with pytest.raises(ValueError): supervised._validate_fit_budget(np.ones((2,1)),[0,1],config,(pop,bad,splits))
    with pytest.raises(ValueError): supervised._validate_fit_budget(np.ones((1,1)),[0],config,(pop,selected,splits))
    with pytest.raises(ValueError): selected.validate(pop,2,99)


def test_bind_rejects_wrong_population(case):
    df,config,splits,pop=case
    changed=df.copy();changed.index=changed.index+1
    with pytest.raises(ValueError): pop.bind(changed,splits,config)


def test_corrupt_frozen_arrays_rejected(case):
    pop=case[3]
    values=np.load(pop.directory/'splits/val_indices.npy').copy();values[0]=0
    np.save(pop.directory/'splits/val_indices.npy',values)
    with pytest.raises(ValueError,match='hash mismatch'): FrozenTrainingPopulation.load(pop.directory)


def test_missing_frozen_path_blocks_before_loading(monkeypatch,tmp_path):
    monkeypatch.setattr(cli,'load_dataset',lambda *a,**k:pytest.fail('Must fail before dataset access'))
    with pytest.raises(ValueError,match='frozen_splits_dir'):
        cli.run_experiment(PipelineConfig(selection_budget=1000),tmp_path)


def test_fit_matrix_uses_train_positions_not_original_index_labels(case):
    _,config,splits,pop=case;config=config.with_updates(selection_budget=2)
    selected=Selection.accept(pop,2,42,[2,6])
    X={'train':np.arange(24)[:,None]}
    x,y=prepare_fit_selection(X,np.arange(24)*10,splits,pop,selected,config)
    np.testing.assert_array_equal(x[:,0],[2,6]);np.testing.assert_array_equal(y,[20,60])


def test_selection_seed_does_not_change_generated_group_split():
    df=pd.DataFrame(dict(g=np.arange(120)//2, label=np.arange(120)%2,
                         type=np.where(np.arange(120)%2,'attack','normal')))
    a=PipelineConfig(split_strategy='group_stratified',group_cols=['g'],split_seed=21,selection_seed=7)
    b=a.with_updates(selection_seed=999)
    x=create_splits(df,a.for_stage('split'));y=create_splits(df,b.for_stage('split'))
    assert manifest(df,x,a.for_stage('split'))['split_hash']==manifest(df,y,b.for_stage('split'))['split_hash']


def test_cli_routes_stage_seeds_without_touching_frozen_apis(case,monkeypatch,tmp_path):
    df,config,splits,_=case
    config=config.with_updates(extra={},split_seed=11,clustering_seed=22,selection_seed=33,model_seed=44)
    seen={}
    monkeypatch.setattr(cli,'load_dataset',lambda *a,**k:df.copy())
    def split_spy(frame,c): seen['split']=c.random_state;return splits
    monkeypatch.setattr(cli,'create_splits',split_spy)
    monkeypatch.setattr(cli,'save_splits',lambda *a,**k:None)
    monkeypatch.setattr(cli,'fit_transform_preprocessing',lambda *a:({},None))
    def cluster_spy(X,frame,s,features,c,out):seen['clustering']=c.random_state;return {'metrics':{}}
    monkeypatch.setattr(cli,'fit_predict_clustering',cluster_spy)
    cli.run_experiment(config,tmp_path/'output',stage='cluster')
    assert seen=={'split':11,'clustering':22}


def test_budget_cli_does_not_duplicate_split_arrays(case,monkeypatch,tmp_path):
    df,config,splits,_=case;config=config.with_updates(selection_budget=2)
    monkeypatch.setattr(cli,'load_dataset',lambda *a,**k:df.copy())
    monkeypatch.setattr(cli,'save_splits',lambda *a,**k:pytest.fail('Do not export frozen arrays per selection'))
    monkeypatch.setattr(cli,'fit_transform_preprocessing',lambda *a:pytest.fail('No selector, must stop before preprocessing'))
    cli.run_experiment(config,tmp_path/'prepare',stage='prepare')
    with pytest.raises(ValueError,match='no selector'):
        cli.run_experiment(config,tmp_path/'attempt',stage='all')
    assert not (tmp_path/'prepare/splits').exists()


def test_budgets_are_not_limited_to_planned_grid():
    for b in [1,137,1000,10000,100000,1000000,13575269,'full']:
        assert 1<=resolve_budget(b,13575269)<=13575269


def test_separate_target_runs_cannot_replace_experiment_selection(case,tmp_path):
    pop=case[3];path=tmp_path/'selection_summary.json'
    first=Selection.accept(pop,2,42,[0,1]);first.save_summary(pop,path)
    second=Selection.accept(pop,2,42,[2,3])
    with pytest.raises(ValueError,match='replace the selection'): second.save_summary(pop,path)
    assert json.loads(path.read_text())['selection_hash']==first.selection_hash
