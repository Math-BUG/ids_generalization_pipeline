import numpy as np
import pandas as pd
import pytest

from ids_pipeline.config import PipelineConfig
from ids_pipeline.splitting import create_splits, save_splits, group_overlap_report, temporal_per_class_report, temporal_split_report
from ids_pipeline.split_protocols import SplitValidationError, manifest, load_frozen_splits, row_validation, group_codes, ordered_bucket_split, targets


def fixture(n=120):
    return pd.DataFrame(dict(g=np.arange(n)//2, ts=np.arange(n)+1700000000,
                             label=np.arange(n)%2, type=np.where(np.arange(n)%2,'attack','normal')),
                        index=np.arange(n)*3+7)


def cfg(strategy='group_stratified', **kwargs):
    return PipelineConfig(split_strategy=strategy,group_cols=['g'],timestamp_col='ts',timestamp_unit='s',**kwargs)


def test_missing_group_column_fails():
    with pytest.raises(ValueError,match='Missing required group'):
        create_splits(fixture().drop(columns='g'),cfg())


@pytest.mark.parametrize('sizes', [[98,1,1],[40,40,20]])
def test_dominant_or_indivisible_groups_do_not_relax_tolerance(sizes):
    df=fixture(sum(sizes))
    df['g']=np.repeat(np.arange(len(sizes)),sizes)
    with pytest.raises(SplitValidationError) as caught: create_splits(df,cfg())
    assert not caught.value.diagnostics['feasible_found']
    assert caught.value.diagnostics['tolerance']==.02
    assert row_validation(caught.value.splits,len(df))['ok']
    if sizes[0]==98: assert caught.value.diagnostics['feasibility_status']=='proven_infeasible'
    else: assert caught.value.diagnostics['feasibility_status']=='search_did_not_find_feasible_solution'


def test_overlap_detected_and_delimiters_do_not_collide():
    df=pd.DataFrame({'g':['a||b','a','a||b'],'h':['c','b||c','c']})
    codes,ng=group_codes(df,['g','h'])
    assert ng==2 and codes[0]==codes[2] and codes[0]!=codes[1]
    assert not group_overlap_report(df,dict(train=np.array([0]),val=np.array([1]),test=np.array([2])),['g','h'])['ok']


def test_rare_class_report_and_impossible_coverage():
    df=fixture()
    df.loc[df.index[0],'type']='rare'
    with pytest.raises(SplitValidationError) as caught: create_splits(df,cfg())
    assert caught.value.diagnostics['class_group_support']['type']['rare']['groups']==1
    assert any('rare' in reason for reason in caught.value.diagnostics['proven_infeasibility_reasons'])


@pytest.mark.parametrize('strategy',['group_stratified','temporal','temporal_per_class'])
def test_reproducible_hash_roundtrip_and_row_conservation(strategy,tmp_path):
    df=fixture()
    config=cfg(strategy)
    a=create_splits(df,config); b=create_splits(df,config)
    assert manifest(df,a,config)['split_hash']==manifest(df,b,config)['split_hash']
    assert np.array_equal(np.sort(np.concatenate(list(a.values()))),np.arange(len(df)))
    save_splits(a,tmp_path,df,config)
    frozen=load_frozen_splits(tmp_path,df,config)
    for s in a: np.testing.assert_array_equal(a[s],frozen[s])
    changed=df.copy(); changed.index=changed.index+1
    with pytest.raises(ValueError,match='mismatch'): load_frozen_splits(tmp_path,changed,config)
    bad=a['train'].copy(); bad[0]=bad[1]; np.save(tmp_path/'splits/train_indices.npy',bad)
    with pytest.raises(ValueError): load_frozen_splits(tmp_path,df,config)


@pytest.mark.parametrize('freq',['1s',None])
def test_tied_timestamps_never_split(freq):
    df=fixture(120); df['ts']=1700000000+np.arange(120)//20
    config=cfg('temporal').with_updates(temporal_bucket_freq=freq)
    split=create_splits(df,config)
    assert temporal_split_report(df,split,config)['global_chronology_preserved']


def test_giant_bucket_preserved_with_real_proportion_report():
    df=fixture(100);df['ts']=1700000000+np.repeat([0,1,2],[98,1,1])
    config=cfg('temporal'); split=create_splits(df,config)
    report=manifest(df,split,config)
    assert report['validation_ok']
    assert report['counts']==dict(train=98,val=1,test=1)
    assert report['proportions']['train']==.98


@pytest.mark.parametrize('strategy',['temporal','temporal_per_class'])
def test_few_buckets_no_row_fallback(strategy):
    df=fixture();df['ts']=1700000000+np.arange(len(df))%2
    with pytest.raises(ValueError): create_splits(df,cfg(strategy))


def test_per_class_valid_but_global_invalid_and_missing_validation():
    df=fixture(120);df['ts']=1700000000+np.arange(120)+np.where(df['type']=='attack',1000,0)
    config=cfg('temporal_per_class');split=create_splits(df,config)
    report,_=temporal_per_class_report(df,split,config)
    assert report['ok'] and not report['global_chronology_preserved']
    absent=split.copy();moved=absent['val'][df['type'].iloc[absent['val']].to_numpy()=='attack']
    absent['val']=np.setdiff1d(absent['val'],moved);absent['test']=np.sort(np.r_[absent['test'],moved])
    report,_=temporal_per_class_report(df,absent,config)
    assert not report['ok'] and report['classes_missing_val']==['attack']


def test_temporal_first_seen_and_small_support():
    df=fixture(100);df['type']=np.repeat(['normal','val_only','test_only'],[60,15,25])
    config=cfg('temporal');report=manifest(df,create_splits(df,config),config)
    d=report['distributions']['type']
    assert d['first_seen_validation']==['val_only']
    assert d['first_seen_test']==['test_only']
    assert d['unseen_from_train_in_test']==['test_only']
    assert 'test_only' in d['splits']['val']['missing']
    assert d['splits']['val']['small_support']=={'val_only':15}


def test_disjoint_but_reversed_temporal_fails():
    df=fixture();config=cfg('temporal');split=create_splits(df,config)
    split['train'],split['test']=split['test'],split['train']
    assert not temporal_split_report(df,split,config)['ok']


@pytest.mark.parametrize('invalid',[None,'invalid'])
def test_missing_invalid_timestamp_fails(invalid):
    df=fixture();df['ts']=df['ts'].astype(object);df.iloc[0,df.columns.get_loc('ts')]=invalid
    with pytest.raises(ValueError,match='valid timestamps'): create_splits(df,cfg('temporal'))


def test_row_checks_detect_duplicate_loss_and_overlap():
    report=row_validation(dict(train=np.array([0,0]),val=np.array([0]),test=np.array([8])),4)
    assert not report['ok'] and report['duplicates_within_splits']==1
    assert report['overlapping_rows']==1 and report['missing_rows']==3 and report['out_of_range']==1


def test_bucket_cut_cost_matches_exhaustive_search():
    rng=np.random.default_rng(7);config=cfg('temporal')
    for _ in range(30):
        sizes=rng.integers(1,100,12);bucket=pd.Series(pd.to_datetime(np.repeat(np.arange(12),sizes),unit='s',utc=True))
        split=ordered_bucket_split(bucket,np.arange(len(bucket)),config)
        actual=sum(abs(len(split[s])-len(bucket)*targets(config)[i]) for i,s in enumerate(('train','val','test')))
        expected=min(sum(abs(np.array([sizes[:i].sum(),sizes[i:j].sum(),sizes[j:].sum()])-sizes.sum()*targets(config))) for i in range(1,11) for j in range(i+1,12))
        assert actual==pytest.approx(expected)


def test_loader_rejects_configuration_and_content_changes(tmp_path):
    df=fixture();config=cfg();split=create_splits(df,config)
    save_splits(split,tmp_path,df,config)
    with pytest.raises(ValueError,match='mismatch'):
        load_frozen_splits(tmp_path,df,config.with_updates(random_state=99))
    changed=df.copy();changed.loc[changed.index[0],'type']='changed'
    with pytest.raises(ValueError,match='mismatch'):
        load_frozen_splits(tmp_path,changed,config)
    with pytest.raises(ValueError,match='overwrite'):
        save_splits(split,tmp_path,df,config.with_updates(random_state=99))


def test_create_consumes_frozen_arrays_without_search(tmp_path,monkeypatch):
    import ids_pipeline.splitting as api
    df=fixture();config=cfg();split=create_splits(df,config)
    save_splits(split,tmp_path,df,config)
    def forbidden(*a,**kw): raise AssertionError('Search must not run')
    monkeypatch.setattr(api,'group_split',forbidden)
    reused=create_splits(df,config.with_updates(extra={'frozen_splits_dir':str(tmp_path)}))
    for s in split: np.testing.assert_array_equal(split[s],reused[s])


def test_datetime_utc_and_string_timestamps_match():
    df=fixture();config=cfg('temporal').with_updates(timestamp_unit='auto')
    df['ts']=pd.to_datetime(df['ts'],unit='s',utc=True)
    first=create_splits(df,config)
    df['ts']=df['ts'].astype(str)
    second=create_splits(df,config)
    for s in first: np.testing.assert_array_equal(first[s],second[s])


@pytest.mark.parametrize('missing_split',['val','test'])
def test_absent_per_class_holdout_invalidates_report(missing_split):
    df=fixture();config=cfg('temporal_per_class');split=create_splits(df,config)
    moved=split[missing_split][df['type'].iloc[split[missing_split]].to_numpy()=='attack']
    split[missing_split]=np.setdiff1d(split[missing_split],moved)
    split['train']=np.sort(np.r_[split['train'],moved])
    report,_=temporal_per_class_report(df,split,config)
    assert not report['ok'] and report[f'classes_missing_{missing_split}']==['attack']


def test_per_class_lists_all_infeasible_classes():
    df=fixture();df['ts']=1700000000
    with pytest.raises(SplitValidationError) as caught: create_splits(df,cfg('temporal_per_class'))
    assert set(caught.value.diagnostics['infeasible_classes'])=={'attack','normal'}
    assert caught.value.splits is None


def test_population_id_original_count_and_positional_identity():
    df=fixture();df.attrs['data_quality_population_id']='frozen-id'
    df.attrs['data_quality_report']={'original_rows':130}
    config=cfg('temporal');split=create_splits(df,config);report=manifest(df,split,config)
    assert report['population']['original_rows']==130
    assert report['population']['eligible_input_rows']==120
    assert report['population']['source_population_id']=='frozen-id'
    assert report['identity_hashes']['train']!=report['index_hashes']['train']


def test_null_group_and_literal_null_are_distinct():
    df=pd.DataFrame({'g':[None,'<NA>',None]})
    codes,ng=group_codes(df,['g'])
    assert ng==2 and codes[0]==codes[2] and codes[0]!=codes[1]
