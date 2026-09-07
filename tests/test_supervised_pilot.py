import importlib.util
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score,precision_recall_curve,auc
from ids_pipeline.config import PipelineConfig
from ids_pipeline.splitting import create_splits,save_splits
from ids_pipeline.selection_budget import FrozenTrainingPopulation,Selection,prepare_fit_selection


@pytest.fixture
def pilot():
    spec=importlib.util.spec_from_file_location('pilot',Path('scripts/run_supervised_pilot.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_ap_is_average_precision_not_trapezoidal_area(pilot):
    y=np.array([0,1,0,1]);scores=np.array([.1,.2,.3,.4])
    result=pilot.metric_payload(y,np.array([0,0,1,1]),['0','1'],scores)
    precision,recall,_=precision_recall_curve(y,scores)
    assert result['average_precision']==average_precision_score(y,scores)
    assert result['average_precision']!=auc(recall,precision)
    assert result['positive_label']=='1'
    assert 0<=result['fpr_at_tpr_95']<=1


def test_missing_prediction_class_retains_confusion_and_support(pilot):
    result=pilot.metric_payload(np.array([0,0,1,2]),np.array([0,0,0,0]),['a','b','rare'])
    assert np.asarray(result['confusion_matrix']).shape==(3,3)
    assert result['per_class']['rare']['support']==1
    assert result['per_class']['rare']['recall']==result['per_class']['rare']['f1-score']==0
    assert result['classes_present']==['a','b','rare']


def test_same_ids_reach_real_rf_for_both_targets_and_chunked_metrics(pilot,tmp_path):
    frame=pd.DataFrame(dict(ts=np.arange(120)+1700000000,label=np.arange(120)%2,type=np.where(np.arange(120)%3,'b','a')))
    config=PipelineConfig(split_strategy='temporal',timestamp_col='ts',timestamp_unit='s',
                          selection_budget=20,selection_seed=42,model_seed=42,random_forest_estimators=3,n_jobs=1)
    splits=create_splits(frame,config);save_splits(splits,tmp_path,frame,config)
    with FrozenTrainingPopulation.load(tmp_path) as pop:
        selected=Selection.accept(pop,20,42,pop.train_indices[:20])
        X={s:np.column_stack([idx,idx%2]).astype(np.float32) for s,idx in splits.items()}
        original={s:arr.copy() for s,arr in X.items()}
        seen=[]
        for target in ('label','type'):
            y=(splits['train']%2).astype(np.int8)
            xs,ys=prepare_fit_selection(X,y,splits,pop,selected,config)
            seen.append(xs[:,0].copy())
            bundle,seconds=pilot.fit_timed(xs,ys,X,config,(pop,selected,splits))
            assert seconds>=0 and bundle['model'].class_weight=='balanced_subsample'
            assert bundle['model'].random_state==42
            path=tmp_path/f'{target}.joblib';joblib.dump(bundle,path);bundle=joblib.load(path)
            result=pilot.evaluate(bundle,X['val'],(splits['val']%2).astype(np.int8),['0','1'],'label')
            assert sum(map(sum,result['confusion_matrix']))==len(splits['val'])
            assert result['prediction_total_seconds']>=result['predict_seconds']>=0
        np.testing.assert_array_equal(seen[0],seen[1])
        np.testing.assert_array_equal(seen[0],selected.indices)
        for s in X:np.testing.assert_array_equal(X[s],original[s])


def test_rf_rejects_incorrect_matrix_size_before_fit(pilot,tmp_path):
    frame=pd.DataFrame(dict(ts=np.arange(50)+1700000000,label=np.arange(50)%2,type='a'))
    config=PipelineConfig(split_strategy='temporal',timestamp_col='ts',timestamp_unit='s',selection_budget=10)
    splits=create_splits(frame,config);save_splits(splits,tmp_path,frame,config)
    with FrozenTrainingPopulation.load(tmp_path) as pop:
        selection=Selection.accept(pop,10,42,pop.train_indices[:10])
        with pytest.raises(ValueError,match='exactly B'):
            pilot.fit_timed(np.ones((9,2)),np.zeros(9),{},config,(pop,selection,splits))
