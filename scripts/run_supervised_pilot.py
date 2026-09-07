"""Seven frozen arms, label/type only; resumable pilot, no algorithm changes."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, roc_auc_score)
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ids_pipeline import supervised
from ids_pipeline.config import load_config,config_to_dict
from ids_pipeline.data_quality_policy import DataQualityPopulation,source_fingerprint,assert_source_unchanged
from ids_pipeline.dataset_schema import normalize_dataset_schema
from ids_pipeline.feature_policy import BEHAVIORAL_STRICT_FEATURES
from ids_pipeline.preprocessing import apply_log1p,_select_log_cols
from ids_pipeline.selection_budget import FrozenTrainingPopulation,Selection,hash_indices,prepare_fit_selection
from ids_pipeline.selection_methods import METHODS,TrainStrata,select_from_training_frame
from ids_pipeline.utils import write_json

SOURCE=Path('reports/increment_5/real')
SPLIT=Path('reports/increment_3/real/group_stratified')
ARMS=(*METHODS,'full')
TARGETS=('label','type')
BLOCK=100000


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def digest_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def frozen_inputs():
    verification=read_json('reports/increment_5/verification.json')
    before=read_json('reports/increment_5/frozen_before.json')
    for name,expected in before.items():
        if digest_file(Path('src/ids_pipeline')/(name+'.py'))!=expected:raise ValueError(f'Frozen source changed: {name}')
    for name,expected in verification['fitted_artifact_sha256'].items():
        if digest_file(SOURCE/name)!=expected:raise ValueError(f'Frozen fitted artifact changed: {name}')
    return {**before,'selection_methods':digest_file('src/ids_pipeline/selection_methods.py')}


def load_splits(pop):
    return {s:np.load(SPLIT/'splits'/f'{s}_indices.npy',mmap_mode='r') for s in ('train','val','test')}


def prepare(cache,output,pop):
    """Reapply frozen quality/schema; only transform with existing train-fitted objects."""
    processor=joblib.load(SOURCE/'preprocessor.joblib');reducer=joblib.load(SOURCE/'svd.joblib')
    reference=read_json('reports/ton_iot_quarantine_checkpoint/parquet/data_quality_report.json')
    splits=load_splits(pop)
    classes={t:sorted(pop.manifest['distributions'][t]['known_in_train']) for t in TARGETS}
    source_x=np.load(SOURCE/'transformed_train.npy',mmap_mode='r')
    arrays={};ys={};offsets={s:0 for s in splits}
    for s,ids in splits.items():
        arrays[s]=np.lib.format.open_memmap(cache/f'X_{s}.npy',mode='w+',dtype=np.float32,shape=(len(ids),source_x.shape[1]))
        for t in TARGETS:ys[s,t]=np.lib.format.open_memmap(cache/f'y_{t}_{s}.npy',mode='w+',dtype=np.int8,shape=(len(ids),))
    quality=DataQualityPopulation();eligible_offset=0;maximum_difference=0.0
    for item in reference['files']:
        path=Path(item['source_path']);fingerprint=source_fingerprint(path)
        if fingerprint['sha256']!=item['source_fingerprint']['sha256']:raise ValueError('Source file changed')
        quality.register_file(item['source_id'],fingerprint,source_path=str(path));original_offset=0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=BLOCK,columns=list(BEHAVIORAL_STRICT_FEATURES)+list(TARGETS)):
            raw=batch.to_pandas()
            eligible=quality.apply(raw,source_id=item['source_id'],row_offset=original_offset)
            normalized,_=normalize_dataset_schema(eligible)
            features=apply_log1p(normalized[list(BEHAVIORAL_STRICT_FEATURES)],_select_log_cols(list(BEHAVIORAL_STRICT_FEATURES)))
            encoded=processor.transform(features)
            transformed=reducer.transform(encoded)
            for s,ids in splits.items():
                a,b=np.searchsorted(ids,[eligible_offset,eligible_offset+len(eligible)])
                if a!=offsets[s]:raise ValueError('Noncontiguous frozen row order')
                local=ids[a:b]-eligible_offset
                current=transformed[local]
                if s=='train' and b>a:
                    # Verify the old training cache against the original rows, then retain its values.
                    old=source_x[a:b]
                    maximum_difference=max(maximum_difference,float(np.max(np.abs(current-old))))
                    if not np.allclose(current,old,rtol=1e-6,atol=1e-6):raise ValueError('Old training cache is not aligned to source rows')
                    current=old
                arrays[s][a:b]=current
                for t in TARGETS:
                    codes=pd.Categorical(normalized[t].iloc[local].astype(str),categories=classes[t]).codes
                    if np.any(codes<0):raise ValueError(f'Unexpected target outside frozen train vocabulary: {t}')
                    ys[s,t][a:b]=codes
                offsets[s]=b
            eligible_offset+=len(eligible);original_offset+=batch.num_rows
        assert_source_unchanged(path,fingerprint)
        print(f'Prepared {item["source_id"]}; eligible={eligible_offset:,}',flush=True)
    qr=quality.save(output/'data_quality_population.json')
    if qr['population_id']!=pop.manifest['population']['source_population_id']:raise ValueError('Eligible population changed')
    artifacts={}
    for s,ids in splits.items():
        assert offsets[s]==len(ids)
        arrays[s].flush();arrays[s]._mmap.close()
        for t in TARGETS:
            observed=np.bincount(ys[s,t],minlength=len(classes[t]))
            expected=pop.manifest['distributions'][t]['splits'][s]['counts']
            assert dict(zip(classes[t],map(int,observed)))==expected
            ys[s,t].flush();ys[s,t]._mmap.close()
    source_x._mmap.close()
    for path in cache.glob('*.npy'):artifacts[path.name]=digest_file(path)
    write_json(cache/'prepared.json',dict(status='complete',classes=classes,split_hash=pop.split_hash,
               train_population_hash=pop.train_population_hash,counts=offsets,artifacts=artifacts,
               maximum_train_transform_difference=maximum_difference,feature_policy='behavioral_strict/2.0.0',
               geometry='existing 21-column nominal preprocessing and fitted 20-component SVD',
               storage='float32, identical to the RF internal feature cast',holdouts_fit=False))


def validate_cache(cache,pop):
    metadata=read_json(cache/'prepared.json')
    assert metadata['status']=='complete' and metadata['split_hash']==pop.split_hash
    assert metadata['train_population_hash']==pop.train_population_hash
    for name,expected in metadata['artifacts'].items():
        if digest_file(cache/name)!=expected:raise ValueError(f'Pilot cache changed: {name}')
    return metadata


def selections(cache,output,pop,metadata):
    """Time reproducible selection calls; RF consumes the original validated ID files."""
    diagnostics={};result={}
    for method in ARMS:
        strata=None
        if method.startswith('stratified_'):
            t=method.removeprefix('stratified_')
            codes=np.load(cache/f'y_{t}_train.npy',mmap_mode='r')
            values=np.take(np.array(metadata['classes'][t],dtype=object),codes)
            strata=TrainStrata.from_values(pop,t,pop.train_indices,values)
            del values;codes._mmap.close()
        elif method.startswith('cluster_'):
            codes=np.load(SOURCE/'train_cluster_ids.npy',mmap_mode='r')
            assert hash_indices(codes)==read_json(SOURCE/'cluster_provenance.json')['cluster_ids_hash']
            strata=TrainStrata.from_values(pop,'cluster',pop.train_indices,codes);codes._mmap.close()
        start=time.perf_counter()
        if method=='full':generated=Selection.accept(pop,'full',42)
        elif method.startswith('stratified_'):
            from ids_pipeline.selection_methods import select_stratified_label,select_stratified_type
            function=select_stratified_label if method=='stratified_label' else select_stratified_type
            generated=function(pop,10000,42,strata).selection
        else:generated=select_from_training_frame(pop,10000,42,method,None,clusters=strata).selection
        seconds=time.perf_counter()-start
        if method!='full':
            path=SOURCE/'B_10000'/method
            original=Selection.accept(pop,10000,42,np.load(path/'selected_ids.npy'))
            assert original.selection_hash==read_json(path/'selection_summary.json')['selection_hash']==generated.selection_hash
            result[method]=original
        else:result[method]=generated
        result[method].save_summary(pop,output/method/'selection_summary.json')
        diagnostics[method]=dict(seconds=seconds,selection_hash=generated.selection_hash,
                                scope='selector call including budget validation/hashing; excludes loading/stratum construction',
                                reused_original_increment_5_ids=method!='full')
        del strata,generated;gc.collect()
    path=output/'selection_timing.json'
    if not path.exists():write_json(path,diagnostics)
    return result


def fit_timed(X_train,y_train,X,config,context):
    """Use the unchanged production RF factory, measuring only its actual fit call."""
    fit_seconds=[]
    class TimedForest(RandomForestClassifier):
        def fit(self,X,y,sample_weight=None):
            supervised._validate_fit_budget(X,y,config,context)
            start=time.perf_counter()
            result=super().fit(X,y,sample_weight=sample_weight)
            fit_seconds.append(time.perf_counter()-start)
            return result
    with patch.object(supervised,'RandomForestClassifier',TimedForest):
        bundle,_=supervised._fit_model(X_train,y_train,X,config,'cpu',selection_context=context)
    # Restore the ordinary class so artifacts contain no local instrumentation class.
    bundle['model'].__class__=RandomForestClassifier
    assert len(fit_seconds)==1
    return bundle,fit_seconds[0]


def metric_payload(y_true,y_pred,classes,scores=None):
    labels=np.arange(len(classes))
    report=classification_report(y_true,y_pred,labels=labels,target_names=classes,output_dict=True,zero_division=0)
    payload=dict(accuracy=float(accuracy_score(y_true,y_pred)),macro_f1=report['macro avg']['f1-score'],
                 weighted_f1=report['weighted avg']['f1-score'],classes_present=[classes[i] for i in np.unique(y_true)],
                 per_class={c:report[c] for c in classes},confusion_matrix=confusion_matrix(y_true,y_pred,labels=labels).tolist(),
                 confusion_class_order=classes)
    if scores is not None:
        payload.update(positive_label='1',roc_auc=float(roc_auc_score(y_true,scores)),
                       average_precision=float(average_precision_score(y_true,scores)),
                       average_precision_definition='sklearn average_precision_score; not trapezoidal PR-AUC')
        payload.update(supervised.compute_fpr_at_tpr(y_true,scores))
        payload['fpr_at_tpr_95_scope']='descriptive ROC operating point on this evaluation split; not a deployed threshold'
    return payload


def evaluate(bundle,X,y,classes,target):
    predicted=np.empty(len(y),dtype=np.int8)
    scores=np.empty(len(y),dtype=np.float64) if target=='label' else None
    predict_seconds=0.0;proba_seconds=0.0
    if scores is not None:
        original_classes=bundle['label_encoder'].inverse_transform(bundle['model'].classes_).astype(int)
        positive=np.flatnonzero(original_classes==1)
        if len(positive)!=1:raise ValueError('Binary RF has no positive class')
    for start in range(0,len(y),BLOCK):
        end=min(start+BLOCK,len(y));matrix=X[start:end]
        t=time.perf_counter();raw=bundle['model'].predict(matrix);predict_seconds+=time.perf_counter()-t
        predicted[start:end]=bundle['label_encoder'].inverse_transform(raw).astype(np.int8)
        if scores is not None:
            t=time.perf_counter();prob=bundle['model'].predict_proba(matrix);proba_seconds+=time.perf_counter()-t
            scores[start:end]=prob[:,positive[0]]
    payload=metric_payload(y,predicted,classes,scores)
    payload.update(predict_seconds=predict_seconds,predict_proba_seconds=proba_seconds,
                   prediction_total_seconds=predict_seconds+proba_seconds)
    return payload


def run_arm(method,pop,selection,cache,output,metadata,base_config):
    directory=output/method;directory.mkdir(exist_ok=True,parents=True)
    config=base_config.with_updates(selection_budget='full' if method=='full' else 10000,selection_method=None)
    splits=load_splits(pop)
    X={s:np.load(cache/f'X_{s}.npy',mmap_mode='r') for s in splits}
    common=dict(selection_hash=selection.selection_hash,selected_index_hash=hash_indices(selection.indices),
                frozen_split_hash=pop.split_hash,evaluation_index_hashes={s:hash_indices(splits[s]) for s in ('val','test')},
                cache_artifact_hashes=metadata['artifacts'],model_seed=42,B=len(selection.indices))
    for target in TARGETS:
        destination=directory/f'{target}.json'
        if destination.exists():
            previous=read_json(destination)
            if previous['identity']!=common:raise ValueError('Cannot resume a different arm population')
            print(f'Reusing complete {method}/{target}',flush=True);continue
        y={s:np.load(cache/f'y_{target}_{s}.npy',mmap_mode='r') for s in splits}
        context=(pop,selection,splits)
        if method=='full':
            # Full is identical ordered train: avoid a redundant multi-GB advanced-index copy.
            positions=selection.validate(pop,'full',42)
            assert np.array_equal(positions,np.arange(len(pop.train_indices)))
            X_train=X['train'];y_train=y['train'];del positions
        else:X_train,y_train=prepare_fit_selection(X,y['train'],splits,pop,selection,config)
        counts=np.bincount(y_train,minlength=len(metadata['classes'][target]))
        info=dict(identity=common,method=method,target=target,train_class_support=dict(zip(metadata['classes'][target],map(int,counts))),
                  absent_training_classes=[c for c,n in zip(metadata['classes'][target],counts) if not n])
        checkpoint=cache/f'rf_{method}_{target}.joblib'
        fit_record=directory/f'{target}_fit.json'
        if checkpoint.exists() and fit_record.exists():
            saved=read_json(fit_record)
            assert saved['identity']==common and saved['model_sha256']==digest_file(checkpoint)
            bundle=joblib.load(checkpoint);fit_seconds=saved['fit_seconds']
        else:
            print(f'RF.fit {method}/{target}: N={len(y_train):,}, trees={config.random_forest_estimators}',flush=True)
            bundle,fit_seconds=fit_timed(X_train,y_train,X,config,context)
            joblib.dump(bundle,checkpoint,compress=3)
            write_json(fit_record,dict(identity=common,fit_seconds=fit_seconds,model_sha256=digest_file(checkpoint),
                                      rf_parameters=bundle['model'].get_params()))
            print(f'RF.fit done {method}/{target}: {fit_seconds:.2f}s',flush=True)
        del X_train,y_train;gc.collect()
        info['fit_seconds']=fit_seconds
        for s in ('val','test'):
            print(f'Predicting {method}/{target}/{s}: N={len(y[s]):,}',flush=True)
            info[s]=evaluate(bundle,X[s],y[s],metadata['classes'][target],target)
            pd.DataFrame(info[s]['confusion_matrix'],index=metadata['classes'][target],columns=metadata['classes'][target]).to_csv(directory/f'confusion_{target}_{s}.csv')
            pd.DataFrame(info[s]['per_class']).T.to_csv(directory/f'classes_{target}_{s}.csv')
        write_json(destination,info)
        print(f'Completed {method}/{target}: test macro-F1={info["test"]["macro_f1"]:.6f}',flush=True)
        del bundle,y;gc.collect()
    for matrix in X.values():matrix._mmap.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir',required=True)
    parser.add_argument('--output',default='reports/supervised_pilot')
    parser.add_argument('--stage',choices=['prepare','run','all'],default='all')
    parser.add_argument('--methods',nargs='+',choices=ARMS,default=list(ARMS))
    args=parser.parse_args();cache=Path(args.cache_dir);output=Path(args.output)
    cache.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    frozen=frozen_inputs()
    config=load_config('configs/ton_iot_behavioral_strict_gpu.yaml').with_updates(
        compute_backend='cpu',selection_seed=42,model_seed=42,n_jobs=2,targets=list(TARGETS),
        extra={'frozen_splits_dir':str(SPLIT)})
    configuration=dict(config=config_to_dict(config),cache_dir=str(cache.resolve()),frozen_sources=frozen,
                       selected_arms=list(ARMS),compressed_budget=10000,full_budget=13575269,
                       note='Single-seed engineering pilot; no final scientific claim. n_jobs=2 for all arms.')
    path=output/'pilot_config.json'
    if path.exists() and read_json(path)!=configuration:raise ValueError('Pilot configuration changed; refusing overwrite')
    write_json(path,configuration)
    with FrozenTrainingPopulation.load(SPLIT) as pop,threadpool_limits(limits=2):
        if args.stage in ('prepare','all') and not (cache/'prepared.json').exists():prepare(cache,output,pop)
        metadata=validate_cache(cache,pop)
        write_json(output/'cache_reference.json',metadata)
        if args.stage=='prepare':return
        chosen=selections(cache,output,pop,metadata)
        for method in args.methods:run_arm(method,pop,chosen[method],cache,output,metadata,config)
    print('Requested pilot arms completed',flush=True)


if __name__=='__main__':main()
