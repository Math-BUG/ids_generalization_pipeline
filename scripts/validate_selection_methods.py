"""Increment 5: train-only current preprocessing/KMeans, twelve small selections, no RF."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import joblib
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ids_pipeline.config import load_config,config_to_dict
from ids_pipeline.data_quality_policy import DataQualityPopulation,source_fingerprint,assert_source_unchanged
from ids_pipeline.dataset_schema import normalize_dataset_schema,model_feature_types
from ids_pipeline.feature_policy import BEHAVIORAL_STRICT_FEATURES,BEHAVIORAL_STRICT_VERSION
from ids_pipeline.preprocessing import build_preprocessor,apply_log1p,_select_log_cols,TruncatedSVD
from ids_pipeline.clustering import MiniBatchKMeans
from ids_pipeline.selection_budget import FrozenTrainingPopulation,hash_indices
from ids_pipeline.selection_methods import (TrainStrata,select_random,select_stratified_label,select_stratified_type,
    select_cluster_uniform,select_cluster_proportional,select_cluster_sqrt)
from ids_pipeline.utils import write_json


def load_train(population,root):
    ref=json.loads(Path('reports/ton_iot_quarantine_checkpoint/parquet/data_quality_report.json').read_text(encoding='utf-8'))
    quality=DataQualityPopulation()
    columns=list(BEHAVIORAL_STRICT_FEATURES)+['label','type']
    membership=np.zeros(ref['eligible_rows'],dtype=bool);membership[population.train_indices]=True
    tables=[];eligible_offset=0;train_seen=0
    for item in ref['files']:
        path=Path(item['source_path']);fingerprint=source_fingerprint(path)
        if fingerprint['sha256']!=item['source_fingerprint']['sha256']:raise ValueError('Source changed since frozen checkpoint')
        quality.register_file(item['source_id'],fingerprint,source_path=str(path))
        original_offset=0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=100000,columns=columns):
            raw=batch.to_pandas();size=len(raw)
            eligible=quality.apply(raw,source_id=item['source_id'],row_offset=original_offset)
            n=len(eligible)
            ids=np.flatnonzero(membership[eligible_offset:eligible_offset+n])+eligible_offset
            if not np.array_equal(ids,population.train_indices[train_seen:train_seen+len(ids)]):raise ValueError('Training order changed')
            selected=eligible.iloc[ids-eligible_offset]
            # Stateless normalization and log transforms; no holdout statistics or vocabulary.
            normalized,_=normalize_dataset_schema(selected)
            normalized=apply_log1p(normalized,_select_log_cols(list(BEHAVIORAL_STRICT_FEATURES)))
            table=pa.Table.from_pandas(normalized,preserve_index=False)
            for col in ['conn_state','type']:
                table=table.set_column(table.schema.get_field_index(col),col,table[col].dictionary_encode())
            tables.append(table)
            train_seen+=len(ids);eligible_offset+=n;original_offset+=size
        assert_source_unchanged(path,fingerprint)
        print(f'Read {item["source_id"]}; cumulative train={train_seen:,}',flush=True)
    result=quality.save(root/'data_quality_population.json')
    if result['population_id']!=population.manifest['population']['source_population_id'] or train_seen!=len(population.train_indices):
        raise ValueError('Frozen eligible/train population mismatch')
    frame=pa.concat_tables(tables).to_pandas(categories=['conn_state','type'])
    return frame


def fit_current_preprocessor(train_frame,config,root):
    """Same frozen components, fitted only to train; disk backing changes no transforms."""
    feature_cols=list(BEHAVIORAL_STRICT_FEATURES)
    numeric,categorical=model_feature_types(train_frame,feature_cols,strict=True)
    processor=build_preprocessor(numeric,categorical,config.onehot_min_frequency)
    print('Fitting frozen median/scaler/nominal encoder on all training rows',flush=True)
    encoded=processor.fit_transform(train_frame)
    if hasattr(encoded,'toarray'):encoded=encoded.toarray()
    path=root/'encoded_train.npy'
    mapped=np.lib.format.open_memmap(path,mode='w+',dtype=encoded.dtype,shape=encoded.shape)
    for start in range(0,len(encoded),100000):mapped[start:start+100000]=encoded[start:start+100000]
    mapped.flush();del encoded;gc.collect()
    joblib.dump(processor,root/'preprocessor.joblib')
    return mapped,processor,numeric,categorical


def fit_current_reducer(encoded,config,root):
    n=min(int(config.svd_components or 0),encoded.shape[0]-1,encoded.shape[1]-1)
    if n<1:return encoded,None
    print(f'Fitting current TruncatedSVD: {encoded.shape} -> {n} components',flush=True)
    reducer=TruncatedSVD(n_components=n,random_state=config.random_state)
    transformed=reducer.fit_transform(encoded)
    mapped=np.lib.format.open_memmap(root/'transformed_train.npy',mode='w+',dtype=transformed.dtype,shape=transformed.shape)
    for start in range(0,len(transformed),100000):mapped[start:start+100000]=transformed[start:start+100000]
    mapped.flush();del transformed;gc.collect()
    joblib.dump(reducer,root/'svd.joblib')
    return mapped,reducer


def fit_current_clusterer(X,config):
    # Identical constructor and fit_predict to the existing CPU pipeline branch.
    # Predicting holdouts and computing distances afterward do not affect the fitted model.
    model=MiniBatchKMeans(n_clusters=config.selected_k,batch_size=config.cluster_batch_size,
                          random_state=config.seed_for('clustering'),n_init=config.cluster_n_init,
                          max_iter=config.cluster_max_iter)
    print(f'Fitting current CPU MiniBatchKMeans, k={config.selected_k}, N={len(X):,}',flush=True)
    labels=model.fit_predict(X)
    return model,labels


def save_result(result,population,root,B):
    directory=root/f'B_{B}'/result.diagnostics['method']
    result.save(population,directory)
    # Small selected-ID artifacts only (1k or 10k), no all-population CSV association.
    np.save(directory/'selected_ids.npy',result.selection.indices,allow_pickle=False)
    ids=np.load(directory/'selected_ids.npy',allow_pickle=False)
    assert len(ids)==B and len(np.unique(ids))==B and np.isin(ids,population.train_indices).all()
    return dict(method=result.diagnostics['method'],B=B,selection_hash=result.selection.selection_hash,
                quotas=result.diagnostics['quotas'],duplicates=0,outside_train=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='reports/increment_5/real')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--resume',action='store_true',help='Resume completed train encoding, without rereading source data')
    args=parser.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    config=load_config('configs/ton_iot_behavioral_strict_gpu.yaml').with_updates(compute_backend='cpu')
    assert config.selected_k==30 and config.feature_policy=='behavioral_strict'
    resolved=config_to_dict(config)
    if args.resume:
        saved=json.loads((root/'execution_config.json').read_text(encoding='utf-8'))
        if saved!=resolved:raise ValueError('Resume configuration differs from saved encoding configuration')
    else:
        write_json(root/'execution_config.json',resolved)
    with FrozenTrainingPopulation.load('reports/increment_3/real/group_stratified') as population,threadpool_limits(limits=args.threads):
        if args.resume:
            quality=json.loads((root/'data_quality_population.json').read_text(encoding='utf-8'))
            if quality['population_id']!=population.manifest['population']['source_population_id']:
                raise ValueError('Resume population mismatch')
            results=json.loads((root/'selections.json').read_text(encoding='utf-8'))['results']
            results=[r for r in results if not r['method'].startswith('cluster_')]
            if len(results)!=6:raise ValueError('Resume requires all six baseline selections')
            processor=joblib.load(root/'preprocessor.joblib')
            numeric=list(processor.transformers_[0][2]);categorical=['conn_state']
            encoded=np.load(root/'encoded_train.npy',mmap_mode='r')
            if encoded.shape!=(len(population.train_indices),len(processor.get_feature_names_out())):
                raise ValueError('Resume encoding shape mismatch')
            print(f'Resuming verified-size encoded training matrix {encoded.shape}',flush=True)
        else:
            train=load_train(population,root)
            labels=TrainStrata.from_values(population,'label',population.train_indices,train['label'].to_numpy())
            types=TrainStrata.from_values(population,'type',population.train_indices,train['type'].to_numpy())
            results=[]
            for B in [1000,10000]:
                for function,arguments in [(select_random,()),(select_stratified_label,(labels,)),(select_stratified_type,(types,))]:
                    result=function(population,B,config.seed_for('selection'),*arguments)
                    results.append(save_result(result,population,root,B))
            write_json(root/'selections.json',{'results':results,'status':'baselines_complete_clusters_pending'})
            del labels,types
            train.drop(columns=['label','type'],inplace=True);gc.collect()
            encoded,processor,numeric,categorical=fit_current_preprocessor(train,config,root)
            del train;gc.collect()
        if args.resume and (root/'svd.joblib').exists() and (root/'transformed_train.npy').exists():
            X=np.load(root/'transformed_train.npy',mmap_mode='r');reducer=joblib.load(root/'svd.joblib')
            expected=min(int(config.svd_components or 0),encoded.shape[0]-1,encoded.shape[1]-1)
            if X.shape!=(len(population.train_indices),expected) or reducer.n_components!=expected:
                raise ValueError('Resume reduced matrix shape mismatch')
        else:
            X,reducer=fit_current_reducer(encoded,config,root)
        if X is not encoded:encoded._mmap.close();del encoded;gc.collect()
        model,cluster_ids=fit_current_clusterer(X,config)
        joblib.dump(model,root/'kmeans.joblib')
        np.save(root/'train_cluster_ids.npy',cluster_ids,allow_pickle=False)
        provenance=dict(feature_policy=BEHAVIORAL_STRICT_VERSION,feature_cols=list(BEHAVIORAL_STRICT_FEATURES),
                        numeric_cols=numeric,categorical_cols=categorical,conn_state=processor.named_transformers_['conn_state'].metadata(),
                        backend='cpu',algorithm='existing MiniBatchKMeans',fit_split='train',fit_rows=len(population.train_indices),
                        frozen_split_hash=population.split_hash,train_population_hash=population.train_population_hash,
                        train_index_hash=hash_indices(population.train_indices),cluster_ids_hash=hash_indices(cluster_ids),
                        clustering_seed=config.seed_for('clustering'),k=config.selected_k,
                        svd_components=None if reducer is None else reducer.n_components,
                        holdouts_used_for_fit=False,labels_used_for_clustering=False)
        write_json(root/'cluster_provenance.json',provenance)
        strata=TrainStrata.from_values(population,'cluster',population.train_indices,cluster_ids)
        for B in [1000,10000]:
            for function in [select_cluster_uniform,select_cluster_proportional,select_cluster_sqrt]:
                result=function(population,B,config.seed_for('selection'),strata)
                results.append(save_result(result,population,root,B))
        write_json(root/'selections.json',{'results':results,'status':'complete','rf_fits':0})
        print('All twelve selections validated and saved; no RF trained',flush=True)


if __name__=='__main__':main()
