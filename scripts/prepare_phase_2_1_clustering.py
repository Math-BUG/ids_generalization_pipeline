"""Prepare frozen samples/representations. Never fit KMeans or Random Forest."""
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import phase_2_1_common as C


def prepare(c):
    C.validate_config(c)
    provenance=C.verify_provenance(c)
    root=Path(c['output_dir'])
    root.mkdir(parents=True,exist_ok=False)
    report=dict(phase='2.1',scientific_result=True,experiment_type='clustering_optimization',
        parent_baseline='phase2',status='PREPARING',started_at=C.utcnow(),expected_runs_A=96,
        expected_runs_B=0,successful_runs_A=0,successful_runs_B=0,finalists=[],config=c,
        source_snapshot_hash=provenance['digest'],git_commit_sha=provenance['git_commit_sha'])
    C.atomic_json(root/C.MANIFEST,report)
    profiler=C.Profiler()
    try:
        with C.lock(root):
            env=C.gpu_environment(c)
            report['environment']=env
            parent_root=Path(c['parent_baseline_dir'])
            parent=C.read_json(parent_root/'experiment_manifest.json')
            C.require(parent['status']=='APPROVED' and parent['experiment_id']==c['parent_experiment_id'], 'Unapproved/wrong parent baseline')
            C.require(parent['data']['population_id']==c['expected_population_id'] and
                parent['split']['observed_hash']==c['expected_split_hash'], 'Parent population/split mismatch')
            C.require(parent['features']['selected_features']==list(C.BEHAVIORAL_STRICT_FEATURES), 'Parent feature policy mismatch')
            C.require(parent['features']['feature_policy_version']==C.BEHAVIORAL_STRICT_VERSION, 'Parent feature policy version mismatch')
            C.require(parent['clustering']['parameters_effective']=={**c['kmeans'],'n_clusters':30,'random_state':42}, 'Parent KMeans parameters differ')
            for name in ('python_version','cuml_version','cupy_version','cudf_version'):
                C.require(env[name]==parent['software'][name], f'Parent software mismatch: {name}')
            C.require(env['software']['numpy']==parent['software']['numpy_version'], 'NumPy version changed')
            C.require(env['software']['scikit-learn']==parent['software']['packages']['scikit-learn'], 'Metrics library changed')
            bundle_path=parent_root/'artifacts/preprocessing_bundle.joblib'
            bundle=joblib.load(bundle_path)
            C.validate_bundle(bundle)
            C.require(bundle.preprocessor['conn_state_encoding']==parent['preprocessing']['conn_state_encoding'], 'Parent encoder mismatch')
            C.require(parent['preprocessing']['encoded_dimension_before_reduction']==21 and
                parent['preprocessing']['svd_components_effective']==20, 'Parent representation changed')
            (root/'representations/reducers').mkdir(parents=True)
            copied=root/'representations/preprocessing_bundle.joblib'
            shutil.copy2(bundle_path,copied)
            parent_reference=dict(experiment_id=parent['experiment_id'],manifest_sha256=C.file_hash(parent_root/'experiment_manifest.json'),
                code_snapshot_hash=parent['code_snapshot_hash'],population_id=c['expected_population_id'],
                split_hash=c['expected_split_hash'],bundle_sha256=C.file_hash(bundle_path),
                kmeans_sha256=C.file_hash(parent_root/'artifacts/cuml_kmeans.joblib'))
            # Dataset audit may check labels for frozen coverage; no resulting labels
            # or external scores are included in prepared search inputs.
            from run_phase_2_full_gpu_baseline import load_population
            audit=root/'audit'
            audit.mkdir()
            audit_report={}
            df,splits=load_population(C.parent_config(c),audit,audit_report,profiler)
            C.require(len(df)==c['expected_eligible_rows'], 'Population cardinality mismatch')
            C.require(audit_report['data']['population_id']==c['expected_population_id'], 'Population identity mismatch')
            arrays=C.make_samples(splits['train'],df.index.to_numpy(),c)
            for other in ('val','test'):
                C.require(not np.intersect1d(arrays['development_indices'],splits[other]).size, 'Development overlaps holdout')
            samples={name:C.save_array(root,name+'.npy',arr) for name,arr in arrays.items()}
            samples_doc=C.sealed(dict(arrays=samples,development_seed=c['development_sample'],metric_seed=c['metric_sample_a'],
                full_metric_seed=c['metric_sample_b'],population_id=c['expected_population_id'],split_hash=c['expected_split_hash'],
                development_indices_hash=C.array_hash(arrays['development_indices']),
                metric_sample_indices_hash=C.array_hash(arrays['metric_sample_indices']),
                full_metric_sample_indices_hash=C.array_hash(arrays['full_metric_sample_indices'])))
            C.atomic_json(root/'sample_manifest.json',samples_doc)
            cache=root/'representations/cache'
            cache.mkdir()
            r0path=cache/'R0_train.npy'
            r0=np.lib.format.open_memmap(r0path,mode='w+',dtype='<f4',shape=(len(splits['train']),21))
            with C.timed(profiler,'replay_frozen_preprocessing',gpu=True):
                for start in range(0,len(r0),65536):
                    indices=splits['train'][start:start+65536]
                    block=C.replay_preprocessing(df.iloc[indices][bundle.feature_cols],bundle)
                    r0[start:start+len(indices)]=C.to_numpy_array(block)
            r0.flush()
            del block,df
            r0_record={**C.artifact(root,r0path),'shape':list(r0.shape),'dtype':str(r0.dtype),'values_hash':C.values_hash(r0)}
            import cupy as cp
            from cuml.decomposition import TruncatedSVD
            X=cp.asarray(r0,dtype=cp.float32)
            C.gpu_matrix(X,len(splits['train']),21)
            parent_params=bundle.svd.get_params(deep=False)
            C.require(parent_params.get('random_state')==c['reducer_seed'], 'Parent reducer seed mismatch')
            reducers={}
            matrices={}
            for representation,components in c['representations'].items():
                reducer=None
                if representation=='R20':
                    reducer=bundle.svd
                elif components:
                    params={**parent_params,'n_components':components,'random_state':c['reducer_seed']}
                    reducer=TruncatedSVD(**params)
                    C.gpu_class(reducer,'TruncatedSVD')
                    with C.timed(profiler,f'fit_{representation}_full_train',gpu=True):
                        reducer.fit(X)
                    C.require(reducer.get_params(deep=False)==params, 'Reducer parameters changed')
                if reducer is not None:
                    C.gpu_class(reducer,'TruncatedSVD')
                    C.require(reducer.n_components==components,'Reducer effective dimension mismatch')
                    C.require(bool(cp.isfinite(cp.asarray(reducer.components_)).all()), 'Nonfinite reducer')
                    path=root/f'representations/reducers/{representation}.joblib'
                    joblib.dump(reducer,path)
                    reducers[representation]=dict(artifact=C.artifact(root,path),parameters=reducer.get_params(deep=False),
                        fit_scope='full_train',fit_rows=len(X),reused_parent=representation=='R20')
                with C.timed(profiler,f'transform_{representation}_development',gpu=True):
                    D=X[cp.asarray(arrays['development_train_offsets'])]
                    if reducer is not None:
                        D=cp.asarray(reducer.transform(D),dtype=cp.float32)
                    C.gpu_matrix(D,len(arrays['development_indices']),components or 21)
                    record=C.save_array(root,f'representations/cache/{representation}_development.npy',C.to_numpy_array(D))
                    matrices[representation]=dict(array=record,row_indices_hash=samples_doc['development_indices_hash'],
                        feature_order=list(bundle.numeric_cols)+bundle.preprocessor['conn_state_encoder'].get_feature_names_out().tolist(),
                        components_requested=components,components_effective=components,input_columns=21)
                    del D
            C.require(C.values_hash(X)==r0_record['values_hash'], 'Reducer fit modified frozen R0 input')
            C.require(C.file_hash(bundle_path)==parent_reference['bundle_sha256'], 'Parent bundle changed during preparation')
            prepared=C.sealed(dict(status='READY',config_hash=C.digest_json(c),source_snapshot_hash=provenance['digest'],
                parent=parent_reference,samples=samples_doc,matrices=matrices,reducers=reducers,r0_train=r0_record,
                bundle=C.artifact(root,copied),environment=env,feature_policy_version=C.BEHAVIORAL_STRICT_VERSION,
                encoded_dimension=21,fit_scope='full_train',feature_order=list(bundle.feature_cols)))
            C.atomic_json(root/'representations/manifest.json',prepared)
            C.atomic_json(root/'source_provenance.json',provenance)
            C.atomic_json(root/'parent_reference.json',parent_reference)
            C.atomic_json(root/'configs/resolved.json',c)
            C.verify_provenance(c)
            report.update(status='READY',preparation_digest=prepared['digest'],parent_experiment_id=parent_reference['experiment_id'],
                dataset_population_id=c['expected_population_id'],frozen_split_hash=c['expected_split_hash'],
                development_indices_hash=samples_doc['development_indices_hash'],metric_sample_indices_hash=samples_doc['metric_sample_indices_hash'])
    except BaseException as exc:
        report.update(status='INVALID',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['preparation_profiling']=profiler.to_dict()
        C.atomic_json(root/C.MANIFEST,report)
    print('READY (no KMeans/RF fitted):',root)
    return prepared


def main():
    p=C.parser(__doc__)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--validate-config',action='store_true')
    mode.add_argument('--write-provenance',action='store_true')
    args=p.parse_args()
    c=C.load_config(args.config)
    if args.validate_config:
        print('Phase 2.1 configuration valid; no data/GPU execution')
    elif args.write_provenance:
        print(C.write_provenance(c)['digest'])
    else:
        prepare(c)


if __name__=='__main__':
    main()
