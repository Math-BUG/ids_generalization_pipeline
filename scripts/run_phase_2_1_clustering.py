"""Stage A/B KMeans only. No labels, holdout features, external metrics or RF."""
from pathlib import Path
import shutil
import time

import joblib
import numpy as np

import phase_2_1_common as C


def stage_matrix(c,prepared,stage,representation):
    root=Path(c['output_dir'])
    import cupy as cp
    if stage=='A':
        record=prepared['matrices'][representation]['array']
        host=C.load_array(root,record)
        return cp.asarray(host,dtype=cp.float32),host,record['values_hash']
    raw=C.load_array(root,prepared['r0_train'])
    if representation=='R0':
        return cp.asarray(raw,dtype=cp.float32),raw,prepared['r0_train']['values_hash']
    reducer=joblib.load(C.checked_artifact(root,prepared['reducers'][representation]['artifact']))
    C.gpu_class(reducer,'TruncatedSVD')
    X=cp.asarray(reducer.transform(cp.asarray(raw,dtype=cp.float32)),dtype=cp.float32)
    host=C.to_numpy_array(X)
    return X,host,C.values_hash(host)


def execute_run(c,prepared,stage,r,k,seed,X,host,input_hash,metric_offsets,stability_offsets,environment):
    import cupy as cp
    from cuml.cluster import KMeans
    root=Path(c['output_dir'])
    name=C.run_id(r,k,seed)
    run_dir=root/f'stage_{stage.lower()}/attempts/{name}/{time.time_ns()}'
    run_dir.mkdir(parents=True,exist_ok=False)
    destination=root/f'stage_{stage.lower()}/runs/{name}.json'
    if destination.exists():
        shutil.copy2(destination,run_dir/'previous_attempt_record.json')
    profiler=C.Profiler()
    start=time.perf_counter()
    rec=dict(run_id=name,stage=stage,representation=r,k=k,seed=seed,
        preparation_digest=prepared['digest'],source_snapshot_hash=prepared['source_snapshot_hash'],
        sample_hashes={key:prepared['samples'][key] for key in ('development_indices_hash','metric_sample_indices_hash','full_metric_sample_indices_hash')},
        input_hash=input_hash,fit_n=len(X),components=c['representations'][r],environment=environment,
        validity=dict(status='RUNNING',reasons=[]),artifacts={},started_at=C.utcnow(),degeneracy_flags={})
    C.atomic_json(destination,C.sealed(rec))
    try:
        cols=c['representations'][r] or 21
        C.gpu_matrix(X,len(host),cols)
        expected=c['development_sample']['size'] if stage=='A' else c['expected_split_rows']['train']
        C.require(len(X)==expected,'Incorrect fit cardinality')
        model=KMeans(n_clusters=k,random_state=seed,**c['kmeans'])
        rec['implementation']=C.gpu_class(model,'KMeans')
        params={**c['kmeans'],'n_clusters':k,'random_state':seed}
        C.require(model.get_params(deep=False)==params,'KMeans effective parameters changed')
        with C.timed(profiler,'fit',gpu=True):
            model.fit(X)
        C.gpu_class(model,'KMeans')
        C.require(model.get_params(deep=False)==params,'KMeans parameters changed during fit')
        centers=cp.asarray(model.cluster_centers_)
        C.gpu_matrix(centers,k,cols)
        labels=np.asarray(C.to_numpy_array(model.labels_))
        counts=C.cluster_counts(labels,len(X),k,require_all=True)
        inertia=float(model.inertia_)
        C.require(np.isfinite(inertia) and inertia>=0,'NaN/Inf/negative inertia')
        with C.timed(profiler,'metrics'):
            metrics=C.internal_metrics(host,labels,metric_offsets,full_geometry=stage=='B',working_memory=c['metrics_working_memory_mib'])
            stats=C.concentration(counts)
            flags=C.degeneracy_flags(stats,c['degeneracy_flags'])
        with C.timed(profiler,'io'):
            path=run_dir/'kmeans.joblib'
            joblib.dump(model,path)
            restored=joblib.load(path)
            C.gpu_class(restored,'KMeans')
            C.require(restored.get_params(deep=False)==params,'Invalid saved model')
            rec['artifacts']['model']=C.artifact(root,path)
            rec['artifacts']['sample_assignments']=C.save_array(root,
                (run_dir/'sample_assignments.npy').relative_to(root),labels[stability_offsets].astype(np.uint8))
        rec.update(validity=dict(status='VALID',reasons=[]),metrics=metrics,concentration=stats,counts=counts.tolist(),
            degeneracy_flags=flags,inertia=inertia,inertia_per_row=inertia/len(X),parameters_effective=params,
            metrics_complete=not metrics['undefined_reasons'],n_iter=int(model.n_iter_) if hasattr(model,'n_iter_') else None,
            metric_scope=dict(silhouette='fixed_metric_sample',metric_sample_size=len(metric_offsets),geometry='full_train' if stage=='B' else 'fixed_development_metric_sample',
                concentration='full_fit_population',stability='fixed_development_sample'),
            warnings=['max_iter_reached'] if hasattr(model,'n_iter_') and int(model.n_iter_)>=c['kmeans']['max_iter'] else [])
        C.require(C.values_hash(X)==input_hash and C.values_hash(host)==input_hash,'KMeans/input storage modified frozen matrix')
    except BaseException as exc:
        rec['validity']=dict(status='INVALID',reasons=[f'{type(exc).__name__}: {exc}'])
        raise
    finally:
        durations=profiler.to_dict()
        rec.update(completed_at=C.utcnow(),profiling=durations,fit_seconds=durations['stages'].get('fit',{}).get('seconds',0.0),
            predict_seconds=0.0,metrics_seconds=durations['stages'].get('metrics',{}).get('seconds',0.0),
            io_seconds=durations['stages'].get('io',{}).get('seconds',0.0),total_seconds=time.perf_counter()-start,
            prediction_note='Fit labels_ indexed for training/stability; no separate predict call')
        final=C.sealed(rec)
        C.atomic_json(run_dir/'run.json',final)
        C.atomic_json(destination,final)
    return final


def run(c,stage,representation=None,finalist_index=None,retry_invalid=False):
    root=Path(c['output_dir'])
    with C.lock(root):
        prepared=C.load_prepared(c)
        decision=C.verify_decision(c,'A',prepared) if stage=='B' else None
        C.require(not C.decision_path(root,stage).exists(),'Stage is frozen; cannot run or retry after selection')
        if stage=='A':
            C.require(representation in c['representations'],'Choose an approved representation')
            pairs=[(representation,k) for k in c['k_values']]
        else:
            C.require(finalist_index is not None and 0<=finalist_index<len(decision['configurations']),'Invalid finalist index')
            pick=decision['configurations'][finalist_index]
            pairs=[(pick['representation'],pick['k'])]
            representation=pick['representation']
        env=C.gpu_environment(c)
        C.require(env['cuml_version']==prepared['environment']['cuml_version'] and env['software']==prepared['environment']['software'], 'Prepared environment changed')
        meta=C.read_json(root/C.MANIFEST)
        meta.update(status=f'RUNNING_{stage}',expected_runs_B=len(decision['configurations'])*3 if decision else 0)
        C.atomic_json(root/C.MANIFEST,meta)
        setup=C.Profiler()
        try:
            with C.timed(setup,'load_representation',gpu=True):
                X,host,h=stage_matrix(c,prepared,stage,representation)
            samples=prepared['samples']['arrays']
            metric_key='metric_offsets' if stage=='A' else 'full_metric_train_offsets'
            metric=C.load_array(root,samples[metric_key])
            stability=np.arange(len(X)) if stage=='A' else C.load_array(root,samples['development_train_offsets'])
            for r,k in pairs:
                for seed in c['clustering_seeds']:
                    path=root/f'stage_{stage.lower()}/runs/{C.run_id(r,k,seed)}.json'
                    if path.exists():
                        previous=C.read_json(path)
                        if previous['validity']['status']=='VALID':
                            C.validate_run(root,previous,prepared)
                            C.require(previous['input_hash']==h,'Resume matrix hash changed')
                            continue
                        C.require(retry_invalid,'Incomplete/invalid prior attempt; explicit --retry-invalid required')
                    execute_run(c,prepared,stage,r,k,seed,X,host,h,metric,stability,env)
            C.require(C.values_hash(X)==h,'Input changed after group')
            C.verify_provenance(c)
            meta['status']=f'{stage}_PARTIAL_OR_READY_FOR_REPORT'
        except BaseException as exc:
            meta.update(status='INCOMPLETE',error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            C.atomic_json(root/f'stage_{stage.lower()}/setup_{representation}.json',setup.to_dict())
            C.atomic_json(root/C.MANIFEST,meta)
            # Release mmap handles even when a failed run's traceback is retained.
            for name in ('host','metric','stability'):
                value=locals().get(name)
                if isinstance(value,np.memmap):
                    value._mmap.close()


def main():
    p=C.parser(__doc__)
    p.add_argument('--stage',choices=['A','B'],required=True)
    p.add_argument('--representation',choices=list(C.REPRESENTATIONS))
    p.add_argument('--finalist-index',type=int)
    p.add_argument('--retry-invalid',action='store_true')
    args=p.parse_args()
    run(C.load_config(args.config),args.stage,args.representation,args.finalist_index,args.retry_invalid)


if __name__=='__main__':
    main()
