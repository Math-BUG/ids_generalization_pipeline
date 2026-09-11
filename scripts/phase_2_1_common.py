"""Phase 2.1 contracts. Selection consumes only internal, label-free artifacts."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn import config_context
from sklearn.metrics import (adjusted_mutual_info_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from ids_pipeline.backend import to_numpy_array
from ids_pipeline.config import load_config as load_pipeline_config
from ids_pipeline.dataset_schema import normalize_dataset_schema
from ids_pipeline.feature_policy import BEHAVIORAL_STRICT_FEATURES, BEHAVIORAL_STRICT_VERSION
from ids_pipeline.preprocessing import apply_log1p
from ids_pipeline.profiling import Profiler
from ids_pipeline.split_protocols import array_hash, digest_json
from ids_pipeline.utils import to_jsonable

CONFIG = 'configs/phase_2_1_clustering_optimization.yaml'
PATH_KEYS = {'data_path', 'output_dir', 'frozen_splits_dir', 'parent_baseline_dir', 'source_provenance_path'}
SCIENCE_HASH = 'sha256:1acf4522208445c10e42a2edd4bc59cba1aa87899332613f661101bcc9a82643'
MANIFEST = 'phase_2_1_manifest.json'
SEEDS = [42, 123, 2026]
REPRESENTATIONS = {'R0': 0, 'R10': 10, 'R15': 15, 'R20': 20}
QUANTITATIVE = [c for c in BEHAVIORAL_STRICT_FEATURES if c != 'conn_state']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if exclusive:
        # Decisions are written under an exclusive phase lock; never overwrite.
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
    else:
        temporary = path.with_name(path.name + '.pending')
        temporary.write_text(text, encoding='utf-8', newline='\n')
        temporary.replace(path)


@contextmanager
def lock(root):
    """One mutating operation per experiment, including Slurm array workers."""
    path = Path(root) / '.phase21.lock'
    with path.open('x', encoding='utf-8') as stream:
        stream.write(f'{os.getpid()} {utcnow()}\n')
    try:
        yield
    finally:
        path.unlink()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return 'sha256:' + h.hexdigest()


def values_hash(values):
    h = hashlib.sha256()
    for start in range(0, len(values), 65536):
        block = np.asarray(to_numpy_array(values[start:start+65536]))
        h.update(np.ascontiguousarray(block).tobytes())
    return 'sha256:' + h.hexdigest()


def artifact(root, path):
    p = Path(path)
    return dict(path=p.relative_to(root).as_posix(), bytes=p.stat().st_size, sha256=file_hash(p))


def checked_artifact(root, record):
    root = Path(root).resolve()
    path = (root / record['path']).resolve()
    require(root in path.parents and path.name != 'cluster_assignments.csv', 'Invalid artifact path')
    require(path.is_file() and path.stat().st_size == record['bytes'], f'Missing/truncated artifact: {path.name}')
    require(file_hash(path) == record['sha256'], f'Corrupted artifact: {path.name}')
    return path


def sealed(value):
    require('digest' not in value, 'Already sealed')
    value=to_jsonable(value)
    return {**value, 'digest': digest_json(value)}


def check_seal(value):
    require(value.get('digest') == digest_json({k:v for k,v in value.items() if k != 'digest'}), 'Manifest/decision digest mismatch')
    return value


def validate_config(c):
    require(digest_json({k:v for k,v in c.items() if k not in PATH_KEYS}) == SCIENCE_HASH,
            'Frozen phase 2.1 scientific configuration/flags changed')
    require(all(isinstance(c.get(k), str) and c[k] for k in PATH_KEYS), 'Missing operational path')
    require(os.environ.get('IDS_COMPUTE_BACKEND', 'gpu') == 'gpu', 'GPU backend required')
    out = Path(c['output_dir']).resolve()
    for name in ('parent_baseline_dir', 'data_path', 'frozen_splits_dir'):
        other = Path(c[name]).resolve()
        require(out != other and other not in out.parents and out not in other.parents, 'Overlapping protected paths')
    require(out != REPO and REPO not in out.parents, 'Scientific output must be outside repository')


def load_config(path=REPO / CONFIG):
    c = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    validate_config(c)
    return c


def parent_config(c):
    base = load_pipeline_config(REPO / 'configs/phase_2_full_gpu_baseline.yaml')
    return base.with_updates(extra={**base.extra, 'data_path':c['data_path'], 'output_dir':c['output_dir'],
        'frozen_splits_dir':c['frozen_splits_dir'], 'reuse_split_config':str(Path(c['frozen_splits_dir'])/'reuse_split.yaml')})


def source_hashes():
    paths = list((REPO/'src/ids_pipeline').glob('*.py')) + list((REPO/'tests').glob('*.py'))
    for pattern in ('*phase_2_1*.py', '*phase_2_1*.slm'):
        paths.extend((REPO/'scripts').glob(pattern))
    paths += [REPO/p for p in (CONFIG, 'scripts/run_phase_2_full_gpu_baseline.py',
        'scripts/run_phase_1e_real_gpu_smoke.py', 'configs/phase_2_full_gpu_baseline.yaml',
        'docs/phase_2_1_clustering_optimization_proposal.md', 'docs/phase_2_1_implementation.md',
        'tests/phase_2_1_sample_hashes.json')]
    return {p.relative_to(REPO).as_posix():file_hash(p) for p in sorted(set(paths))}


def write_provenance(c):
    sha = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    require(len(sha)==40 and all(x in '0123456789abcdef' for x in sha), 'Source Git identity required')
    status = subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=REPO,text=True)
    pending = [x for x in status.splitlines() if x[3:] != c['source_provenance_path']]
    result = sealed(dict(format='phase_2_1_source/1', git_commit_sha=sha, git_pending_changes=pending,
        files=source_hashes(), config_hash=digest_json(c)))
    atomic_json(REPO/c['source_provenance_path'], result)
    return result


def verify_provenance(c):
    result = check_seal(read_json(REPO/c['source_provenance_path']))
    require(result['files'] == source_hashes() and result['config_hash'] == digest_json(c), 'Source/config differs from snapshot')
    return result


def gpu_environment(c):
    from run_phase_1e_real_gpu_smoke import gpu_environment as original
    info = original(parent_config(c))
    import psutil
    from importlib.metadata import version
    info.update(software={p:version(p) for p in ('numpy','pandas','scikit-learn','scipy','joblib','pyarrow','psutil')},
        cpus_allocated=os.environ.get('SLURM_CPUS_PER_TASK'), ram_allocated_mib=os.environ.get('SLURM_MEM_PER_NODE'),
        cpu_count=psutil.cpu_count(), ram_available_bytes=psutil.virtual_memory().available,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpu_uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip())
    return info


def gpu_matrix(X, rows=None, cols=None):
    import cupy as cp
    require(isinstance(X, cp.ndarray), 'Matrix backend is not CuPy GPU')
    require(X.ndim==2 and X.dtype == cp.float32, 'Invalid matrix dimensionality/dtype')
    if rows is not None:
        require(X.shape[0]==rows, 'Matrix cardinality mismatch')
    if cols is not None:
        require(X.shape[1]==cols, 'Representation dimension mismatch')
    for start in range(0, len(X), 65536):
        require(bool(cp.isfinite(X[start:start+65536]).all()), 'NaN/Inf in matrix')


def gpu_class(model, kind):
    name = type(model).__module__ + '.' + type(model).__name__
    prefix = 'cuml.cluster.' if kind=='KMeans' else 'cuml.decomposition.'
    require(name.startswith(prefix) and type(model).__name__ == kind, f'Wrong GPU implementation: {name}')
    return name


@contextmanager
def timed(profiler, name, *, gpu=False):
    if gpu:
        import cupy as cp
        cp.cuda.runtime.deviceSynchronize()
    with profiler.track(name):
        try:
            yield
        finally:
            if gpu:
                cp.cuda.runtime.deviceSynchronize()


def save_array(root, name, array):
    path = Path(root)/name
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), 'Refusing existing array')
    temp = path.with_name(path.name+'.pending')
    with temp.open('xb') as stream:
        np.save(stream, array, allow_pickle=False)
    temp.replace(path)
    return {**artifact(root,path), 'shape':list(array.shape), 'dtype':str(array.dtype), 'values_hash':values_hash(array)}


def load_array(root, record):
    path = checked_artifact(root,record)
    arr = np.load(path, mmap_mode='r', allow_pickle=False)
    require(list(arr.shape)==record['shape'] and str(arr.dtype)==record['dtype'], 'Array shape/dtype differs')
    return arr


def make_samples(train, original_ids, c):
    T = np.asarray(train)
    require(T.ndim==1 and T.dtype.kind in 'iu' and len(np.unique(T))==len(T), 'Invalid train identities')
    require(T.min()>=0 and T.max()<len(original_ids), 'Train indices out of population')
    n, m = c['development_sample']['size'], c['metric_sample_a']['size']
    b = c['metric_sample_b']['size']
    require(0<m<=n<=len(T) and 0<b<=len(T), 'Impossible sample size')
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(c['development_sample']['seed_sequence'])))
    offsets = np.sort(rng.choice(len(T),n,replace=False))
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(c['metric_sample_a']['seed_sequence'])))
    metric = np.sort(rng.choice(n,m,replace=False))
    rng = np.random.Generator(np.random.PCG64(c['metric_sample_b']['seed']))
    full = rng.choice(len(T),b,replace=False)
    dev = T[offsets]
    arrays = dict(development_indices=dev, development_train_offsets=offsets,
        development_original_ids=original_ids[dev], metric_sample_indices=dev[metric], metric_offsets=metric,
        metric_original_ids=original_ids[dev[metric]], full_metric_sample_indices=T[full],
        full_metric_train_offsets=full, full_metric_original_ids=original_ids[T[full]], train_indices=T)
    return {k:np.asarray(v,dtype='<i8') for k,v in arrays.items()}


def validate_bundle(bundle, expected_states=13):
    require(bundle.backend=='gpu', 'Parent preprocessing backend mismatch')
    require(bundle.feature_cols==list(BEHAVIORAL_STRICT_FEATURES), 'Feature policy/order mismatch')
    require(bundle.numeric_cols==QUANTITATIVE and bundle.categorical_cols==['conn_state'], 'Semantic branches changed')
    require(bundle.log_cols==QUANTITATIVE, 'Frozen log1p columns changed')
    stats = bundle.preprocessor['numeric_stats']
    for name in ('median','mean','std'):
        require(set(stats[name])==set(QUANTITATIVE), 'Incomplete numeric statistics')
        require(np.isfinite(list(stats[name].values())).all(), 'Invalid numeric statistics')
    require(all(x>0 for x in stats['std'].values()), 'Invalid scaling domain')
    encoder = bundle.preprocessor['conn_state_encoder']
    require(len(encoder.categories_)==expected_states, 'Parent vocabulary dimension changed')
    require(encoder.metadata()==bundle.preprocessor['conn_state_encoding'], 'Parent nominal metadata mismatch')
    gpu_class(bundle.svd,'TruncatedSVD')
    require(int(bundle.svd.n_components)==20, 'Parent R20 dimension mismatch')
    import cupy as cp
    components=cp.asarray(bundle.svd.components_)
    require(components.shape==(20,len(QUANTITATIVE)+expected_states) and bool(cp.isfinite(components).all()),
            'Parent R20 components invalid')
    require(bundle.svd.get_params(deep=False).get('random_state')==42,'Parent reducer seed changed')


def replay_preprocessing(frame, bundle):
    """Exactly the existing frozen pandas numeric recipe + GPU nominal scatter; no fit."""
    import cupy as cp
    require(list(frame.columns)==bundle.feature_cols, 'Replay accepts only ordered feature columns')
    clean, _ = normalize_dataset_schema(frame)
    clean = apply_log1p(clean,bundle.log_cols)
    stats = bundle.preprocessor['numeric_stats']
    num = clean[bundle.numeric_cols].astype('float64').fillna(pd.Series(stats['median']))
    num = ((num-pd.Series(stats['mean']))/pd.Series(stats['std'])).to_numpy(dtype=np.float32)
    X = cp.concatenate([cp.asarray(num,dtype=cp.float32),
        bundle.preprocessor['conn_state_encoder'].transform_gpu(clean[['conn_state']])],axis=1)
    gpu_matrix(X,len(frame))
    return X


def cluster_counts(labels, n, k, *, require_all=False):
    labels = np.asarray(labels)
    require(labels.ndim==1 and len(labels)==n, 'Cluster cardinality mismatch')
    require(labels.dtype.kind in 'iu', 'Cluster IDs must be integral')
    require(n>0 and labels.min()>=0 and labels.max()<k, 'Cluster IDs out of range')
    counts = np.bincount(labels.astype(np.int64), minlength=k)
    require(counts.sum()==n, 'Cluster counts mismatch')
    if require_all:
        require(np.count_nonzero(counts)==k, 'Observed fit clusters incompatible with requested k')
    return counts


def concentration(counts):
    counts=np.asarray(counts)
    require(counts.ndim==1 and len(counts)>1 and counts.dtype.kind in 'iu' and (counts>=0).all() and counts.sum()>0,
            'Invalid cluster count vector')
    n,k=int(counts.sum()),len(counts)
    p=counts/n
    entropy=float(-np.sum(p[p>0]*np.log(p[p>0])))
    result=dict(n=n,k=k,observed_clusters=int(np.count_nonzero(counts)),empty_clusters=int(np.sum(counts==0)),
        min_size=int(counts.min()),max_size=int(counts.max()),mean_size=float(counts.mean()),
        median_size=float(np.median(counts)),std_size=float(counts.std(ddof=0)),
        max_cluster_fraction=float(p.max()),top5_cluster_fraction=float(np.sort(p)[-min(5,k):].sum()),
        top10_cluster_fraction=float(np.sort(p)[-min(10,k):].sum()),entropy=entropy,
        normalized_entropy=float(entropy/np.log(k)),effective_number_of_clusters=float(np.exp(entropy)),
        effective_clusters_ratio=float(np.exp(entropy)/k))
    for t in (10,100,1000):
        result[f'clusters_lt{t}']=int(np.sum(counts<t))
        result[f'fraction_clusters_lt{t}']=float(np.mean(counts<t))
    return result


def degeneracy_flags(stats, policy):
    thresholds=policy['thresholds']
    checks=[('dominant_cluster','max_cluster_fraction',thresholds['max_cluster_fraction'],'ge',True),
        ('low_entropy','normalized_entropy',thresholds['min_normalized_entropy'],'lt',True),
        ('low_effective_ratio','effective_clusters_ratio',thresholds['min_effective_clusters_ratio'],'lt',True),
        ('top5_concentration','top5_cluster_fraction',thresholds['top5_cluster_fraction'],'ge',stats['k']>5),
        ('top10_concentration','top10_cluster_fraction',thresholds['top10_cluster_fraction'],'ge',stats['k']>10)]
    for t in (10,100,1000):
        checks.append((f'microclusters_lt{t}',f'fraction_clusters_lt{t}',thresholds['microcluster_fractions'][f'lt{t}'],'ge',True))
    result={}
    for name,key,limit,op,applicable in checks:
        value=stats[key]
        triggered=bool(value>=limit if op=='ge' else value<limit) if applicable else None
        result[name]=dict(value=value,threshold=limit,operator=op,triggered=triggered,
            state=('triggered' if triggered else 'clear') if applicable else 'not_applicable',
            reason=None if applicable else 'top-m includes all requested clusters')
    return result


def internal_metrics(X, labels, sample_offsets, *, full_geometry, working_memory=512):
    # Missing clusters in the fixed sample do not invalidate a valid fit.
    sample=np.asarray(X[sample_offsets]); sample_labels=np.asarray(labels)[sample_offsets]
    result=dict(sample_observed_clusters=int(len(np.unique(sample_labels))),undefined_reasons={})
    with config_context(working_memory=working_memory):
        if 1<len(np.unique(sample_labels))<len(sample_labels):
            result['silhouette']=float(silhouette_score(sample,sample_labels,metric='euclidean'))
        else:
            result['silhouette']=None
            result['undefined_reasons']['silhouette']='Fixed metric sample has insufficient distinct clusters'
        geom,y=(X,labels) if full_geometry else (sample,sample_labels)
        if 1<len(np.unique(y))<len(y):
            result['davies_bouldin']=float(davies_bouldin_score(geom,y))
            result['calinski_harabasz']=float(calinski_harabasz_score(geom,y))
        else:
            for name in ('davies_bouldin','calinski_harabasz'):
                result[name]=None
                result['undefined_reasons'][name]='Insufficient distinct clusters in metric scope'
    require(all(np.isfinite(v) for v in result.values() if isinstance(v,(int,float))), 'NaN/Inf in internal metrics')
    return result


def stability(assignments):
    require(sorted(assignments)==sorted(SEEDS), 'Three seed assignments required')
    shape=np.asarray(assignments[42]).shape
    require(len(shape)==1 and shape[0]>0 and all(np.asarray(x).shape==shape for x in assignments.values()), 'Stability row mismatch')
    pairs=[dict(seed_a=a,seed_b=b,ami=float(adjusted_mutual_info_score(assignments[a],assignments[b],average_method='arithmetic')))
           for a,b in itertools.combinations(SEEDS,2)]
    vals=[p['ami'] for p in pairs]
    require(np.isfinite(vals).all(), 'Nonfinite stability')
    return dict(pairs=pairs,mean_pairwise_stability_ami=float(np.mean(vals)),
        min_pairwise_stability_ami=float(np.min(vals)),max_pairwise_stability_ami=float(np.max(vals)),n_evaluated=shape[0])


PARETO_COLUMNS = ('mean_silhouette','mean_davies_bouldin','mean_calinski_harabasz',
                  'min_pairwise_stability_ami','worst_max_cluster_fraction')


def pareto_front(rows):
    """Closed internal-only interface; exact ties remain non-dominated."""
    allowed={'configuration',*PARETO_COLUMNS}
    for row in rows:
        require(set(row)==allowed, 'Pareto accepts only registered internal columns')
        require(all(np.isfinite(row[x]) for x in PARETO_COLUMNS), 'Nonfinite Pareto input')
    scores=np.asarray([[r[PARETO_COLUMNS[0]],-r[PARETO_COLUMNS[1]],r[PARETO_COLUMNS[2]],
                        r[PARETO_COLUMNS[3]],-r[PARETO_COLUMNS[4]]] for r in rows])
    return [r['configuration'] for i,r in enumerate(rows) if not any(
        np.all(scores[j]>=scores[i]) and np.any(scores[j]>scores[i]) for j in range(len(rows)) if i!=j)]


def parser(description):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--config',default=str(REPO/CONFIG))
    return p


def load_prepared(c):
    provenance=verify_provenance(c)
    root=Path(c['output_dir'])
    manifest=read_json(root/MANIFEST)
    prepared=check_seal(read_json(root/'representations/manifest.json'))
    require(prepared['status']=='READY' and manifest['preparation_digest']==prepared['digest'], 'Preparation incomplete/changed')
    require(prepared['config_hash']==digest_json(c) and prepared['source_snapshot_hash']==provenance['digest'], 'Prepared config/source differs')
    check_seal(prepared['samples'])
    require(read_json(root/'sample_manifest.json')==prepared['samples'], 'Sample manifest differs')
    for record in prepared['samples']['arrays'].values():
        checked_artifact(root,record)
    return prepared


def run_id(representation,k,seed):
    return f'{representation}_k{k}_s{seed}'


def expected_runs(c,stage,decision=None):
    pairs=[(r,k) for r in c['representations'] for k in c['k_values']] if stage=='A' else [
        (p['representation'],p['k']) for p in decision['configurations']]
    return [(r,k,s) for r,k in pairs for s in c['clustering_seeds']]


def validate_run(root,record,prepared):
    check_seal(record)
    require(record['preparation_digest']==prepared['digest'], 'Run preparation mismatch')
    require(record['validity']['status']=='VALID', 'Run is not valid')
    for item in record['artifacts'].values():
        checked_artifact(root,item)
    require(record['sample_hashes']=={k:prepared['samples'][k] for k in (
        'development_indices_hash','metric_sample_indices_hash','full_metric_sample_indices_hash')}, 'Run sample hashes differ')
    expected_sample=prepared['samples']['arrays']['development_indices']['shape'][0]
    require(record['artifacts']['sample_assignments']['shape']==[expected_sample], 'Stability sample cardinality differs')
    require(len(record['counts'])==record['k'] and sum(record['counts'])==record['fit_n'] and min(record['counts'])>0, 'Published fit cluster counts incompatible')
    require(record['implementation'].startswith('cuml.cluster.') and record['implementation'].endswith('.KMeans'), 'Published backend mismatch')
    return record


def decision_path(root,stage):
    return Path(root)/('stage_a/finalists.json' if stage=='A' else 'stage_b/final_selection.json')


def verify_decision(c,stage,prepared=None):
    root=Path(c['output_dir'])
    prepared=prepared or load_prepared(c)
    result=check_seal(read_json(decision_path(root,stage)))
    require(result['stage']==stage and result['preparation_digest']==prepared['digest'], 'Decision preparation/stage changed')
    require(result['external_metrics_used'] is False and result['approved_by'].strip() and result['rationale'].strip(), 'Decision needs explicit internal-only approval')
    configs=result['configurations']
    require(1<=len(configs)<=3 and len({(x['representation'],x['k']) for x in configs})==len(configs), 'Invalid finalist count')
    require(all(x['representation'] in c['representations'] and x['k'] in c['k_values'] for x in configs), 'Unknown finalist')
    require(result['seeds']==c['clustering_seeds'], 'Decision cannot choose a seed')
    if stage=='B':
        previous=verify_decision(c,'A',prepared)
        require(result['previous_decision_digest']==previous['digest'], 'Stage A decision changed')
        require(all(x in previous['configurations'] for x in configs), 'Stage B cannot add a candidate')
    else:
        previous=None
    evidence=check_seal(read_json(root/f'stage_{stage.lower()}/internal_evidence.json'))
    require(evidence['digest']==result['internal_evidence_digest'] and evidence['run_digests']==result['run_digests'], 'Internal evidence changed after freeze')
    require(all(f"{x['representation']}:k{x['k']}" in evidence['pareto'] for x in configs), 'Decision contains non-Pareto candidate')
    expected={run_id(r,k,s) for r,k,s in expected_runs(c,stage,previous)}
    require(set(result['run_digests'])==expected, 'Frozen decision lacks planned run evidence')
    for name,record_hash in result['run_digests'].items():
        rec=read_json(root/f'stage_{stage.lower()}/runs/{name}.json')
        check_seal(rec)
        if rec['validity']['status']=='VALID':
            validate_run(root,rec,prepared)
        require(rec['digest']==record_hash, 'Run changed after selection was frozen')
    return result
