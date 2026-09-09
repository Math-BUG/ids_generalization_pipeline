"""Scientific GPU extension baseline. Requires the approved full frozen population.

--write-provenance only records local source identity; it never loads data/CUDA.
--validate-config only checks configuration; neither mode starts an experiment.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))

from ids_pipeline import data_loading, supervised
from ids_pipeline.config import config_to_dict, load_config
from ids_pipeline.data_quality_policy import DATA_QUALITY_POLICY_VERSION
from ids_pipeline.feature_policy import BEHAVIORAL_STRICT_FEATURES, BEHAVIORAL_STRICT_VERSION, FeaturePolicy, apply_proxy_feature_policies
from ids_pipeline.leakage_checks import write_forbidden_columns_check
from ids_pipeline.preprocessing import fit_transform_preprocessing
from ids_pipeline.clustering import fit_predict_clustering
from ids_pipeline.profiling import Profiler
from ids_pipeline.schema import TON_IOT_SCHEMA_VERSION
from ids_pipeline.split_protocols import NAMES, array_hash, digest_json, load_frozen_splits, manifest
from ids_pipeline.utils import read_json, write_json, setup_logging, to_jsonable
# Only stateless checks/observers are reused. Never call the integration executor,
# subset derivation, integration artifact marker, or integration config validator.
from run_phase_1e_real_gpu_smoke import (
    IOTrace, check_inputs, class_name, gpu_environment, gpu_matrices,
    optional_command, require, verify_clustering, verify_preprocessing,
)

FLAGS = dict(phase='2', scientific_result=True, integration_only=False,
             baseline_type='gpu_extension', exact_replication_of_original_article=False,
             phase_1g_omitted_by_project_decision=True)
EXPECTED_HASH = 'sha256:828c57acef5ca3a953762a2dcf836dafd6b8dd513dd476c88ba924841a5f0a2a'
COUNTS = dict(train=13575269, val=3244049, test=5518834)
TYPE_CLASSES = ['backdoor', 'ddos', 'dos', 'injection', 'mitm', 'normal', 'password', 'ransomware', 'scanning', 'xss']
CONFIG_PATH = 'configs/phase_2_full_gpu_baseline.yaml'
PROVENANCE_PATH = 'provenance/phase_2_source_provenance.json'
FROZEN_CONFIG = dict(
    dataset_name='ton_iot_network_full', compute_backend='gpu', feature_policy='behavioral_strict',
    label_col='label', type_col='type', split_strategy='group_stratified',
    group_cols=['src_ip', 'dst_ip', 'service', 'proto'], test_size=.25, val_size=.20,
    timestamp_col='ts', timestamp_unit='s', temporal_bucket_freq='1s',
    timestamp_policy='drop', service_policy='drop', svd_components=50,
    onehot_min_frequency=10, gpu_max_categories_per_col=64, log1p_numeric=True,
    log1p_columns=None, selected_k=30, cluster_batch_size=16384, cluster_n_init=5,
    cluster_max_iter=100, silhouette_sample_size=20000, export_cluster_assignments=False,
    representative_strategy='full', use_representatives_for_supervised=False,
    representatives_per_cluster=20, boundary_per_cluster=5,
    targets=['label', 'type', 'cluster_id'], random_forest_estimators=150, random_state=42, n_jobs=32,
)
FROZEN_EXTRA = dict(
    **FLAGS, strict_gpu=True, expected_feature_policy_version=BEHAVIORAL_STRICT_VERSION,
    expected_selected_features=list(BEHAVIORAL_STRICT_FEATURES),
    expected_split_protocol_version='group_stratified/2.0.0', expected_split_hash=EXPECTED_HASH,
    group_split_tolerance=.02, group_split_candidates=16, split_min_class_support=1,
    split_small_support_threshold=30, split_export_csv=False, expected_files=23,
    expected_original_rows=22339021, expected_quarantined_rows=869, expected_eligible_rows=22338152,
    expected_schema_version=TON_IOT_SCHEMA_VERSION,
    expected_data_quality_policy_version=DATA_QUALITY_POLICY_VERSION,
    expected_full_split_rows=COUNTS, expected_type_classes=TYPE_CLASSES,
    methodological_reference='docs/phase_1f_gpu_methodological_equivalence.md',
    source_provenance_path=PROVENANCE_PATH,
)
PATH_KEYS = {'data_path', 'output_dir', 'frozen_splits_dir', 'reuse_split_config'}


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def validate_config(config):
    for key, value in FROZEN_CONFIG.items():
        require(getattr(config, key) == value, f'Frozen configuration mismatch: {key}')
    for key, value in FROZEN_EXTRA.items():
        require(config.extra.get(key) == value, f'Frozen expectation mismatch: {key}')
    require(set(config.extra) == set(FROZEN_EXTRA) | PATH_KEYS, 'Unknown/missing operational options; no sampling or selectors allowed')
    for key in PATH_KEYS:
        require(isinstance(config.extra[key], str) and bool(config.extra[key]), f'Missing path: {key}')
    for key in ('strict_gpu', 'scientific_result', 'integration_only', 'split_export_csv'):
        require(type(config.extra[key]) is bool, f'Boolean required: {key}')
    require(config.export_cluster_assignments is False and config.use_representatives_for_supervised is False,
            'Export/representative flags must be false')
    require(os.environ.get('IDS_COMPUTE_BACKEND', 'gpu') == 'gpu', 'Backend environment override must be gpu')


def source_files():
    names = [CONFIG_PATH, 'configs/ton_iot_group_stratified_checkpoint.yaml',
             'configs/ton_iot_behavioral_strict_gpu.yaml', 'pyproject.toml',
             'scripts/run_phase_2_full_gpu_baseline.py', 'scripts/run_phase_2_full_gpu_baseline.slm',
             'scripts/run_phase_1e_real_gpu_smoke.py', 'docs/phase_1f_gpu_methodological_equivalence.md',
             'docs/phase_2_full_gpu_baseline_proposal.md']
    names += [p.relative_to(REPO).as_posix() for p in (REPO / 'src/ids_pipeline').glob('*.py')]
    names += [p.relative_to(REPO).as_posix() for p in (REPO / 'tests').glob('*.py')]
    return sorted(set(names))


def source_hashes():
    return {p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in source_files()}


def git_output(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True, encoding='utf-8').rstrip('\n')


def write_provenance():
    sha = git_output('rev-parse', 'HEAD')
    require(len(sha) == 40 and all(c in '0123456789abcdef' for c in sha), 'Valid Git commit required')
    # Keep Git's exact status records (including rename records); generated
    # provenance is excluded so recording it cannot create circular identity.
    pending = [s for s in git_output('status', '--porcelain=v1', '--untracked-files=all').splitlines()
               if s[3:] != PROVENANCE_PATH]
    identity = dict(git_commit_sha=sha, git_dirty=bool(pending), git_pending_changes=pending,
                    code_sha256=source_hashes())
    payload = dict(identity, code_snapshot_hash=digest_json(identity), generated_at_utc=utcnow(),
                   format='phase_2_source_provenance/1.0.0')
    write_json(REPO / PROVENANCE_PATH, payload)
    return payload


def verify_provenance(config):
    saved = read_json(REPO / config.extra['source_provenance_path'])
    require(saved['format'] == 'phase_2_source_provenance/1.0.0', 'Unsupported source provenance')
    identity = {k: saved[k] for k in ('git_commit_sha', 'git_dirty', 'git_pending_changes', 'code_sha256')}
    sha = identity['git_commit_sha']
    require(isinstance(sha, str) and len(sha) == 40 and all(c in '0123456789abcdef' for c in sha), 'Missing Git SHA')
    require(digest_json(identity) == saved['code_snapshot_hash'], 'Source provenance digest mismatch')
    require(source_hashes() == saved['code_sha256'], 'Source snapshot differs; regenerate provenance before transfer')
    return saved


def assert_new_output(config):
    path = Path(config.extra['output_dir'])
    if path.exists() or path.is_symlink():
        # Only file metadata, never read/delete an existing baseline.
        files = [p for p in path.rglob('*') if p.is_file()] if path.is_dir() and not path.is_symlink() else []
        total = sum(p.stat().st_size for p in files)
        names = [p.relative_to(path).as_posix() for p in files[:20]]
        raise ValueError(f'Refusing existing output {path}: files={len(files)}, bytes={total}, first_entries={names}')
    resolved = path.resolve()
    for parent in (REPO.resolve(), Path(config.extra['data_path']).resolve(), Path(config.extra['frozen_splits_dir']).resolve()):
        require(resolved != parent and parent not in resolved.parents, 'Output must be outside repository, data and frozen split')
    return resolved


def environment(config):
    info = gpu_environment(config)
    # Confirm estimator classes are importable before data loading.
    from cuml.cluster import KMeans
    from cuml.ensemble import RandomForestClassifier
    require(KMeans is not None and RandomForestClassifier is not None, 'cuML estimators unavailable')
    extras = {}
    for name in ('scipy', 'scikit-learn', 'joblib', 'pyarrow', 'psutil'):
        try:
            extras[name] = version(name)
        except PackageNotFoundError:
            extras[name] = None
    info.update(packages=extras, gcc=optional_command(['gcc', '--version']), hostname=socket.gethostname(),
                gpu_uuids=optional_command(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader']),
                cpus_allocated=os.environ.get('SLURM_CPUS_PER_TASK'),
                ram_allocated_mib=os.environ.get('SLURM_MEM_PER_NODE'),
                allocation_evidence_path=os.environ.get('IDS_PHASE2_ALLOCATION_FILE'))
    return info


@contextmanager
def stage(profiler, name, gpu=False):
    if gpu:
        import cupy as cp
        cp.cuda.runtime.deviceSynchronize()
    with profiler.track(name):
        try:
            yield
        finally:
            if gpu:
                cp.cuda.runtime.deviceSynchronize()


def load_population(config, output, report, profiler):
    root, saved = check_inputs(config)
    e = config.extra
    trace = IOTrace()
    logger = data_loading.LOGGER
    old_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(trace)
    try:
        with stage(profiler, 'loading'):
            df = data_loading.load_dataset(e['data_path'], compute_backend='gpu',
                schema_report_path=output / 'schema_report.json', data_quality_report_path=output / 'data_quality_report.json')
    finally:
        logger.removeHandler(trace)
        logger.setLevel(old_level)
        report.setdefault('data', {}).update(trace.result())
    require(trace.attempted_files == e['expected_files'], 'Missing official loader I/O evidence')
    q = df.attrs['data_quality_report']
    for key in ('original_rows', 'quarantined_rows', 'eligible_rows'):
        require(q[key] == e['expected_' + key], f'Population mismatch: {key}')
    require(len(df) == q['eligible_rows'], 'Eligible dataframe size mismatch')
    require(q['policy_version'] == DATA_QUALITY_POLICY_VERSION, 'Quality version mismatch')
    require(q['schema_version'] == df.attrs['schema_version'] == TON_IOT_SCHEMA_VERSION, 'Schema version mismatch')
    require(q['population_id'] == df.attrs['data_quality_population_id'] == saved['population']['source_population_id'], 'Population identity mismatch')
    schema = read_json(output / 'schema_report.json')
    require(schema['ok'] and schema['index_preserved'] and schema['rows_before'] == schema['rows_after'] == len(df), 'Schema validation failed')
    report['data'].update(dataset=config.dataset_name, data_path=e['data_path'], files=len(q['files']),
        ordered_source_ids=[f['source_id'] for f in q['files']],
        **{k: q[k] for k in ('original_rows', 'quarantined_rows', 'eligible_rows', 'population_id', 'schema_version')},
        data_quality_policy_version=q['policy_version'])
    write_json(output / 'dataset_info.json', report['data'])
    with stage(profiler, 'split_loading_validation'):
        splits = load_frozen_splits(root, df, config)
        observed = manifest(df, splits, config)
        validate_split(observed, config)
        require(observed['split_hash'] == saved['split_hash'], 'Frozen reference changed')
        original = np.load(root / 'eligible_original_positions.npy', mmap_mode='r', allow_pickle=False)
        require(original.shape == (len(df),) and original.dtype.kind in 'iu', 'Invalid original identity map')
        for start in range(0, len(df), 100000):
            require(np.array_equal(original[start:start+100000], df.index[start:start+100000].to_numpy()), 'Original row identities changed')
        del original
    report['split'] = dict(protocol=config.split_strategy, protocol_version=observed['protocol_version'],
        frozen_splits_dir=str(root), expected_hash=e['expected_split_hash'], observed_hash=observed['split_hash'],
        hash_match=True, **{s+'_rows': len(splits[s]) for s in NAMES},
        group_columns=config.group_cols, parameters=observed['parameters'], index_hashes=observed['index_hashes'],
        identity_hashes=observed['identity_hashes'], group_overlap=observed['group_overlap'],
        class_coverage=observed['distributions'], regenerated=False)
    write_json(output / 'split_manifest_observed.json', observed)
    write_json(output / 'split_reference.json', report['split'])
    write_json(output / 'target_distributions.json', observed['distributions'])
    return df, splits


def validate_split(observed, config):
    require(observed['validation_ok'], 'Split invariants failed')
    require(observed['protocol_version'] == 'group_stratified/2.0.0', 'Protocol changed')
    require(observed['split_hash'] == config.extra['expected_split_hash'], 'Approved split hash mismatch')
    require(observed['counts'] == config.extra['expected_full_split_rows'], 'Split counts mismatch')
    overlap = observed['group_overlap']
    require(overlap['ok'] and all(overlap[k] == 0 for k in ('train_val_overlap', 'train_test_overlap', 'val_test_overlap')), 'Group overlap detected')
    for target, expected in [('label', {'0', '1'}), ('type', set(config.extra['expected_type_classes']))]:
        for name in NAMES:
            counts = observed['distributions'][target]['splits'][name]['counts']
            require(set(counts) == expected and all(v > 0 for v in counts.values()), f'Class coverage failed: {target}/{name}')


def model_parameters(model):
    params = model.get_params(deep=False)
    # JSON-safe values only; never serialize estimator objects as opaque strings.
    return json.loads(json.dumps(to_jsonable(params), default=str))


def clustering_evidence(clustering, counts, config, output):
    evidence = verify_clustering(clustering, counts, config)
    model = clustering['clusterer']
    require(model.random_state == config.random_state and model.max_iter == config.cluster_max_iter, 'KMeans parameters changed')
    metrics = clustering['metrics']
    require(metrics['fit_split'] == 'train' and metrics['fit_n_samples'] == counts['train'], 'Clustering fit population mismatch')
    require(np.isfinite(metrics['inertia']) and metrics['inertia'] >= 0, 'Invalid clustering inertia')
    for key, value in metrics.items():
        if value is not None and isinstance(value, (int, float, np.number)):
            require(np.isfinite(value), f'Nonfinite clustering metric: {key}')
    evidence.update(backend='gpu', parameters_effective=model_parameters(model),
        configured_but_not_forwarded=dict(cluster_batch_size=config.cluster_batch_size, cluster_n_init=config.cluster_n_init),
        algorithmic_difference_from_original=True, original_algorithm='MiniBatchKMeans', current_algorithm='cuML KMeans',
        export_cluster_assignments=False,
        metric_scopes=dict(inertia='full_train', silhouette='train_sample', silhouette_sample_size=config.silhouette_sample_size,
                          davies_bouldin='full_train_cpu', calinski_harabasz='full_train_cpu', external='each_full_split_cpu'))
    distributions = {}
    for name in NAMES:
        sizes = np.bincount(clustering['labels'][name], minlength=config.selected_k)
        require(int(sizes.sum()) == counts[name], 'Cluster distribution row mismatch')
        distributions[name] = {str(i): dict(count=int(n), fraction=float(n/counts[name])) for i, n in enumerate(sizes)}
    write_json(output / 'cluster_distribution.json', distributions)
    write_json(output / 'clustering_metadata.json', evidence)
    return evidence


def metrics_from_confusion(cm):
    """Exact report from existing counts, with sklearn zero_division=0 semantics."""
    values = cm.to_numpy()
    require(values.ndim == 2 and values.shape[0] == values.shape[1] and np.isfinite(values).all(), 'Invalid confusion matrix')
    require((values >= 0).all() and (values == np.floor(values)).all(), 'Invalid confusion counts')
    true_names = [str(v).removeprefix('true_') for v in cm.index]
    pred_names = [str(v).removeprefix('pred_') for v in cm.columns]
    require(true_names == pred_names and len(set(true_names)) == len(true_names), 'Confusion class order mismatch')
    supports = values.sum(axis=1)
    predicted = values.sum(axis=0)
    tp = np.diag(values)
    precision = np.divide(tp, predicted, out=np.zeros(len(tp), dtype=float), where=predicted != 0)
    recall = np.divide(tp, supports, out=np.zeros(len(tp), dtype=float), where=supports != 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(len(tp), dtype=float), where=(precision+recall) != 0)
    n = int(supports.sum())
    require(n > 0, 'Empty confusion matrix')
    active = (supports + predicted) > 0  # same union as official y_true/y_pred metrics
    per_class = {c: dict(precision=float(precision[i]), recall=float(recall[i]), f1=float(f1[i]), support=int(supports[i])) for i, c in enumerate(true_names)}
    global_metrics = dict(accuracy=float(tp.sum()/n))
    for metric, arr in [('precision', precision), ('recall', recall), ('f1', f1)]:
        global_metrics['macro_' + metric] = float(arr[active].mean())
        global_metrics['weighted_' + metric] = float(np.dot(arr, supports)/n)
    return dict(per_class=per_class, global_metrics=global_metrics, rows=n)


def audit_supervised(metrics, X, df, splits, config, clustering, output):
    from cuml.ensemble import RandomForestClassifier
    require(metrics['backend'] == 'gpu' and set(metrics['targets']) == set(config.targets), 'RF backend/targets changed')
    require(metrics['use_representatives_for_supervised'] is False, 'Unexpected representative training')
    reports, parameters, implementations = {}, {}, {}
    for target in config.targets:
        entry = metrics['targets'][target]
        require(entry['train_n_samples_original'] == entry['train_n_samples_used'] == len(splits['train']), 'RF training population changed')
        bundle = joblib.load(output / 'artifacts' / f'random_forest_{target}.joblib')
        model = bundle['model']
        require(isinstance(model, RandomForestClassifier) and bundle['model_name'] == 'cuML RandomForestClassifier', 'RF artifact is not cuML/GPU')
        p = model_parameters(model)
        for key, expected in [('n_estimators', 150), ('random_state', 42), ('class_weight', None), ('bootstrap', True)]:
            require(key in p and p[key] == expected, f'RF parameter changed: {target}/{key}')
        require(model.n_features_in_ == X['train'].shape[1], 'RF input dimension mismatch')
        y = supervised._target_arrays(target, df, splits, config, clustering['labels'])
        require(set(bundle['label_encoder'].classes_) == set(y['train']), 'RF target vocabulary differs from full train')
        parameters[target] = p
        implementations[target] = class_name(model)
        reports[target] = {}
        for name in ('val', 'test'):
            if name == 'test':
                cm = pd.read_csv(output / f'confusion_matrix_{target}.csv', index_col=0)
            else:
                pred = supervised._predict_decoded(bundle, X[name], backend='gpu')
                cm = supervised._confusion_dataframe(y[name], pred)
                destination = output / 'validation_confusion_matrices'
                destination.mkdir(exist_ok=True)
                cm.to_csv(destination / f'{target}.csv')
                del pred
            result = metrics_from_confusion(cm)
            require(result['rows'] == len(splits[name]), f'Confusion support mismatch: {target}/{name}')
            observed_support = {str(k): int(v) for k, v in pd.Series(y[name]).value_counts().items()}
            require({c: v['support'] for c, v in result['per_class'].items() if v['support']} == observed_support, 'Per-class support differs from actual target')
            for key, value in result['global_metrics'].items():
                require(np.isfinite(entry[name][key]) and np.isclose(value, entry[name][key], rtol=1e-9, atol=1e-12), f'Metric mismatch: {target}/{name}/{key}')
            reference = [str(i) for i in range(config.selected_k)] if target == 'cluster_id' else (['0', '1'] if target == 'label' else config.extra['expected_type_classes'])
            result['reference_support'] = {c: observed_support.get(c, 0) for c in reference}
            result['absent_classes'] = [c for c in reference if observed_support.get(c, 0) == 0]
            result['small_support'] = {c: n for c, n in result['reference_support'].items() if 0 < n < 30}
            if target == 'label':
                for key in ('roc_auc', 'pr_auc', 'fpr_at_tpr_95', 'threshold_at_tpr_95'):
                    require(entry[name].get(key) is not None and np.isfinite(entry[name][key]), f'Missing binary metric: {name}/{key}')
                result['binary_metrics'] = {k: entry[name][k] for k in ('roc_auc', 'pr_auc', 'fpr_at_tpr_95', 'threshold_at_tpr_95', 'positive_label')}
                result['pr_auc_definition'] = 'average_precision_not_trapezoidal_area'
            reports[target][name] = result
        # Do not hold all three saved forests simultaneously during audit.
        del bundle, model, y
        gc.collect()
    write_json(output / 'metrics_by_class.json', reports)
    return dict(implementation='cuml.ensemble.RandomForestClassifier', backend='gpu', estimators=150,
        targets=config.targets, random_state=42, class_weight=None, sample_weight_supplied=False,
        difference_from_original_balanced_subsample=True, per_target_parameters_effective=parameters,
        per_target_implementations=implementations,
        per_target_train_rows={t: len(splits['train']) for t in config.targets}, training_index_hash=array_hash(splits['train']))


def artifact_inventory(output, complete=False):
    files = []
    for path in sorted(output.rglob('*')):
        require(path.name != 'cluster_assignments.csv', 'Forbidden cluster_assignments.csv created')
        if path.is_file() and path.name not in ('experiment_manifest.json', '.experiment_manifest.tmp'):
            files.append(dict(path=path.relative_to(output).as_posix(), bytes=path.stat().st_size))
    if complete:
        present = {f['path'] for f in files if f['bytes'] > 0}
        required = {'effective_config.json', 'source_provenance.json', 'dataset_info.json', 'schema_report.json',
            'data_quality_report.json', 'split_manifest_observed.json', 'split_reference.json', 'selected_features.json',
            'forbidden_columns_check.json', 'clustering_metadata.json', 'metrics_clustering.json', 'metrics_supervised.json',
            'cluster_distribution.json', 'target_distributions.json', 'metrics_by_class.json', 'profiling.json',
            'artifacts/preprocessing_metadata.json', 'artifacts/preprocessing_bundle.joblib', 'artifacts/cuml_kmeans.joblib'}
        for t in ('label', 'type', 'cluster_id'):
            required.update({f'artifacts/random_forest_{t}.joblib', f'classification_report_{t}.txt', f'confusion_matrix_{t}.csv', f'validation_confusion_matrices/{t}.csv'})
        require(required <= present, f'Missing/empty artifacts: {sorted(required-present)}')
    return files


def save_manifest(output, report):
    temporary = output / '.experiment_manifest.tmp'
    write_json(temporary, report)
    temporary.replace(output / 'experiment_manifest.json')


def run(config):
    # Never overwrite even an invalid prior experiment. Once our output exists,
    # every catchable failure, including invalid config, is marked INVALID.
    output = assert_new_output(config)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(FLAGS, status='RUNNING', started_at_utc=utcnow(), completed_at_utc=None,
        experiment_id='phase2_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '_' + os.environ.get('SLURM_JOB_ID', 'local'),
        validity=dict(checks={}, cluster_assignments_created=False, error=None))
    profiler = Profiler(metadata=FLAGS.copy())
    save_manifest(output, report)
    try:
        with stage(profiler, 'total'):
            validate_config(config)
            setup_logging(output, config.log_level)
            logging.getLogger(__name__).info('Scientific baseline of GPU extension; not exact article replication; phase 1G omitted')
            provenance = verify_provenance(config)
            for k in ('git_commit_sha', 'git_dirty', 'git_pending_changes', 'code_sha256', 'code_snapshot_hash'):
                report[k] = provenance[k]
            report['methodological_reference'] = config.extra['methodological_reference']
            write_json(output / 'source_provenance.json', provenance)
            env = environment(config)
            report['software'] = {k: v for k, v in env.items() if 'version' in k or k in ('packages', 'gcc')}
            report['hardware'] = {k: v for k, v in env.items() if k not in report['software']}
            report['validity']['checks'].update(configuration=True, source_provenance=True, cuda_environment=True)
            write_json(output / 'effective_config.json', {**config_to_dict(config), **config.extra})
            report['stage'] = 'loading_and_frozen_validation'
            save_manifest(output, report)
            df, splits = load_population(config, output, report, profiler)
            report['validity']['checks'].update(population=True, schema=True, quarantine=True, frozen_hash=True, split_invariants=True)
            train_hashes = {s: array_hash(splits[s]) for s in NAMES}
            policy = FeaturePolicy.from_name(config.feature_policy)
            features = policy.select_features(df, label_col=config.label_col, type_col=config.type_col, extra_target_cols=['cluster_id'])
            df, features, proxies = apply_proxy_feature_policies(df, features, timestamp_col=config.timestamp_col,
                timestamp_policy=config.timestamp_policy, service_policy=config.service_policy)
            require(features == list(BEHAVIORAL_STRICT_FEATURES) and policy.version == BEHAVIORAL_STRICT_VERSION, 'Feature policy/order changed')
            selected = dict(feature_policy=policy.name, selected_features=features, **policy.selection_metadata(features), **proxies)
            write_json(output / 'selected_features.json', selected)
            write_forbidden_columns_check(output / 'forbidden_columns_check.json', df, features,
                feature_policy=config.feature_policy, label_col=config.label_col, type_col=config.type_col, context='phase 2 scientific baseline')
            report['stage'] = 'preprocessing_and_reduction'
            save_manifest(output, report)
            with stage(profiler, 'preprocessing_and_reduction', gpu=True):
                X, bundle = fit_transform_preprocessing(df, splits, features, config, output)
            metadata = read_json(output / 'artifacts/preprocessing_metadata.json')
            counts = {s: len(splits[s]) for s in NAMES}
            representation = verify_preprocessing(X, bundle, metadata, config, counts)
            restored = joblib.load(output / 'artifacts/preprocessing_bundle.joblib')
            require(verify_preprocessing(X, restored, metadata, config, counts) == representation, 'Preprocessing artifact mismatch')
            del restored
            report['features'] = dict(selected, numeric_features=bundle.numeric_cols, categorical_features=bundle.categorical_cols,
                conn_state_encoding=metadata['conn_state_encoding'])
            report['preprocessing'] = dict(representation, log1p_columns=metadata['log_cols'],
                imputation='training_median_numeric_and_training_mode_conn_state', scaling='training_statistics_numeric_only',
                statistics_backend='pandas_cpu', numeric_preparation_backend='pandas_cpu', matrix_backend='cupy_gpu',
                end_to_end_gpu=False, feature_transformations=metadata['feature_transformations'])
            selected.update(feature_transformations=metadata['feature_transformations'], **representation)
            write_json(output / 'selected_features.json', selected)
            report['validity']['checks']['preprocessing'] = True
            report['stage'] = 'clustering'
            save_manifest(output, report)
            with stage(profiler, 'clustering', gpu=True):
                clustering = fit_predict_clustering(X, df, splits, features, config, output)
            report['clustering'] = clustering_evidence(clustering, counts, config, output)
            restored = joblib.load(output / 'artifacts/cuml_kmeans.joblib')
            verify_clustering({**clustering, 'clusterer': restored}, counts, config)
            del restored
            artifact_inventory(output)
            report['validity']['checks']['clustering_gpu'] = True
            report['stage'] = 'supervised'
            save_manifest(output, report)
            validate_config(config)
            gpu_matrices(X, counts)
            require(train_hashes == {s: array_hash(splits[s]) for s in NAMES}, 'Split indices changed before RF')
            with stage(profiler, 'supervised_total', gpu=True):
                metrics = supervised.train_random_forest(X, df, splits, features, config, output,
                    cluster_labels=clustering['labels'], representatives=None, profiler=profiler)
            with stage(profiler, 'supplemental_reporting', gpu=True):
                report['supervised'] = audit_supervised(metrics, X, df, splits, config, clustering, output)
            require(train_hashes == {s: array_hash(splits[s]) for s in NAMES}, 'Split indices changed during training')
            report['validity']['checks'].update(supervised_gpu=True, complete_training_rows=True, unchanged_indices=True, metrics=True)
            verify_provenance(config)
        report.update(status='APPROVED', stage='complete')
    except BaseException as exc:
        report.update(status='INVALID')
        report['validity']['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['completed_at_utc'] = utcnow()
        timing = profiler.to_dict()
        report['timing'] = dict(stages=timing['stages'], scope='stage_wall_time_including_metrics_and_io',
            preprocessing_seconds=None, reduction_seconds=None, rf_fit_only_measured=False,
            gnu_time_path=os.environ.get('IDS_PHASE2_TIME_FILE'), continuous_vram_peak_measured=False,
            note='Preprocessing and reduction measured together; per-target stage includes fit, prediction, metrics and I/O. VRAM/RAM sampled at boundaries.')
        write_json(output / 'profiling.json', dict(timing, **FLAGS))
        try:
            report['validity']['artifact_inventory'] = artifact_inventory(output, complete=report['status'] == 'APPROVED')
            report['validity']['cluster_assignments_created'] = False
        except Exception as exc:
            report['status'] = 'INVALID'
            report['validity']['artifact_error'] = str(exc)
            report['validity']['cluster_assignments_created'] = any(p.name == 'cluster_assignments.csv' for p in output.rglob('*'))
            save_manifest(output, report)
            raise
        save_manifest(output, report)
    print(f'APPROVED GPU EXTENSION BASELINE: {output}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(REPO / CONFIG_PATH))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--validate-config', action='store_true')
    modes.add_argument('--write-provenance', action='store_true')
    args = parser.parse_args()
    require(Path(args.config).resolve() == (REPO / CONFIG_PATH).resolve(), 'Use the provenance-bound official phase 2 config path')
    config = load_config(args.config)
    if args.write_provenance:
        validate_config(config)
        recorded = write_provenance()
        print(json.dumps({k: recorded[k] for k in ('git_commit_sha', 'git_dirty', 'code_snapshot_hash')}, indent=2))
        print(f'Provenance saved: {REPO / PROVENANCE_PATH}')
    elif args.validate_config:
        validate_config(config)
        print('Phase 2 config valid; no dataset/GPU/job execution')
    else:
        run(config)


if __name__ == '__main__':
    main()
