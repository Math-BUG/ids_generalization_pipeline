"""Integration-only real TON_IoT smoke. Never generates a scientific split."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from ids_pipeline import data_loading
from ids_pipeline.backend import resolve_backend
from ids_pipeline.clustering import fit_predict_clustering
from ids_pipeline.config import PipelineConfig, config_to_dict, load_config
from ids_pipeline.data_quality_policy import DATA_QUALITY_POLICY_VERSION
from ids_pipeline.feature_policy import (
    BEHAVIORAL_STRICT_FEATURES, BEHAVIORAL_STRICT_VERSION, FeaturePolicy, apply_proxy_feature_policies,
)
from ids_pipeline.leakage_checks import write_forbidden_columns_check
from ids_pipeline.preprocessing import fit_transform_preprocessing
from ids_pipeline.profiling import Profiler
from ids_pipeline.schema import TON_IOT_SCHEMA_VERSION
from ids_pipeline.split_protocols import NAMES, array_hash, digest_json, load_frozen_splits, manifest
from ids_pipeline.supervised import train_random_forest
from ids_pipeline.utils import read_json, setup_logging, write_json

FLAGS = {'integration_only': True, 'scientific_result': False}
BANNER = 'INTEGRATION ONLY | integration_only=true | scientific_result=false\n'
EXPECTED_HASH = 'sha256:828c57acef5ca3a953762a2dcf836dafd6b8dd513dd476c88ba924841a5f0a2a'
QUANTITATIVE = [c for c in BEHAVIORAL_STRICT_FEATURES if c != 'conn_state']
REUSE_KEYS = {
    'split_strategy', 'random_state', 'test_size', 'val_size', 'group_cols', 'timestamp_col',
    'timestamp_unit', 'temporal_bucket_freq', 'label_col', 'type_col', 'group_split_tolerance',
    'group_split_candidates', 'split_min_class_support', 'split_small_support_threshold',
    'frozen_splits_dir', 'split_export_csv',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_config(config):
    e = config.extra
    require(e.get('integration_only') is True and e.get('scientific_result') is False, 'Integration markers required')
    require(config.compute_backend == 'gpu' and e.get('strict_gpu') is True, 'Strict GPU configuration required')
    require(os.environ.get('IDS_COMPUTE_BACKEND', 'gpu') == 'gpu', 'Backend environment override must be gpu')
    require(config.feature_policy == 'behavioral_strict' and e['expected_feature_policy_version'] == BEHAVIORAL_STRICT_VERSION,
            'Frozen behavioral policy required')
    require(config.split_strategy == 'group_stratified' and e['expected_split_protocol_version'] == 'group_stratified/2.0.0',
            'Only the frozen group protocol is permitted')
    require(e['expected_split_hash'] == EXPECTED_HASH, 'Approved parent split hash required')
    require(config.svd_components == 50 and config.selected_k == 30, 'Keep requested reduction=50 and k=30')
    require(config.random_forest_estimators == 20 and config.targets == ['label', 'type', 'cluster_id'], 'Smoke RF configuration changed')
    require(config.random_state == 42 and e['subset_seed'] == 42, 'Only seed 42 is authorized')
    require(e['smoke_rows'] == dict(train=60000, val=15000, test=25000), 'Smoke must contain 60k/15k/25k rows')
    require(not config.export_cluster_assignments and not e.get('split_export_csv', True), 'Large CSV exports must be disabled')
    require(not config.use_representatives_for_supervised and config.representative_strategy == 'full', 'No representative selection in this smoke')
    require(not {'selection_budget', 'selection_methods', 'sample_size'} & e.keys(), 'No budget selectors or pre-split sampling')


class IOTrace(logging.Handler):
    """Observe the existing loader's decisions; never change its fallback behavior."""
    def __init__(self):
        super().__init__(logging.INFO)
        self.attempted_files = 0
        self.fallback_reasons = []

    def emit(self, record):
        message = record.getMessage()
        if str(record.msg).startswith('Reading %d Parquet file(s) with cuDF'):
            self.attempted_files += int(record.args[0])
        if 'Falling back to pandas for IO' in message:
            self.fallback_reasons.append(message[:2000])

    def result(self):
        n = len(self.fallback_reasons)
        backend = 'unobserved' if not self.attempted_files else 'cudf'
        if n:
            backend = 'pandas' if n == self.attempted_files else 'mixed_cudf_pandas'
        return dict(io_backend=backend, io_gpu_fallback=bool(n), io_fallback_reason=self.fallback_reasons or None,
                    io_cudf_attempted_files=self.attempted_files, io_fallback_files=n)


def optional_command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=10, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def gpu_environment(config):
    require(resolve_backend(config.compute_backend) == 'gpu', 'Effective backend is not GPU')
    import cupy as cp
    import cudf
    import cuml
    require(cp.cuda.runtime.getDeviceCount() == 1, 'Exactly one CUDA device must be visible')
    device = cp.cuda.runtime.getDevice()
    props = cp.cuda.runtime.getDeviceProperties(device)
    name = props['name'].decode() if isinstance(props['name'], bytes) else str(props['name'])
    require('A100' in name and props['totalGlobalMem'] >= 75 * 2**30, 'An A100 with 80 GB is required')
    return dict(python_version=platform.python_version(), numpy_version=np.__version__, pandas_version=pd.__version__,
                cupy_version=cp.__version__, cudf_version=cudf.__version__, cuml_version=cuml.__version__,
                cuda_toolkit_version=optional_command(['nvcc', '--version']),
                cuda_runtime_version=cp.cuda.runtime.runtimeGetVersion(), cuda_driver_api_version=cp.cuda.runtime.driverGetVersion(),
                nvidia_driver_version=optional_command(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader']),
                gpu_model=name, gpu_memory_bytes=int(props['totalGlobalMem']), cuda_device=device,
                CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'), SLURM_JOB_ID=os.environ.get('SLURM_JOB_ID'))


def check_inputs(config):
    e = config.extra
    root = Path(e['frozen_splits_dir']).resolve(strict=True)
    reuse = yaml.safe_load(Path(e['reuse_split_config']).read_text(encoding='utf-8'))
    require(set(reuse) == REUSE_KEYS, 'Unexpected or missing fields in reuse_split.yaml')
    effective = {**config_to_dict(config), **e}
    for key, value in reuse.items():
        same = Path(value).resolve() == root if key == 'frozen_splits_dir' else effective.get(key) == value
        require(same, f'reuse_split.yaml mismatch: {key}')
    saved = read_json(root / 'split_manifest.json')
    require(saved['validation_ok'] and saved['split_hash'] == e['expected_split_hash'], 'Parent split is not the approved checkpoint')
    require(saved['protocol_version'] == e['expected_split_protocol_version'], 'Parent split protocol mismatch')
    require(saved['counts'] == e['expected_full_split_rows'], 'Parent split counts mismatch')
    reference = read_json(root / 'data_quality_population.json')
    require(reference['population_id'] == saved['population']['source_population_id'], 'Frozen population reference mismatch')
    data_root = Path(e['data_path']).resolve(strict=True)
    require(data_root.is_dir(), 'Use the complete dataset directory, not a file or glob')
    files = data_loading._resolve_files(str(data_root))
    ids = [p.resolve().relative_to(data_root).as_posix() for p in files]
    require(len(files) == 23 and all(p.suffix == '.parquet' for p in files), 'Expected exactly 23 Parquet partitions')
    require(ids == [f['source_id'] for f in reference['files']], 'Dataset source order/set differs from frozen reference')
    return root, saved


def derive_subset(frozen, sizes, seed):
    """Integration subset only; called after validation, without access to targets."""
    require(set(frozen) == set(sizes) == set(NAMES), 'Exactly train/val/test required')
    selected, local, offset = {}, {}, 0
    for ordinal, name in enumerate(NAMES):
        parent = frozen[name]
        n = sizes[name]
        require(isinstance(n, int) and not isinstance(n, bool) and 0 < n <= len(parent), f'Impossible smoke size: {name}')
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, ordinal])))
        selected[name] = np.sort(parent[rng.choice(len(parent), size=n, replace=False)])
        require(len(np.unique(selected[name])) == n and np.isin(selected[name], parent).all(), 'Subset membership/uniqueness failure')
        local[name] = np.arange(offset, offset + n, dtype=np.int64)
        offset += n
    positions = np.concatenate([selected[s] for s in NAMES])
    require(len(np.unique(positions)) == len(positions), 'Smoke subsets overlap')
    return selected, local, positions


def target_distributions(df, local, parent_report):
    distributions = {}
    for target in ('label', 'type'):
        classes = sorted(parent_report['distributions'][target]['splits']['train']['counts'])
        distributions[target] = {}
        for name in NAMES:
            observed = df[target].iloc[local[name]].astype(str).value_counts()
            counts = {c: int(observed.get(c, 0)) for c in classes}
            require(sum(counts.values()) == len(local[name]), 'Unexpected target values in smoke')
            distributions[target][name] = dict(counts=counts, absent_classes=[c for c in classes if not counts[c]])
    return distributions


def load_smoke(config, output, report, profiler):
    root, saved = check_inputs(config)
    e = config.extra
    trace = IOTrace()
    logger = data_loading.LOGGER
    old_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(trace)
    try:
        with profiler.track('loading'):
            df = data_loading.load_dataset(
                e['data_path'], compute_backend='gpu',
                schema_report_path=output / 'schema_report.json',
                data_quality_report_path=output / 'data_quality_report.json')
    finally:
        logger.removeHandler(trace)
        logger.setLevel(old_level)
        report.update(trace.result())
    require(trace.attempted_files == 23, 'Could not observe the expected loader I/O path')
    quality = df.attrs['data_quality_report']
    for name in ('original_rows', 'quarantined_rows', 'eligible_rows'):
        require(quality[name] == e['expected_' + name], f'Population count mismatch: {name}')
    require(len(df) == quality['eligible_rows'], 'Eligible frame size mismatch')
    require(quality['policy_version'] == DATA_QUALITY_POLICY_VERSION, 'Quality policy mismatch')
    require(quality['schema_version'] == TON_IOT_SCHEMA_VERSION and df.attrs['schema_version'] == TON_IOT_SCHEMA_VERSION, 'Schema version mismatch')
    require(df.attrs['data_quality_population_id'] == quality['population_id'] == saved['population']['source_population_id'], 'Population identity mismatch')
    schema = read_json(output / 'schema_report.json')
    require(schema['ok'] and schema['index_preserved'] and schema['rows_before'] == schema['rows_after'] == len(df), 'Schema validation failed')
    report.update(dataset_name=config.dataset_name, data_path=e['data_path'],
                  **{k: quality[k] for k in ('original_rows', 'quarantined_rows', 'eligible_rows')},
                  schema_version=TON_IOT_SCHEMA_VERSION, data_quality_policy_version=DATA_QUALITY_POLICY_VERSION,
                  data_quality_population_id=quality['population_id'])
    with profiler.track('frozen_split_validation'):
        frozen = load_frozen_splits(root, df, config)
        observed = manifest(df, frozen, config)
        require(observed['validation_ok'] and observed['split_hash'] == saved['split_hash'] == e['expected_split_hash'], 'Observed split hash mismatch')
        require(observed['counts'] == e['expected_full_split_rows'], 'Observed split counts mismatch')
    report.update(split_protocol_version=observed['protocol_version'], frozen_splits_dir=str(root),
                  expected_split_hash=e['expected_split_hash'], observed_split_hash=observed['split_hash'], split_hash_match=True,
                  split_parameters=observed['parameters'], **{f'full_{s}_rows': len(frozen[s]) for s in NAMES})
    selected, local, positions = derive_subset(frozen, e['smoke_rows'], e['subset_seed'])
    smoke = df.iloc[positions].copy()
    original_map = np.load(root / 'eligible_original_positions.npy', mmap_mode='r', allow_pickle=False)
    require(len(original_map) == len(df), 'Original-position map length mismatch')
    require(np.array_equal(original_map[positions], smoke.index.to_numpy()), 'Original smoke identities differ from frozen map')
    identities = {}
    subset_dir = output / 'smoke_indices'
    subset_dir.mkdir()
    for name in NAMES:
        original = smoke.index.to_numpy()[local[name]]
        np.savez_compressed(subset_dir / f'{name}.npz', eligible_position=selected[name],
                            original_global_position=original, smoke_position=local[name])
        identities[name] = dict(eligible_hash=array_hash(selected[name]), original_hash=array_hash(original),
                                local_hash=array_hash(local[name]), path=f'smoke_indices/{name}.npz')
    identity = dict(parent_split_hash=observed['split_hash'], subset_seed=e['subset_seed'],
                    generator='PCG64(SeedSequence([subset_seed, split_ordinal]))', split_order=list(NAMES), indices=identities)
    distributions = target_distributions(smoke, local, observed)
    report.update(subset_name='integration smoke subset', subset_seed=e['subset_seed'], subset_identity=identity,
                  subset_hash=digest_json(identity), smoke_total_rows=len(smoke),
                  **{f'smoke_{s}_rows': len(local[s]) for s in NAMES}, distributions=distributions)
    write_json(output / 'smoke_distributions.json', {**FLAGS, 'distributions': distributions})
    print(BANNER + json.dumps(distributions, indent=2), flush=True)
    # These are the last full-population references; only 100k rows leave this function.
    del df, frozen, original_map
    gc.collect()
    return smoke, local


def gpu_matrices(X, counts):
    import cupy as cp
    require(set(X) == set(NAMES), 'Missing transformed split')
    widths = set()
    for name in NAMES:
        x = X[name]
        require(isinstance(x, cp.ndarray), f'{name} matrix is not a real CuPy array')
        require(x.device.id == cp.cuda.runtime.getDevice(), 'Matrix is on another CUDA device')
        require(x.ndim == 2 and x.shape[0] == counts[name], f'Matrix row count mismatch: {name}')
        require(bool(cp.isfinite(x).all()), 'Nonfinite transformed features')
        widths.add(x.shape[1])
    require(len(widths) == 1, 'Feature width changed between train/val/test')
    return widths.pop()


def class_name(obj):
    return f'{type(obj).__module__}.{type(obj).__name__}'


def verify_preprocessing(X, bundle, metadata, config, counts):
    from cuml.decomposition import PCA, TruncatedSVD
    require(bundle.backend == metadata.get('backend') == 'gpu', 'Preprocessing did not use GPU path')
    require(bundle.feature_cols == list(BEHAVIORAL_STRICT_FEATURES), 'Feature order changed')
    require(bundle.numeric_cols == metadata['numeric_cols'] == QUANTITATIVE, 'Quantitative semantic branch changed')
    require(bundle.categorical_cols == metadata['categorical_cols'] == ['conn_state'], 'Categorical semantic branch changed')
    encoding = metadata['conn_state_encoding']
    require(encoding['encoding'] == 'one_hot' and encoding['fit_split'] == 'train' and encoding['unknown'] == 'all_zero', 'Nominal encoding contract changed')
    require(not metadata['legacy_ordinal_columns'], 'Unexpected ordinal coordinates')
    require(bundle.preprocessor['conn_state_encoder'].metadata() == encoding, 'Encoding metadata/bundle mismatch')
    before = len(QUANTITATIVE) + len(encoding['vocabulary'])
    requested = int(config.svd_components or 0)
    effective = min(requested, max(0, min(counts['train'] - 1, before - 1)))
    after = gpu_matrices(X, counts)
    if effective >= 1:
        require(isinstance(bundle.svd, (TruncatedSVD, PCA)), 'Configured reduction must use a real cuML reducer')
        require(int(bundle.svd.n_components) == effective == metadata['svd_components_effective'] == after, 'Effective reduction dimension mismatch')
        algorithm = 'TruncatedSVD' if isinstance(bundle.svd, TruncatedSVD) else 'PCA'
        require(metadata['reducer'] == algorithm, 'Reported reducer differs from actual object')
    else:
        require(bundle.svd is None and metadata['reducer'] is None and after == before, 'Unexpected reduction for effective=0')
        algorithm = None
    return dict(preprocessing_backend='gpu', preprocessing_execution='hybrid_pandas_cupy',
                reduction_backend='gpu' if effective >= 1 else 'not_applicable',
                reduction_algorithm=algorithm, reduction_implementation=class_name(bundle.svd) if bundle.svd is not None else None,
                encoded_dimension_before_reduction=before, encoded_dimension_after_reduction=after,
                svd_components_requested=requested, svd_components_effective=effective,
                conn_state_encoding=encoding, transformed_shapes={s: list(X[s].shape) for s in NAMES})


def verify_clustering(result, counts, config):
    from cuml.cluster import KMeans
    require(isinstance(result['clusterer'], KMeans), 'Clustering object is not cuML KMeans')
    require(int(result['clusterer'].n_clusters) == config.selected_k, 'KMeans k changed')
    require(result['metrics']['backend'] == 'gpu' and result['metrics']['algorithm'] == 'cuML KMeans', 'Clustering backend mismatch')
    for name in NAMES:
        labels, distances = np.asarray(result['labels'][name]), np.asarray(result['distances'][name])
        require(labels.shape == distances.shape == (counts[name],), 'Clustering output row count mismatch')
        require(labels.dtype.kind in 'iu' and ((labels >= 0) & (labels < config.selected_k)).all(), 'Invalid cluster labels')
        require(np.isfinite(distances).all(), 'Nonfinite clustering distances')
    return dict(algorithm='cuML KMeans', implementation=class_name(result['clusterer']), k=config.selected_k, random_state=config.random_state)


def verify_supervised(metrics, X, frame, local, config, output):
    from cuml.ensemble import RandomForestClassifier
    require(metrics['backend'] == 'gpu' and set(metrics['targets']) == set(config.targets), 'Supervised targets/backend mismatch')
    require(not metrics['use_representatives_for_supervised'], 'RF must receive the complete smoke train')
    evidence = {}
    for target in config.targets:
        entry = metrics['targets'][target]
        require(entry['train_n_samples_original'] == entry['train_n_samples_used'] == len(local['train']), 'RF training population changed')
        for name in ('val', 'test'):
            require(all(np.isfinite(entry[name][key]) for key in ('accuracy', 'macro_f1', 'weighted_f1')), 'Missing supervised metrics')
        bundle = joblib.load(output / 'artifacts' / f'random_forest_{target}.joblib')
        model = bundle['model']
        require(isinstance(model, RandomForestClassifier) and bundle['model_name'] == 'cuML RandomForestClassifier', 'RF artifact is not cuML')
        require(int(model.n_estimators) == config.random_forest_estimators, 'RF tree count mismatch')
        require(int(model.n_features_in_) == X['train'].shape[1], 'RF did not receive the reduced feature width')
        if target != 'cluster_id':
            require(set(bundle['label_encoder'].classes_) == set(frame[target].iloc[local['train']].astype(str)), 'RF target vocabulary differs from smoke train')
        for filename in (f'classification_report_{target}.txt', f'confusion_matrix_{target}.csv'):
            require((output / filename).is_file(), f'Missing supervised artifact: {filename}')
        evidence[target] = dict(implementation=class_name(model), number_of_estimators=int(model.n_estimators),
                                random_state=config.random_state, train_rows=len(local['train']),
                                input_dimension=int(model.n_features_in_))
    return evidence


def mark_artifacts(output):
    """Only files from this new run; do not read assignment CSVs, even on failure."""
    inventory = []
    for path in sorted(output.rglob('*')):
        if not path.is_file() or path.name.endswith('.metadata.json') or path.name == 'integration_manifest.json':
            continue
        require(path.name != 'cluster_assignments.csv', 'Forbidden assignment export was created')
        relative = path.relative_to(output).as_posix()
        if path.suffix == '.json':
            payload = read_json(path)
            write_json(path, {**payload, **FLAGS})
        else:
            if path.suffix == '.txt':
                content = path.read_text(encoding='utf-8')
                if not content.startswith(BANNER):
                    path.write_text(BANNER + content, encoding='utf-8')
            write_json(path.with_name(path.name + '.metadata.json'), {**FLAGS, 'artifact': relative, 'manifest': 'integration_manifest.json'})
        inventory.append(relative)
    return inventory


def save_manifest(output, report):
    temporary = output / '.integration_manifest.tmp'
    write_json(temporary, {**report, **FLAGS})
    temporary.replace(output / 'integration_manifest.json')


def code_hashes():
    paths = [Path(__file__)] + sorted((REPO / 'src/ids_pipeline').glob('*.py'))
    return {p.relative_to(REPO).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(config):
    validate_config(config)
    e = config.extra
    output = Path(e['output_dir']).resolve()
    for parent in (REPO, Path(e['data_path']).resolve(), Path(e['frozen_splits_dir']).resolve()):
        require(output != parent and parent not in output.parents, 'Output must be outside repository, dataset and frozen split')
    output.mkdir(parents=True, exist_ok=False)
    setup_logging(output, config.log_level)
    logging.getLogger(__name__).info(BANNER.strip())
    (output / 'INTEGRATION_ONLY.txt').write_text(BANNER + 'These artifacts are not scientific results.\n', encoding='utf-8')
    report = dict(phase='1E', status='RUNNING', backend_requested='gpu', **FLAGS)
    profiler = Profiler(metadata=FLAGS.copy())
    save_manifest(output, report)
    try:
        with profiler.track('total'):
            report['stage'] = 'environment'
            report['environment'] = gpu_environment(config)
            report['code_sha256'] = code_hashes()
            write_json(output / 'effective_config.json', {**config_to_dict(config), **FLAGS})
            report['stage'] = 'loading_and_frozen_subset'
            frame, local = load_smoke(config, output, report, profiler)
            mark_artifacts(output)
            save_manifest(output, report)
            counts = {s: len(local[s]) for s in NAMES}
            require(counts == e['smoke_rows'], 'Heavy processing must use only the configured smoke subset')
            policy = FeaturePolicy.from_name(config.feature_policy)
            features = policy.select_features(frame, label_col=config.label_col, type_col=config.type_col, extra_target_cols=['cluster_id'])
            frame, features, proxies = apply_proxy_feature_policies(frame, features, timestamp_col=config.timestamp_col,
                                        timestamp_policy=config.timestamp_policy, service_policy=config.service_policy)
            require(features == list(BEHAVIORAL_STRICT_FEATURES), 'Frozen feature policy changed')
            selected = {**FLAGS, 'feature_policy': policy.name, 'selected_features': features,
                        **policy.selection_metadata(features), **proxies}
            write_json(output / 'selected_features.json', selected)
            write_forbidden_columns_check(output / 'forbidden_columns_check.json', frame, features,
                feature_policy=config.feature_policy, label_col=config.label_col, type_col=config.type_col, context='integration smoke')
            report.update(feature_policy=policy.name, feature_policy_version=policy.version, selected_features=features)
            report['stage'] = 'preprocessing_and_reduction'
            with profiler.track('preprocessing_and_reduction'):
                X, bundle = fit_transform_preprocessing(frame, local, features, config, output)
            metadata = read_json(output / 'artifacts/preprocessing_metadata.json')
            representation = verify_preprocessing(X, bundle, metadata, config, counts)
            restored = joblib.load(output / 'artifacts/preprocessing_bundle.joblib')
            require(verify_preprocessing(X, restored, metadata, config, counts) == representation, 'Preprocessing artifact mismatch')
            report.update(representation, feature_transformations=metadata['feature_transformations'])
            selected.update(feature_transformations=metadata['feature_transformations'], **representation)
            write_json(output / 'selected_features.json', selected)
            mark_artifacts(output)
            save_manifest(output, report)
            report['stage'] = 'clustering'
            with profiler.track('clustering'):
                clustering = fit_predict_clustering(X, frame, local, features, config, output)
            report['clustering'] = verify_clustering(clustering, counts, config)
            restored_kmeans = joblib.load(output / 'artifacts/cuml_kmeans.joblib')
            verify_clustering({**clustering, 'clusterer': restored_kmeans}, counts, config)
            report['clustering_backend'] = 'gpu'
            require(not (output / 'cluster_assignments.csv').exists(), 'Assignment CSV must not be created')
            mark_artifacts(output)
            save_manifest(output, report)
            report['stage'] = 'supervised'
            gpu_matrices(X, counts)
            metrics = train_random_forest(X, frame, local, features, config, output,
                                          cluster_labels=clustering['labels'], representatives=None, profiler=profiler)
            report['random_forest'] = dict(targets=config.targets,
                per_target=verify_supervised(metrics, X, frame, local, config, output))
            report['supervised_backend'] = 'gpu'
            import cupy as cp
            cp.cuda.runtime.deviceSynchronize()
            report['artifacts'] = mark_artifacts(output)
        report.update(status='APPROVED', stage='complete')
    except Exception as exc:
        report.update(status='INVALID', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        timing = profiler.to_dict()
        write_json(output / 'profiling.json', {**timing, **FLAGS})
        report['timing'] = dict(stages=timing['stages'], preprocessing_seconds=None, reduction_seconds=None,
            separate_reduction_reason='not_measured_separately',
            interpretation='Stage wall times include metrics and I/O; not RF.fit-only or isolated CUDA kernel timings. RAM observed at stage boundaries.')
        try:
            report['artifacts'] = mark_artifacts(output)
        except Exception as exc:
            report.update(status='INVALID', artifact_error=str(exc))
            save_manifest(output, report)
            raise
        save_manifest(output, report)
    print(BANNER + f'APPROVED: {output}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(REPO / 'configs/phase_1e_real_gpu_smoke.yaml'))
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == '__main__':
    main()
