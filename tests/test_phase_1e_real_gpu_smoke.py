"""Local fixtures and explicit GPU doubles; these are NOT real CUDA tests."""
import copy
import importlib.util
import json
import logging
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.cluster import KMeans as CPUKMeans
from sklearn.ensemble import RandomForestClassifier as CPURandomForest

from ids_pipeline import data_loading, preprocessing, splitting, supervised
from ids_pipeline.config import load_config
from ids_pipeline.split_protocols import manifest

SPEC = importlib.util.spec_from_file_location('phase_1e', Path(__file__).resolve().parents[1] / 'scripts/run_phase_1e_real_gpu_smoke.py')
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class GPUArray(np.ndarray):
    @property
    def device(self):
        return SimpleNamespace(id=0)


def gpu_array(data, dtype=None):
    return np.asarray(data, dtype=dtype).view(GPUArray)


class BaseGPUReducer:
    def __init__(self, n_components, random_state=42):
        self.n_components = n_components
        self.random_state = random_state

    def fit_transform(self, X):
        self.n_features_in_ = X.shape[1]
        self.fit_rows_ = len(X)
        return self.transform(X)

    def transform(self, X):
        return gpu_array(X[:, :self.n_components])


class GPUReducer(BaseGPUReducer):
    pass


class GPUPCA(BaseGPUReducer):
    pass


class GPUKMeans(CPUKMeans):
    def __init__(self, n_clusters=30, max_iter=100, random_state=42):
        super().__init__(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state, n_init=1)


class GPURandomForest(CPURandomForest):
    def __init__(self, n_estimators=20, random_state=42, n_streams=1):
        self.n_streams = n_streams
        super().__init__(n_estimators=n_estimators, random_state=random_state, n_jobs=1)


@pytest.fixture
def gpu_doubles(monkeypatch):
    cp = ModuleType('cupy')
    cp.__version__ = 'test-double'
    cp.ndarray, cp.float32, cp.int32 = GPUArray, np.float32, np.int32
    cp.asarray, cp.asnumpy, cp.isfinite = gpu_array, np.asarray, np.isfinite
    cp.zeros = lambda shape, dtype: gpu_array(np.zeros(shape, dtype=dtype))
    cp.concatenate = lambda parts, axis: gpu_array(np.concatenate(parts, axis=axis))
    cp.linalg = np.linalg
    cp.cuda = SimpleNamespace(runtime=SimpleNamespace(
        getDeviceCount=lambda: 1, getDevice=lambda: 0, deviceSynchronize=lambda: None,
        getDeviceProperties=lambda _: dict(name=b'A100 test double', totalGlobalMem=80*2**30),
        runtimeGetVersion=lambda: 12090, driverGetVersion=lambda: 12090, memGetInfo=lambda: (70*2**30, 80*2**30)))
    cuml = ModuleType('cuml')
    cuml.__version__ = 'test-double'
    cudf = ModuleType('cudf')
    cudf.__version__ = 'test-double'
    for name, module in [('cupy', cp), ('cuml', cuml), ('cudf', cudf),
                         ('cuml.decomposition', SimpleNamespace(TruncatedSVD=GPUReducer, PCA=GPUPCA)),
                         ('cuml.cluster', SimpleNamespace(KMeans=GPUKMeans)),
                         ('cuml.ensemble', SimpleNamespace(RandomForestClassifier=GPURandomForest))]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv('IDS_COMPUTE_BACKEND', 'gpu')
    monkeypatch.setattr(smoke, 'optional_command', lambda _: 'test-double')
    return cp


@pytest.fixture
def real_small_sources(tmp_path, monkeypatch):
    """23 tiny real Parquet files exercise the official loader and frozen validation."""
    monkeypatch.delenv('IDS_COMPUTE_BACKEND', raising=False)
    source = tmp_path / 'data'
    source.mkdir()
    for partition in range(1, 24):
        n = np.arange(10) + (partition-1)*10
        frame = pd.DataFrame({c: (n+1).astype(float) for c in smoke.QUANTITATIVE})
        frame['src_bytes'] = (n+1).astype(str)
        if partition == 1:
            frame.loc[0, 'src_bytes'] = '0.0.0.0'
        frame['conn_state'] = np.where(n % 3 == 0, 'S0', 'SF')
        frame['src_ip'] = [f'10.0.0.{i//2}' for i in n]
        frame['dst_ip'], frame['service'], frame['proto'] = '10.1.0.1', 'http', 'tcp'
        frame['ts'], frame['label'], frame['type'], frame['dns_qtype'] = 1700000000+n, n % 2, np.where(n % 2, 'attack', 'normal'), 1
        frame.to_parquet(source / f'Network_dataset_{partition}.parquet', index=False)
    root = tmp_path / 'frozen'
    root.mkdir()
    base = load_config(smoke.REPO / 'configs/phase_1e_real_gpu_smoke.yaml')
    cfg = base.with_updates(extra={k: v for k, v in base.extra.items() if k != 'frozen_splits_dir'})
    df = data_loading.load_dataset(str(source), compute_backend='cpu',
        schema_report_path=root/'schema_reference.json', data_quality_report_path=root/'data_quality_population.json')
    frozen = splitting.create_splits(df, cfg)
    splitting.save_splits(frozen, root, df, cfg)
    np.save(root/'eligible_original_positions.npy', df.index.to_numpy(dtype='<i8'))
    parent = manifest(df, frozen, cfg)
    effective = {**cfg.__dict__, **cfg.extra, 'frozen_splits_dir': str(root)}
    (root/'reuse_split.yaml').write_text(yaml.safe_dump({k: effective[k] for k in smoke.REUSE_KEYS}), encoding='utf-8')
    cfg = cfg.with_updates(extra={**cfg.extra, 'data_path': str(source), 'output_dir': str(tmp_path/'smoke'),
        'frozen_splits_dir': str(root), 'reuse_split_config': str(root/'reuse_split.yaml'),
        'expected_split_hash': parent['split_hash'], 'expected_original_rows': 230, 'expected_quarantined_rows': 1,
        'expected_eligible_rows': 229, 'expected_full_split_rows': parent['counts'], 'smoke_rows': dict(train=60, val=15, test=25)})
    return cfg, df, frozen


def prepare_run(monkeypatch, cfg, fallback=False):
    # Only fixed production sizes/hash and the outside-repo location are relaxed for fixtures.
    monkeypatch.setattr(smoke, 'validate_config', lambda _: None)
    monkeypatch.setattr(smoke, 'REPO', Path(cfg.extra['output_dir']).parent / 'not_the_repository')
    monkeypatch.setattr(smoke, 'code_hashes', lambda: {'test_fixture': 'not-scientific'})
    monkeypatch.setattr(smoke, 'setup_logging', lambda *args: None)
    def read_gpu(files, *, require_cudf):
        assert require_cudf
        data_loading.LOGGER.info('Reading %d Parquet file(s) with cuDF', len(files))
        if fallback:
            data_loading.LOGGER.warning('cuDF could not read all Parquet files together (fixture). Falling back to pandas for IO; test double')
        return data_loading._read_parquet_with_pandas(files)
    monkeypatch.setattr(data_loading, '_read_parquet_with_cudf', read_gpu)


def test_approved_config_and_no_scientific_baseline_change(monkeypatch):
    monkeypatch.delenv('IDS_COMPUTE_BACKEND', raising=False)
    config = load_config(smoke.REPO/'configs/phase_1e_real_gpu_smoke.yaml')
    smoke.validate_config(config)
    baseline = load_config(smoke.REPO/'configs/ton_iot_behavioral_strict_gpu.yaml')
    assert baseline.random_forest_estimators == 150
    for field in ('svd_components', 'selected_k', 'cluster_max_iter', 'cluster_n_init', 'feature_policy', 'log1p_numeric'):
        assert getattr(config, field) == getattr(baseline, field)
    with pytest.raises(ValueError, match='seed'):
        smoke.validate_config(config.with_updates(random_state=43))
    monkeypatch.setenv('IDS_COMPUTE_BACKEND', 'cpu')
    with pytest.raises(ValueError, match='override'):
        smoke.validate_config(config)


def test_subset_reproducible_no_replacement_and_stable_identity(real_small_sources):
    config, df, frozen = real_small_sources
    selected, local, positions = smoke.derive_subset(frozen, config.extra['smoke_rows'], 42)
    changed = df.copy()
    changed['label'], changed['type'] = 0, 'arbitrary'
    again, _, second_positions = smoke.derive_subset(frozen, config.extra['smoke_rows'], 42)
    np.testing.assert_array_equal(positions, second_positions)
    assert len(np.unique(positions)) == 100
    for name in smoke.NAMES:
        np.testing.assert_array_equal(selected[name], again[name])
        np.testing.assert_array_equal(positions[local[name]], selected[name])
        assert np.isin(selected[name], frozen[name]).all()
    with pytest.raises(ValueError, match='Impossible'):
        smoke.derive_subset(frozen, dict(train=len(df)+1, val=15, test=25), 42)


@pytest.mark.parametrize('fallback', [False, True])
def test_complete_orchestration_with_gpu_doubles(real_small_sources, gpu_doubles, monkeypatch, fallback):
    config, original, frozen = real_small_sources
    prepare_run(monkeypatch, config, fallback)
    actual_load = data_loading.load_dataset
    events = []
    def loader(*args, **kwargs):
        assert args == (config.extra['data_path'],)
        assert 'sample_size' not in kwargs and kwargs['compute_backend'] == 'gpu'
        events.append('load')
        return actual_load(*args, **kwargs)
    monkeypatch.setattr(data_loading, 'load_dataset', loader)
    actual_frozen = smoke.load_frozen_splits
    def load_parent(*args):
        result = actual_frozen(*args)
        events.append('validated')
        return result
    monkeypatch.setattr(smoke, 'load_frozen_splits', load_parent)
    actual_subset = smoke.derive_subset
    def subset(*args):
        assert events == ['load', 'validated']
        return actual_subset(*args)
    monkeypatch.setattr(smoke, 'derive_subset', subset)
    def no_search(*args):
        raise AssertionError('No split regeneration is allowed')
    monkeypatch.setattr(splitting, 'group_split', no_search)
    matrices = []
    selected_parent, _, _ = actual_subset(frozen, config.extra['smoke_rows'], 42)
    expected_targets = [original.iloc[selected_parent['train']][target].astype(str).to_numpy() for target in ('label', 'type')]
    actual_clustering = smoke.fit_predict_clustering
    def cluster(*args):
        result = actual_clustering(*args)
        expected_targets.append(result['labels']['train'].astype(str))
        return result
    monkeypatch.setattr(smoke, 'fit_predict_clustering', cluster)
    actual_fit = supervised._fit_model
    def fit(X_train, y_train, X, cfg, backend):
        assert X_train is X['train'] and len(y_train) == 60 and backend == 'gpu'
        np.testing.assert_array_equal(y_train, expected_targets[len(matrices)])
        matrices.append(X_train)
        return actual_fit(X_train, y_train, X, cfg, backend)
    monkeypatch.setattr(supervised, '_fit_model', fit)
    result = smoke.run(config)
    assert result['status'] == 'APPROVED' and result['split_hash_match']
    assert result['io_gpu_fallback'] is fallback
    assert result['io_backend'] == ('pandas' if fallback else 'cudf')
    assert bool(result['io_fallback_reason']) is fallback
    assert len(matrices) == 3 and all(x is matrices[0] for x in matrices)
    assert result['svd_components_requested'] == 50
    assert result['svd_components_effective'] == result['encoded_dimension_after_reduction'] == 9
    assert result['preprocessing_backend'] == result['reduction_backend'] == result['supervised_backend'] == 'gpu'
    root = Path(config.extra['output_dir'])
    assert not (root/'cluster_assignments.csv').exists()
    for path in root.rglob('*.json'):
        payload = json.loads(path.read_text())
        assert payload['integration_only'] is True and payload['scientific_result'] is False
    selected, _, _ = actual_subset(frozen, config.extra['smoke_rows'], 42)
    for name in smoke.NAMES:
        with np.load(root/f'smoke_indices/{name}.npz') as ids:
            np.testing.assert_array_equal(ids['eligible_position'], selected[name])
            np.testing.assert_array_equal(ids['original_global_position'], original.index.to_numpy()[selected[name]])
    with pytest.raises(FileExistsError):
        smoke.run(config)


def test_io_trace_mixed_fallback_does_not_change_data_decisions():
    trace = smoke.IOTrace()
    for _ in range(2):
        trace.emit(logging.LogRecord('loader', logging.INFO, '', 0, 'Reading %d Parquet file(s) with cuDF', (1,), None))
    trace.emit(logging.LogRecord('loader', logging.WARNING, '', 0, 'Falling back to pandas for IO: example', (), None))
    assert trace.result()['io_backend'] == 'mixed_cudf_pandas'
    assert trace.result()['io_gpu_fallback']


@pytest.mark.parametrize('failure', ['hash', 'source', 'schema', 'quarantine', 'original_map'])
def test_identity_failures_prevent_subset_and_models_even_with_io_fallback(real_small_sources, gpu_doubles, monkeypatch, failure):
    config, _, _ = real_small_sources
    prepare_run(monkeypatch, config, fallback=True)
    root = Path(config.extra['frozen_splits_dir'])
    if failure == 'hash':
        path = root/'split_manifest.json'
        content = json.loads(path.read_text())
        content['split_hash'] = 'wrong'
        path.write_text(json.dumps(content))
    elif failure in ('source', 'schema', 'quarantine'):
        path = Path(config.extra['data_path'])/'Network_dataset_1.parquet'
        frame = pd.read_parquet(path)
        if failure == 'source':
            frame.loc[1, 'src_ip'] = 'changed'
        elif failure == 'schema':
            frame.loc[1, 'src_bytes'] = 'another-invalid-token'
        else:
            frame.loc[0, 'src_bytes'] = '10'
        frame.to_parquet(path, index=False)
    else:
        positions = np.load(root/'eligible_original_positions.npy')
        np.save(root/'eligible_original_positions.npy', positions+1)
    monkeypatch.setattr(smoke, 'fit_transform_preprocessing', lambda *args: pytest.fail('Models must not start'))
    with pytest.raises(ValueError):
        smoke.run(config)
    report = json.loads((Path(config.extra['output_dir'])/'integration_manifest.json').read_text())
    assert report['status'] == 'INVALID'


@pytest.mark.parametrize('failure', ['cpu_matrix', 'cpu_reducer', 'no_reducer', 'wrong_effective', 'serialization'])
def test_gpu_or_reduction_failures_are_invalid(real_small_sources, gpu_doubles, monkeypatch, failure):
    config, _, _ = real_small_sources
    prepare_run(monkeypatch, config)
    actual = smoke.fit_transform_preprocessing
    def broken(*args):
        X, bundle = actual(*args)
        if failure == 'cpu_matrix':
            X['val'] = np.asarray(X['val'])
        elif failure == 'cpu_reducer':
            bundle.svd = SimpleNamespace(n_components=9)
        elif failure == 'no_reducer':
            bundle.svd = None
        elif failure == 'wrong_effective':
            bundle.svd.n_components = 50
        else:
            (Path(config.extra['output_dir'])/'artifacts/preprocessing_bundle.joblib').unlink()
        return X, bundle
    monkeypatch.setattr(smoke, 'fit_transform_preprocessing', broken)
    monkeypatch.setattr(smoke, 'fit_predict_clustering', lambda *args: pytest.fail('Clustering must not start'))
    with pytest.raises((ValueError, FileNotFoundError)):
        smoke.run(config)
    report = json.loads((Path(config.extra['output_dir'])/'integration_manifest.json').read_text())
    assert report['status'] == 'INVALID'


def test_existing_gpu_pca_alternative_and_effective_zero(real_small_sources, gpu_doubles, monkeypatch, tmp_path):
    config, frame, frozen = real_small_sources
    monkeypatch.setattr(preprocessing, '_make_gpu_reducer', lambda n, seed: (GPUPCA(n, seed), 'PCA'))
    selected, local, positions = smoke.derive_subset(frozen, config.extra['smoke_rows'], 42)
    small = frame.iloc[positions]
    counts = {s: len(local[s]) for s in smoke.NAMES}
    for requested in (50, 0):
        cfg = config.with_updates(svd_components=requested)
        output = tmp_path / str(requested)
        X, bundle = preprocessing.fit_transform_preprocessing(small, local, list(smoke.BEHAVIORAL_STRICT_FEATURES), cfg, output)
        metadata = json.loads((output/'artifacts/preprocessing_metadata.json').read_text())
        result = smoke.verify_preprocessing(X, bundle, metadata, cfg, counts)
        assert result['svd_components_effective'] == (9 if requested else 0)
        assert result['reduction_algorithm'] == ('PCA' if requested else None)


def test_absent_classes_reported_without_resampling():
    df = pd.DataFrame({'label': [0, 1, 0], 'type': ['normal', 'attack', 'normal']})
    local = {s: np.array([i]) for i, s in enumerate(smoke.NAMES)}
    parent = {'distributions': {target: {'splits': {'train': {'counts': classes}}} for target, classes in
              [('label', {'0': 10, '1': 10}), ('type', {'normal': 10, 'attack': 9, 'rare': 1})]}}
    result = smoke.target_distributions(df, local, parent)
    assert result['type']['val']['counts']['rare'] == 0
    assert 'rare' in result['type']['val']['absent_classes']
    assert sum(result['type']['val']['counts'].values()) == 1


def test_assignment_export_cannot_be_marked_approved(tmp_path, monkeypatch):
    assignment = tmp_path/'cluster_assignments.csv'
    assignment.write_text('synthetic guard fixture; do not read')
    with pytest.raises(ValueError, match='Forbidden'):
        smoke.mark_artifacts(tmp_path)


@pytest.mark.parametrize('failure', ['cpu_kmeans', 'cpu_rf', 'missing_target', 'rf_dimension'])
def test_model_backend_and_target_guards(real_small_sources, gpu_doubles, monkeypatch, failure):
    config, _, _ = real_small_sources
    prepare_run(monkeypatch, config)
    if failure == 'cpu_kmeans':
        actual = smoke.fit_predict_clustering
        def wrong_clustering(*args):
            result = actual(*args)
            result['clusterer'] = CPUKMeans(n_clusters=30)
            return result
        monkeypatch.setattr(smoke, 'fit_predict_clustering', wrong_clustering)
    else:
        actual = smoke.train_random_forest
        def wrong_supervised(*args, **kwargs):
            result = actual(*args, **kwargs)
            if failure == 'missing_target':
                del result['targets']['cluster_id']
            else:
                import joblib
                path = Path(config.extra['output_dir'])/'artifacts/random_forest_label.joblib'
                bundle = joblib.load(path)
                if failure == 'cpu_rf':
                    bundle['model'] = CPURandomForest(n_estimators=20)
                else:
                    bundle['model'].n_features_in_ = 50
                joblib.dump(bundle, path)
            return result
        monkeypatch.setattr(smoke, 'train_random_forest', wrong_supervised)
    with pytest.raises(ValueError):
        smoke.run(config)
    report = json.loads((Path(config.extra['output_dir'])/'integration_manifest.json').read_text())
    assert report['status'] == 'INVALID'


def test_source_order_and_reuse_parameters_checked(real_small_sources):
    config, _, _ = real_small_sources
    root = Path(config.extra['frozen_splits_dir'])
    quality_path = root/'data_quality_population.json'
    reference = json.loads(quality_path.read_text())
    reference['files'].reverse()
    quality_path.write_text(json.dumps(reference))
    with pytest.raises(ValueError, match='order/set'):
        smoke.check_inputs(config)
    reference['files'].reverse()
    quality_path.write_text(json.dumps(reference))
    reuse_path = root/'reuse_split.yaml'
    reuse = yaml.safe_load(reuse_path.read_text())
    reuse['timestamp_unit'] = 'auto'
    reuse_path.write_text(yaml.safe_dump(reuse))
    with pytest.raises(ValueError, match='timestamp_unit'):
        smoke.check_inputs(config)
