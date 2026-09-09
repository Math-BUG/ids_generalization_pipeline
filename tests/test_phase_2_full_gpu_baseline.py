"""Small fixtures only. GPU doubles here do not constitute CUDA validation."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier as CPUForest

from ids_pipeline import data_loading, splitting, split_protocols, supervised, representatives
from ids_pipeline.config import load_config

# Reuse tiny official-loader fixtures and explicitly labelled GPU doubles.
from test_phase_1e_real_gpu_smoke import gpu_doubles, real_small_sources, GPUKMeans, GPURandomForest

SPEC = importlib.util.spec_from_file_location('phase_2', Path(__file__).resolve().parents[1] / 'scripts/run_phase_2_full_gpu_baseline.py')
phase = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phase)


def base_config():
    return load_config(phase.REPO / phase.CONFIG_PATH)


def test_scientific_config_and_frozen_parameters(monkeypatch):
    monkeypatch.delenv('IDS_COMPUTE_BACKEND', raising=False)
    c = base_config()
    phase.validate_config(c)
    assert c.extra['scientific_result'] is True
    assert c.extra['integration_only'] is False
    assert c.extra['baseline_type'] == 'gpu_extension'
    assert c.extra['exact_replication_of_original_article'] is False
    assert c.targets == ['label', 'type', 'cluster_id']
    assert c.random_forest_estimators == 150 and not c.export_cluster_assignments
    ref = load_config(phase.REPO / 'configs/ton_iot_group_stratified_checkpoint.yaml')
    assert split_protocols.parameters(c) == split_protocols.parameters(ref)


@pytest.mark.parametrize('key,value', [
    ('compute_backend', 'auto'), ('feature_policy', 'all_except_labels'), ('timestamp_unit', 'auto'),
    ('selected_k', 31), ('random_state', 43), ('random_forest_estimators', 20),
    ('targets', ['label', 'type']), ('svd_components', 20), ('export_cluster_assignments', True),
    ('use_representatives_for_supervised', True), ('log1p_numeric', False),
    ('cluster_n_init', 10), ('cluster_batch_size', 4096), ('group_cols', ['src_ip'])])
def test_frozen_config_rejects_changes(key, value):
    with pytest.raises(ValueError, match='Frozen configuration'):
        phase.validate_config(base_config().with_updates(**{key: value}))


@pytest.mark.parametrize('key,value', [
    ('expected_split_hash', 'sha256:changed'), ('expected_eligible_rows', 100),
    ('expected_quarantined_rows', 0), ('expected_feature_policy_version', 'behavioral_strict/1'),
    ('integration_only', True), ('scientific_result', False), ('expected_type_classes', ['normal']),
    ('expected_full_split_rows', {'train': 1, 'val': 1, 'test': 1}),
    ('selection_budget', 1000), ('selection_methods', ['random']), ('sample_size', 10),
    ('smoke_rows', {'train': 10}), ('class_weight', 'balanced'), ('bootstrap', False)])
def test_extra_options_cannot_rewrite_contract(key, value):
    c = base_config()
    with pytest.raises(ValueError):
        phase.validate_config(c.with_updates(extra={**c.extra, key: value}))


def test_cpu_environment_override_rejected(monkeypatch):
    monkeypatch.setenv('IDS_COMPUTE_BACKEND', 'cpu')
    with pytest.raises(ValueError, match='override'):
        phase.validate_config(base_config())


def tiny_config(real_small_sources, tmp_path):
    old, df, splits = real_small_sources
    c = base_config()
    e = {**c.extra, **{k: old.extra[k] for k in phase.PATH_KEYS | {'expected_split_hash', 'expected_original_rows',
         'expected_quarantined_rows', 'expected_eligible_rows', 'expected_full_split_rows'}}}
    e.update(output_dir=str(tmp_path/'scientific'), expected_type_classes=['attack', 'normal'])
    return c.with_updates(extra=e), df, splits


def prepare(monkeypatch, config, fallback=False):
    # Only immutable production population expectations and external output paths
    # are relaxed. The real loader, frozen loader, feature/preprocessing functions,
    # cluster/RF orchestration and reporting run against 229 eligible fixture rows.
    monkeypatch.setattr(phase, 'validate_config', lambda c: None)
    monkeypatch.setattr(phase, 'REPO', Path(config.extra['output_dir']).parent/'not_repository')
    monkeypatch.setattr(phase, 'setup_logging', lambda *a: None)
    monkeypatch.setattr(phase, 'verify_provenance', lambda c: dict(git_commit_sha='a'*40, git_dirty=False,
        git_pending_changes=[], code_sha256={}, code_snapshot_hash='fixture'))
    def read(files, *, require_cudf):
        assert require_cudf
        data_loading.LOGGER.info('Reading %d Parquet file(s) with cuDF', len(files))
        if fallback:
            data_loading.LOGGER.warning('Falling back to pandas for IO; fixture')
        return data_loading._read_parquet_with_pandas(files)
    monkeypatch.setattr(data_loading, '_read_parquet_with_cudf', read)
    params = GPURandomForest.get_params
    monkeypatch.setattr(GPURandomForest, 'get_params', lambda self, deep=True: {
        **params(self, deep=deep), 'bootstrap': self.bootstrap, 'class_weight': self.class_weight})


@pytest.mark.parametrize('fallback', [False, True])
def test_full_population_orchestration_tiny_fixture(real_small_sources, gpu_doubles, tmp_path, monkeypatch, fallback):
    config, original, frozen = tiny_config(real_small_sources, tmp_path)
    prepare(monkeypatch, config, fallback)
    events, seen = [], {}
    def forbidden(*a, **k):
        pytest.fail('Generation/subsets/representatives must never run')
    monkeypatch.setattr(splitting, 'create_splits', forbidden)
    monkeypatch.setattr(splitting, 'save_splits', forbidden)
    monkeypatch.setattr(split_protocols, 'group_split', forbidden)
    monkeypatch.setattr(representatives, 'build_cluster_representatives', forbidden)
    official_load = data_loading.load_dataset
    def loader(*a, **kw):
        assert 'sample_size' not in kw
        assert a == (config.extra['data_path'],)
        events.append('load')
        return official_load(*a, **kw)
    monkeypatch.setattr(data_loading, 'load_dataset', loader)
    official_frozen = phase.load_frozen_splits
    def frozen_load(*a):
        result = official_frozen(*a)
        seen['splits'] = result
        events.append('frozen')
        return result
    monkeypatch.setattr(phase, 'load_frozen_splits', frozen_load)
    official_prep = phase.fit_transform_preprocessing
    def prep(df, splits, *a):
        assert len(df) == 229 and splits is seen['splits']
        np.testing.assert_array_equal(df.index, original.index)
        events.append('preprocess')
        return official_prep(df, splits, *a)
    monkeypatch.setattr(phase, 'fit_transform_preprocessing', prep)
    official_fit = supervised._fit_model
    def fit(X_train, y_train, *a):
        assert len(X_train) == len(y_train) == len(frozen['train'])
        seen.setdefault('rf_matrices', []).append(X_train)
        events.append('rf')
        return official_fit(X_train, y_train, *a)
    monkeypatch.setattr(supervised, '_fit_model', fit)
    report = phase.run(config)
    assert events[:3] == ['load', 'frozen', 'preprocess'] and events.count('rf') == 3
    assert all(x is seen['rf_matrices'][0] for x in seen['rf_matrices'])
    assert report['status'] == 'APPROVED' and report['scientific_result'] is True
    assert report['integration_only'] is False and report['baseline_type'] == 'gpu_extension'
    assert report['data']['eligible_rows'] == 229
    assert report['data']['io_gpu_fallback'] is fallback
    assert report['data']['io_backend'] == ('pandas' if fallback else 'cudf')
    assert report['split']['hash_match'] and report['split']['regenerated'] is False
    assert report['clustering']['original_algorithm'] == 'MiniBatchKMeans'
    assert report['clustering']['algorithmic_difference_from_original'] is True
    assert report['supervised']['difference_from_original_balanced_subsample'] is True
    assert report['supervised']['class_weight'] is None
    assert report['preprocessing']['end_to_end_gpu'] is False
    assert report['preprocessing']['svd_components_requested'] == 50
    assert report['preprocessing']['svd_components_effective'] == 9  # 8 quantities + 2 states - 1
    out = Path(config.extra['output_dir'])
    assert not (out/'cluster_assignments.csv').exists()
    assert not (out/'splits').exists() and not (out/'smoke_indices').exists()
    by_class = json.loads((out/'metrics_by_class.json').read_text())
    for target in config.targets:
        for s in ('val', 'test'):
            assert sum(v['support'] for v in by_class[target][s]['per_class'].values()) == len(frozen[s])
    for s in phase.NAMES:
        np.testing.assert_array_equal(seen['splits'][s], frozen[s])
    assert json.loads((out/'experiment_manifest.json').read_text())['status'] == 'APPROVED'


@pytest.mark.parametrize('mutation', ['population', 'quarantine', 'hash', 'counts', 'coverage', 'overlap', 'schema', 'identity', 'map', 'missing_frozen', 'feature', 'gpu'])
def test_guards_abort_before_learning(real_small_sources, gpu_doubles, tmp_path, monkeypatch, mutation):
    config, _, _ = tiny_config(real_small_sources, tmp_path)
    prepare(monkeypatch, config)
    def no_training(*a, **k):
        pytest.fail('Invalid inputs reached learning')
    monkeypatch.setattr(phase, 'fit_transform_preprocessing', no_training)
    monkeypatch.setattr(phase, 'fit_predict_clustering', no_training)
    monkeypatch.setattr(supervised, 'train_random_forest', no_training)
    if mutation in ('population', 'quarantine'):
        key = 'expected_eligible_rows' if mutation == 'population' else 'expected_quarantined_rows'
        config = config.with_updates(extra={**config.extra, key: -1})
    elif mutation in ('hash', 'counts'):
        key, value = ('expected_split_hash', 'sha256:wrong') if mutation=='hash' else ('expected_full_split_rows', dict(train=1,val=1,test=1))
        config = config.with_updates(extra={**config.extra, key: value})
    elif mutation in ('coverage', 'overlap'):
        original_manifest = phase.manifest
        def bad_manifest(*a):
            result = original_manifest(*a)
            if mutation == 'coverage':
                result['distributions']['type']['splits']['train']['counts']['normal'] = 0
            else:
                result['group_overlap']['train_val_overlap'] = 1
            return result
        monkeypatch.setattr(phase, 'manifest', bad_manifest)
    elif mutation in ('schema', 'identity'):
        original_load = data_loading.load_dataset
        def bad_data(*a, **kw):
            df = original_load(*a, **kw)
            df.attrs['schema_version' if mutation=='schema' else 'data_quality_population_id'] = 'invalid'
            return df
        monkeypatch.setattr(data_loading, 'load_dataset', bad_data)
    elif mutation == 'map':
        p = Path(config.extra['frozen_splits_dir'])/'eligible_original_positions.npy'
        arr=np.load(p); arr[0] += 1; np.save(p,arr)
    elif mutation == 'missing_frozen':
        (Path(config.extra['frozen_splits_dir'])/'splits/train_indices.npy').unlink()
    elif mutation == 'feature':
        monkeypatch.setattr(phase.FeaturePolicy, 'select_features', lambda *a, **kw: ['src_ip'])
    elif mutation == 'gpu':
        monkeypatch.setattr(phase, 'environment', lambda c: (_ for _ in ()).throw(RuntimeError('GPU unavailable')))
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        phase.run(config)
    report=json.loads((Path(config.extra['output_dir'])/'experiment_manifest.json').read_text())
    assert report['status']=='INVALID'


def test_existing_output_not_overwritten(tmp_path):
    output=tmp_path/'old'; output.mkdir()
    (output/'manifest.json').write_text('previous')
    c=base_config(); c=c.with_updates(extra={**c.extra,'output_dir':str(output)})
    with pytest.raises(ValueError, match='Refusing existing output.*bytes=8'):
        phase.run(c)
    assert (output/'manifest.json').read_text()=='previous'
    assert not (output/'experiment_manifest.json').exists()


def test_forbidden_csv_detected_by_name_without_reading(tmp_path, monkeypatch):
    nested=tmp_path/'nested'; nested.mkdir()
    (nested/'cluster_assignments.csv').write_bytes(b'not to be read')
    monkeypatch.setattr(Path, 'read_bytes', lambda p: pytest.fail('Must not read assignments'))
    with pytest.raises(ValueError, match='Forbidden'):
        phase.artifact_inventory(tmp_path)


def test_missing_artifacts_not_approved(tmp_path):
    with pytest.raises(ValueError, match='Missing/empty'):
        phase.artifact_inventory(tmp_path, complete=True)


def test_metrics_from_counts_match_sklearn_with_unpredicted_class():
    actual=np.array(['normal','mitm','normal','ransomware','backdoor','backdoor'])
    pred=np.array(['normal','normal','normal','backdoor','backdoor','normal'])
    cm=supervised._confusion_dataframe(actual,pred)
    report=phase.metrics_from_confusion(cm)
    assert report['per_class']['mitm']['recall']==0
    assert report['per_class']['ransomware']['f1']==0
    assert report['global_metrics']['accuracy']==accuracy_score(actual,pred)
    for avg in ['macro','weighted']:
        expected=precision_recall_fscore_support(actual,pred,average=avg,zero_division=0)
        for name,value in zip(['precision','recall','f1'],expected[:3]):
            assert report['global_metrics'][f'{avg}_{name}']==pytest.approx(value)


def test_source_provenance_is_reproducible_and_detects_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(phase,'REPO',tmp_path)
    (tmp_path/'code.py').write_text('original')
    monkeypatch.setattr(phase,'source_files',lambda:['code.py'])
    monkeypatch.setattr(phase,'git_output',lambda *a: 'a'*40 if a[0]=='rev-parse' else ' M code.py')
    first=phase.write_provenance(); second=phase.write_provenance()
    assert first['code_snapshot_hash']==second['code_snapshot_hash']
    assert first['git_pending_changes']==[' M code.py']
    config=load_config(Path(__file__).resolve().parents[1]/phase.CONFIG_PATH)
    assert phase.verify_provenance(config)['git_dirty'] is True
    (tmp_path/'code.py').write_text('changed')
    with pytest.raises(ValueError,match='snapshot differs'):
        phase.verify_provenance(config)


def test_cpu_estimator_rejected(gpu_doubles):
    result=dict(clusterer=object(),metrics={},labels={},distances={})
    with pytest.raises(ValueError,match='not cuML'):
        phase.verify_clustering(result,dict(train=2,val=1,test=1),base_config())


def test_cpu_rf_artifact_rejected_before_reporting(gpu_doubles, monkeypatch, tmp_path):
    monkeypatch.setattr(joblib, 'load', lambda p: dict(model=CPUForest(), model_name='cuML RandomForestClassifier'))
    c=base_config()
    metrics=dict(backend='gpu', use_representatives_for_supervised=False,
                 targets={t: dict(train_n_samples_original=2, train_n_samples_used=2) for t in c.targets})
    with pytest.raises(ValueError, match='not cuML/GPU'):
        phase.audit_supervised(metrics, {}, None, {'train': np.array([0,1])}, c, {}, tmp_path)


@pytest.mark.parametrize('key,value', [('inertia', float('nan')), ('fit_n_samples', 1), ('davies_bouldin_train', float('inf'))])
def test_invalid_clustering_metrics_rejected(gpu_doubles, tmp_path, key, value):
    m=dict(backend='gpu', algorithm='cuML KMeans', fit_split='train', fit_n_samples=2, inertia=1.0)
    m[key]=value
    result=dict(clusterer=GPUKMeans(), metrics=m,
                labels={'train':np.array([0,1]),'val':np.array([0]),'test':np.array([1])},
                distances={'train':np.ones(2),'val':np.ones(1),'test':np.ones(1)})
    with pytest.raises(ValueError):
        phase.clustering_evidence(result,dict(train=2,val=1,test=1),base_config(),tmp_path)


def test_missing_binary_metric_marks_run_invalid(real_small_sources, gpu_doubles, tmp_path, monkeypatch):
    c, _, _=tiny_config(real_small_sources,tmp_path)
    prepare(monkeypatch,c)
    real_train=supervised.train_random_forest
    def train(*a, **kw):
        result=real_train(*a,**kw)
        result['targets']['label']['val']['roc_auc']=None
        return result
    monkeypatch.setattr(supervised,'train_random_forest',train)
    with pytest.raises(ValueError,match='Missing binary metric'):
        phase.run(c)
    report=json.loads((Path(c.extra['output_dir'])/'experiment_manifest.json').read_text())
    assert report['status']=='INVALID'


@pytest.mark.parametrize('violation', ['cpu_kmeans', 'csv'])
def test_invalid_manifest_on_clustering_violation(real_small_sources, gpu_doubles, tmp_path, monkeypatch, violation):
    c, _, _=tiny_config(real_small_sources,tmp_path)
    prepare(monkeypatch,c)
    real_cluster=phase.fit_predict_clustering
    def bad_cluster(*args):
        result=real_cluster(*args)
        if violation=='cpu_kmeans':
            result['clusterer']=object()
        else:
            (args[-1]/'cluster_assignments.csv').write_text('fixture forbidden export')
        return result
    monkeypatch.setattr(phase,'fit_predict_clustering',bad_cluster)
    monkeypatch.setattr(supervised,'train_random_forest',lambda *a, **k: pytest.fail('RF ran after invalid clustering'))
    with pytest.raises(ValueError):
        phase.run(c)
    out=Path(c.extra['output_dir'])
    report=json.loads((out/'experiment_manifest.json').read_text())
    assert report['status']=='INVALID'
    if violation=='csv':
        assert report['validity']['cluster_assignments_created'] is True
        assert (out/'cluster_assignments.csv').exists()  # do not erase evidence


def test_slurm_reservation_and_no_smoke_or_cli():
    text=(phase.REPO/'scripts/run_phase_2_full_gpu_baseline.slm').read_text()
    for expected in ['--cpus-per-task=32','--mem=256G','--gres=gpu:1','--time=12:00:00',
                     'Python/3.13.5-GCCcore-14.3.0','CUDA/12.9.1','source .venv/bin/activate',
                     'IDS_COMPUTE_BACKEND=gpu','/usr/bin/time -v','exit "$run_status"']:
        assert expected in text
    assert 'run_phase_1e_real_gpu_smoke.py' not in text and '--stage all' not in text
