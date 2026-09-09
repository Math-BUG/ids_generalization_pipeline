import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ids_pipeline.config import load_config
from ids_pipeline.data_quality_policy import DataQualityPopulation, source_fingerprint
from ids_pipeline.dataset_schema import normalize_dataset_schema
from ids_pipeline.split_protocols import manifest, load_frozen_splits
from ids_pipeline.splitting import create_splits


SPEC = importlib.util.spec_from_file_location('group_checkpoint', Path(__file__).resolve().parents[1] / 'scripts/validate_ton_iot_group_split.py')
checkpoint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checkpoint)


@pytest.fixture
def small_population(tmp_path):
    n = np.arange(120)
    raw = pd.DataFrame(dict(src_ip=[f'host{i//2}' for i in n], dst_ip='10.0.0.1',
                           service=np.where(n % 6 < 2, '-', 'http'), proto='tcp',
                           ts=1700000000+n, label=n % 2, type=np.where(n % 2, 'attack', 'normal'),
                           src_bytes='10'), index=n+37)
    path = tmp_path / 'Network_dataset_1.parquet'
    raw.to_parquet(path, index=False)
    population = DataQualityPopulation()
    population.register_file(path.name, source_fingerprint(path), source_path='C:/old/location/file.parquet')
    kept = population.apply(raw, source_id=path.name, row_offset=0)
    quality = population.report()
    df, _ = normalize_dataset_schema(kept)
    for col in ('src_ip', 'dst_ip', 'service', 'proto', 'type'):
        df[col] = df[col].astype('category')
    df.attrs.update(data_quality_population_id=quality['population_id'], data_quality_report=quality)
    config = load_config(checkpoint.REPO / 'configs/ton_iot_group_stratified_checkpoint.yaml')
    splits = create_splits(df, config)
    report = manifest(df, splits, config)
    expected = dict(protocol_version='group_stratified/2.0.0', files=1, original_rows=120,
                    quarantined_rows=0, eligible_rows=120, counts=report['counts'], type_classes=2,
                    split_hash=report['split_hash'])
    return df, config, report, quality, expected


def test_reference_config_matches_audited_parameters():
    config = load_config(checkpoint.REPO / 'configs/ton_iot_group_stratified_checkpoint.yaml')
    expected = config.extra['checkpoint_reference']
    assert config.timestamp_unit == 's'
    assert config.random_state == 42
    assert config.extra['split_export_csv'] is False
    assert expected['eligible_rows'] == sum(expected['counts'].values()) == 22338152
    assert expected['split_hash'] == 'sha256:828c57acef5ca3a953762a2dcf836dafd6b8dd513dd476c88ba924841a5f0a2a'


def test_relocate_preserves_identity_and_checks_extra_files(tmp_path, small_population):
    _, _, _, quality, expected = small_population
    path = tmp_path / 'reference.json'
    path.write_text(json.dumps(quality), encoding='utf-8')
    moved = checkpoint.relocate_reference(path, tmp_path, expected)
    assert moved['population_id'] == quality['population_id']
    assert moved['population_identity_manifest'] == quality['population_identity_manifest']
    assert moved['files'][0]['source_id'] == quality['files'][0]['source_id']
    assert moved['files'][0]['source_fingerprint'] == quality['files'][0]['source_fingerprint']
    assert moved['files'][0]['source_path'] == str((tmp_path / 'Network_dataset_1.parquet').resolve())
    (tmp_path / 'unexpected.parquet').write_bytes(b'not read')
    with pytest.raises(ValueError, match='file set'):
        checkpoint.relocate_reference(path, tmp_path, expected)


@pytest.mark.parametrize('change', ['hash', 'counts', 'overlap', 'coverage', 'rows', 'population'])
def test_reference_rejects_divergence(small_population, change):
    _, _, report, quality, expected = small_population
    report = copy.deepcopy(report)
    if change == 'hash':
        report['split_hash'] = 'sha256:' + '0'*64
    elif change == 'counts':
        report['counts']['train'] += 1
    elif change == 'overlap':
        report['group_overlap']['train_test_overlap'] = 1
    elif change == 'coverage':
        report['distributions']['type']['splits']['test']['counts']['attack'] = 0
    elif change == 'rows':
        report['row_validation']['ok'] = False
    else:
        report['population']['source_population_id'] = 'different'
    assert not checkpoint.verify_reference(report, quality, expected)['ok']


def test_freeze_reuses_indices_without_new_search(tmp_path, small_population, monkeypatch):
    df, config, _, _, expected = small_population
    directory = tmp_path / 'checkpoint'
    directory.mkdir()
    result = checkpoint.freeze_population(df, config, directory, expected)
    assert result['split_hash'] == expected['split_hash']
    assert not list((directory / 'splits').glob('*_indices.csv'))
    np.testing.assert_array_equal(np.load(directory / 'eligible_original_positions.npy'), df.index)
    verified = json.loads((directory / 'checkpoint_verification.json').read_text())
    assert verified['ok'] and verified['saved_arrays_reloaded_and_verified']
    from ids_pipeline import splitting
    def forbidden_search(*args):
        raise AssertionError('Frozen reuse must not regenerate the split')
    monkeypatch.setattr(splitting, 'group_split', forbidden_search)
    reused = create_splits(df, config.with_updates(extra={**config.extra, 'frozen_splits_dir': str(directory)}))
    # Production normalization uses strings instead of the audit's compact categories.
    normalized, _ = normalize_dataset_schema(df)
    loaded = load_frozen_splits(directory, normalized, config)
    for split in reused:
        np.testing.assert_array_equal(reused[split], loaded[split])


def test_wrong_expected_hash_does_not_freeze(tmp_path, small_population):
    df, config, _, _, expected = small_population
    expected = {**expected, 'split_hash': 'sha256:' + '0'*64}
    directory = tmp_path / 'rejected'
    directory.mkdir()
    with pytest.raises(ValueError, match='audited reference'):
        checkpoint.freeze_population(df, config, directory, expected)
    assert not (directory / 'split_manifest.json').exists()
    assert not (directory / 'splits').exists()
    assert (directory / 'candidate_not_approved.json').is_file()
