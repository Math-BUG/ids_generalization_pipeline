"""Reproduce only the audited group split, without importing model runners."""
from __future__ import annotations

import argparse
import copy
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path, PurePosixPath
import platform
import sys
import tempfile

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))
from audit_ton_iot_splits import read_population
from ids_pipeline.config import load_config
from ids_pipeline.data_quality_policy import DATA_QUALITY_POLICY_VERSION
from ids_pipeline.schema import TON_IOT_SCHEMA_VERSION
from ids_pipeline.split_protocols import NAMES, digest_json, manifest, load_frozen_splits
from ids_pipeline.splitting import create_splits, save_splits
from ids_pipeline.utils import write_json


def relocate_reference(checkpoint, data_root, expected):
    """Change only absolute paths; retain source IDs, content hashes and file order."""
    reference = json.loads(Path(checkpoint).read_text(encoding='utf-8'))
    for key in ('original_rows', 'quarantined_rows', 'eligible_rows'):
        if reference[key] != expected[key]:
            raise ValueError(f'Increment 1 reference mismatch: {key}')
    if reference['schema_version'] != TON_IOT_SCHEMA_VERSION or reference['policy_version'] != DATA_QUALITY_POLICY_VERSION:
        raise ValueError('Increment 1 schema/policy version mismatch')
    if len(reference['files']) != expected['files']:
        raise ValueError('Unexpected number of reference files')
    root = Path(data_root).resolve(strict=True)
    relocated = copy.deepcopy(reference)
    paths = []
    for item in relocated['files']:
        source_id = PurePosixPath(item['source_id'])
        if source_id.is_absolute() or '..' in source_id.parts or '\\' in item['source_id']:
            raise ValueError(f'Invalid source-relative identity: {source_id}')
        path = (root / str(source_id)).resolve(strict=True)
        if root not in path.parents or not path.is_file() or path.suffix != '.parquet':
            raise ValueError(f'Invalid Parquet path: {path}')
        item['source_path'] = str(path)
        paths.append(path)
    if len(set(paths)) != len(paths) or set(paths) != {p.resolve() for p in root.rglob('*.parquet')}:
        raise ValueError('Parquet file set differs from the Increment 1 reference')
    # Later production loading sorts these paths; do not silently use another order.
    if paths != sorted(paths):
        raise ValueError('Reference file order differs from production lexicographical order')
    return relocated


def verify_reference(report, quality, expected):
    """Reject divergence even if the split satisfies the general protocol contract."""
    checks = {
        'protocol_version': report['protocol_version'] == expected['protocol_version'],
        'population_counts': all(quality[k] == expected[k] for k in ('original_rows', 'quarantined_rows', 'eligible_rows')),
        'file_count': len(quality['files']) == expected['files'],
        'population_identity': report['population']['source_population_id'] == quality['population_id'],
        'split_population_size': report['population']['eligible_input_rows'] == expected['eligible_rows'],
        'counts': report['counts'] == expected['counts'],
        'row_partition': report['row_validation']['ok'],
        'group_columns': report['parameters']['group_cols'] == ['src_ip', 'dst_ip', 'service', 'proto'],
        'group_overlap': report['group_overlap']['ok'] and all(
            report['group_overlap'][key] == 0 for key in ('train_val_overlap', 'train_test_overlap', 'val_test_overlap')),
        'protocol_validation': report['validation_ok'],
        'manifest_hash': report['split_hash'] == digest_json({
            k: report[k] for k in ('protocol_version', 'population', 'parameters', 'seed', 'index_hashes')}),
        'expected_hash': report['split_hash'] == expected['split_hash'],
    }
    for target in ('label', 'type'):
        classes = set(quality['eligible_target_distributions'][target]['values'])
        if target == 'type':
            checks['ten_type_classes'] = len(classes) == expected['type_classes']
        checks[f'{target}_coverage'] = all(
            set(report['distributions'][target]['splits'][s]['counts']) == classes
            and all(n > 0 for n in report['distributions'][target]['splits'][s]['counts'].values())
            for s in NAMES)
    return dict(ok=all(checks.values()), checks=checks, expected=expected,
                actual=dict(split_hash=report['split_hash'], counts=report['counts'], population=report['population']))


def freeze_population(df, config, staging, expected):
    """Freeze only after exact reference agreement; verify saved arrays by reloading."""
    splits = create_splits(df, config)
    report = manifest(df, splits, config, df.attrs.get('group_split_search'))
    verification = verify_reference(report, df.attrs['data_quality_report'], expected)
    write_json(staging / 'checkpoint_verification.json', verification)
    if not verification['ok']:
        write_json(staging / 'candidate_not_approved.json', report)
        failed = [key for key, ok in verification['checks'].items() if not ok]
        raise ValueError(f'Checkpoint differs from audited reference: {failed}')
    save_splits(splits, staging, df, config)
    reloaded = load_frozen_splits(staging, df, config)
    for name in NAMES:
        if not np.array_equal(reloaded[name], splits[name]):
            raise ValueError(f'Saved indices changed: {name}')
    np.save(staging / 'eligible_original_positions.npy', df.index.to_numpy(dtype='<i8'), allow_pickle=False)
    verification['saved_arrays_reloaded_and_verified'] = True
    write_json(staging / 'checkpoint_verification.json', verification)
    return report


def environment():
    packages = {}
    for name in ('numpy', 'pandas', 'scipy', 'pyarrow', 'scikit-learn', 'PyYAML'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    paths = ['scripts/audit_ton_iot_splits.py', 'scripts/validate_ton_iot_group_split.py'] + [
        f'src/ids_pipeline/{name}.py' for name in ('schema', 'dataset_schema', 'data_quality_policy', 'splitting', 'split_protocols')]
    return dict(python=platform.python_version(), platform=platform.platform(), packages=packages,
                code_sha256={p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in paths})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Original Increment 1 Parquet data_quality_report.json')
    parser.add_argument('--data-root', required=True, help='Cluster directory containing the same 23 Parquet files')
    parser.add_argument('--output', required=True, help='New persistent directory outside the repository')
    parser.add_argument('--config', default=str(REPO / 'configs/ton_iot_group_stratified_checkpoint.yaml'))
    parser.add_argument('--chunk-size', type=int, default=100000)
    parser.add_argument('--preflight-only', action='store_true', help='Check paths/reference metadata only; no data scan or split')
    args = parser.parse_args()
    config = load_config(args.config)
    if config.split_strategy != 'group_stratified' or config.extra.get('frozen_splits_dir'):
        raise ValueError('This checkpoint must regenerate only group_stratified')
    if config.extra.get('split_export_csv', True):
        raise ValueError('This checkpoint requires split_export_csv: false')
    if args.chunk_size < 1:
        raise ValueError('chunk-size must be positive')
    output = Path(args.output).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve(strict=True)
    if output == REPO or REPO in output.parents or output == data_root or data_root in output.parents:
        raise ValueError('Output must be outside the repository and source directory')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite checkpoint: {output}')
    expected = config.extra['checkpoint_reference']
    relocated = relocate_reference(args.checkpoint, data_root, expected)
    if args.preflight_only:
        print('Preflight OK: paths and reference metadata only; content hashes and split NOT yet verified.')
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}.pending_', dir=output.parent))
    print(f'Working directory (not approved): {staging}', flush=True)
    try:
        write_json(staging / 'execution_environment.json', environment())
        write_json(staging / 'reference_relocated.json', relocated)
        (staging / 'checkpoint_config.yaml').write_text(Path(args.config).read_text(encoding='utf-8'), encoding='utf-8')
        df = read_population(staging / 'reference_relocated.json', staging, args.chunk_size)
        report = freeze_population(df, config, staging, expected)
        p = report['parameters']
        reuse = {k: p[k] for k in ('test_size', 'val_size', 'group_cols', 'timestamp_col', 'timestamp_unit',
                                  'temporal_bucket_freq', 'label_col', 'type_col')}
        reuse.update(split_strategy='group_stratified', random_state=report['seed'],
                     group_split_tolerance=p['group_tolerance'], group_split_candidates=p['group_candidates'],
                     split_min_class_support=p['minimum_class_support'], split_small_support_threshold=p['small_support_threshold'],
                     frozen_splits_dir=str(output), split_export_csv=False)
        (staging / 'reuse_split.yaml').write_text(yaml.safe_dump(reuse, sort_keys=False), encoding='utf-8')
        if output.exists():
            raise FileExistsError(f'Refusing to overwrite checkpoint: {output}')
        staging.rename(output)
    except Exception as exc:
        write_json(staging / 'failure.json', {'error': str(exc), 'approved': False,
                   'diagnostics': getattr(exc, 'diagnostics', getattr(exc, 'report', None))})
        print(f'FAILED; diagnostics retained at {staging}', file=sys.stderr, flush=True)
        raise
    print(f'APPROVED: {output}\n{report["split_hash"]}', flush=True)


if __name__ == '__main__':
    main()
