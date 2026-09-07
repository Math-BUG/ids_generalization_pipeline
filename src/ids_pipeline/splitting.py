"""Reproducible split API; versioned contracts live in split_protocols."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import PipelineConfig
from .utils import ensure_dir, write_json
from .split_protocols import (NAMES, SplitValidationError, group_split, temporal_split,
    group_codes, overlap_from_codes, chronology, manifest, load_frozen_splits, row_validation)


def create_splits(df, config):
    frozen = config.extra.get('frozen_splits_dir')
    if frozen:
        return load_frozen_splits(frozen, df, config)
    if config.split_strategy == 'random_stratified':
        splits = _random_stratified_split(df, config)
    elif config.split_strategy == 'group_stratified':
        splits, search = group_split(df, config)
        df.attrs['group_split_search'] = search
    elif config.split_strategy in ('temporal', 'temporal_per_class'):
        splits = temporal_split(df, config, config.split_strategy == 'temporal_per_class')
    else:
        raise ValueError('Unknown split strategy')
    report = manifest(df, splits, config, df.attrs.get('group_split_search'))
    if not report['validation_ok']:
        raise SplitValidationError('Split failed protocol validation', report, splits)
    return splits


def save_splits(splits, output_dir, df=None, config=None):
    root = ensure_dir(output_dir)
    split_dir = ensure_dir(root/'splits')
    report = manifest(df, splits, config, df.attrs.get('group_split_search')) if df is not None and config is not None else None
    if report is not None:
        if not report['validation_ok']:
            write_json(root/'split_validation_failure.json', report)
            raise SplitValidationError('Refusing to freeze an invalid split',report,splits)
        previous = root/'split_manifest.json'
        if previous.exists():
            import json
            saved = json.loads(previous.read_text(encoding='utf-8'))
            if saved['split_hash'] != report['split_hash']:
                raise ValueError('Refusing to overwrite different frozen splits; choose a new output directory')
    for name, idx in splits.items():
        np.save(split_dir/f'{name}_indices.npy', np.asarray(idx,dtype='<i8'), allow_pickle=False)
        # Legacy small-file consumers remain supported; large exports use the exact binary arrays.
        if config is None or config.extra.get('split_export_csv', True):
            pd.DataFrame({'row_index':idx}).to_csv(split_dir/f'{name}_indices.csv', index=False)
    write_json(root/'split_summary.json', split_summary(splits,config))
    if df is not None and config is not None:
        if config.split_strategy == 'group_stratified':
            write_json(split_dir/'group_overlap_report.json',report['group_overlap'])
        if config.split_strategy == 'temporal':
            write_json(split_dir/'temporal_split_report.json',temporal_split_report(df,splits,config))
        if config.split_strategy == 'temporal_per_class':
            per_report, per_class = temporal_per_class_report(df,splits,config)
            write_json(split_dir/'temporal_per_class_report.json',per_report)
            per_class.to_csv(split_dir/'temporal_per_class_report.csv',index=False)
        cols = [c for c in (config.label_col,config.type_col) if c in df]
        if cols:
            distribution = target_distribution_table(df,splits,cols)
            distribution.to_csv(split_dir/'target_distribution.csv',index=False)
            for col in cols:
                target_distribution_pivot(distribution,col).to_csv(split_dir/f'target_distribution_{col}.csv',index=False)
    # Publish the manifest only after all arrays/reports were successfully saved.
    if report is not None:
        write_json(root/'split_manifest.json', report)


def split_summary(splits,config=None):
    out = {s:int(len(idx)) for s,idx in splits.items()}
    out['proportions'] = {s:len(idx)/sum(map(len,splits.values())) for s,idx in splits.items()}
    if config is not None:
        out.update(split_strategy=config.split_strategy,test_size=config.test_size,val_size=config.val_size,
                   timestamp_col=config.timestamp_col,timestamp_unit=config.timestamp_unit,temporal_bucket_freq=config.temporal_bucket_freq)
    return out


def group_overlap_report(df,splits,group_cols):
    codes,_ = group_codes(df,group_cols)
    return overlap_from_codes(codes,splits)


def temporal_split_report(df,splits,config):
    out = chronology(df,splits,config)
    out.update(timestamp_col=config.timestamp_col,timestamp_unit=config.timestamp_unit,temporal_bucket_freq=config.temporal_bucket_freq)
    for s in NAMES:
        out[f'n_{s}_time_buckets'] = out['intervals'][s]['buckets']
        out[f'{s}_timestamp_min'] = out['intervals'][s]['start_utc']
        out[f'{s}_timestamp_max'] = out['intervals'][s]['end_utc']
    return out


def temporal_per_class_report(df,splits,config):
    target = config.type_col if config.type_col in df else config.label_col
    membership = np.full(len(df),-1,dtype=np.int8)
    for i,s in enumerate(NAMES): membership[splits[s]] = i
    rows=[]
    missing={s:[] for s in NAMES}
    invalid=[]
    for c,pos in df.groupby(target,sort=True,observed=True).indices.items():
        sub = df.iloc[pos]
        sub_splits = {s:np.flatnonzero(membership[pos]==i) for i,s in enumerate(NAMES)}
        report=chronology(sub,sub_splits,config)
        row={'class_value':str(c),'chronology_preserved':report['ok']}
        for s in NAMES:
            interval=report['intervals'][s]
            row.update({f'{s}_count':len(sub_splits[s]),f'{s}_timestamp_min':interval['start_utc'],
                        f'{s}_timestamp_max':interval['end_utc'],f'{s}_buckets':interval['buckets']})
            if not len(sub_splits[s]): missing[s].append(str(c))
        if not report['ok']: invalid.append(str(c))
        rows.append(row)
    global_report=chronology(df,splits,config)
    out=dict(target_col=target,n_classes=len(rows),invalid_classes=invalid,
             global_chronology_preserved=global_report['global_chronology_preserved'],global_chronology=global_report,
             auxiliary=True,ok=not invalid and row_validation(splits,len(df))['ok'])
    out.update({f'classes_missing_{s}':missing[s] for s in NAMES})
    return out,pd.DataFrame(rows)


def _random_stratified_split(df: pd.DataFrame, config: PipelineConfig) -> dict[str, np.ndarray]:
    indices = np.arange(len(df))
    stratify = _safe_stratify(df, config.label_col)
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify,
    )

    train_val_df = df.iloc[train_val_idx]
    train_val_stratify = _safe_stratify(train_val_df, config.label_col)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=config.val_size,
        random_state=config.random_state,
        stratify=train_val_stratify,
    )
    return _sorted_splits(train_idx, val_idx, test_idx)


def _safe_stratify(df: pd.DataFrame, label_col: str) -> pd.Series | None:
    if label_col not in df.columns:
        return None
    y = df[label_col]
    counts = y.value_counts(dropna=False)
    return y if len(counts) > 1 and counts.min() >= 2 else None


def _sorted_splits(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "train": np.sort(np.asarray(train_idx, dtype=int)),
        "val": np.sort(np.asarray(val_idx, dtype=int)),
        "test": np.sort(np.asarray(test_idx, dtype=int)),
    }


def target_distribution_table(
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    target_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split, idx in splits.items():
        split_df = df.iloc[idx]
        split_n = int(len(split_df))
        for target_col in target_cols:
            counts = split_df[target_col].astype(str).value_counts(dropna=False).sort_index()
            for class_value, count in counts.items():
                rows.append(
                    {
                        "split": split,
                        "target": target_col,
                        "class_value": class_value,
                        "count": int(count),
                        "split_total": split_n,
                        "fraction": float(count / split_n) if split_n else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def target_distribution_pivot(distribution: pd.DataFrame, target_col: str) -> pd.DataFrame:
    subset = distribution[distribution["target"] == target_col].copy()
    if subset.empty:
        return pd.DataFrame()

    count_pivot = subset.pivot_table(
        index="class_value",
        columns="split",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )
    fraction_pivot = subset.pivot_table(
        index="class_value",
        columns="split",
        values="fraction",
        aggfunc="sum",
        fill_value=0.0,
    )
    for split in ["train", "val", "test"]:
        if split not in count_pivot.columns:
            count_pivot[split] = 0
        if split not in fraction_pivot.columns:
            fraction_pivot[split] = 0.0

    out = pd.DataFrame({"class_value": count_pivot.index.astype(str)})
    for split in ["train", "val", "test"]:
        out[f"{split}_count"] = count_pivot[split].astype(int).to_numpy()
        out[f"{split}_fraction"] = fraction_pivot[split].astype(float).to_numpy()
    out["total_count"] = out[["train_count", "val_count", "test_count"]].sum(axis=1)
    return out.sort_values("class_value").reset_index(drop=True)


def parse_timestamp_series(series: pd.Series, *, unit: str | None = "auto") -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce", utc=True)
    if unit and unit != "auto":
        return pd.to_datetime(pd.to_numeric(series, errors="coerce"), unit=unit, errors="coerce", utc=True)

    inferred = infer_timestamp_unit(series)
    if inferred is not None:
        numeric = pd.to_numeric(series, errors="coerce")
        return pd.to_datetime(numeric, unit=inferred, errors="coerce", utc=True)
    return pd.to_datetime(series, errors="coerce", utc=True)


def infer_timestamp_unit(series: pd.Series) -> str | None:
    numeric = pd.to_numeric(series, errors="coerce")
    valid_ratio = float(numeric.notna().mean()) if len(numeric) else 0.0
    if valid_ratio < 0.9:
        return None

    non_null = numeric.dropna()
    if non_null.empty:
        return None
    median_abs = float(non_null.abs().median())

    # Unix epoch magnitudes: seconds ~1e9, ms ~1e12, us ~1e15, ns ~1e18.
    if median_abs >= 1e17:
        return "ns"
    if median_abs >= 1e14:
        return "us"
    if median_abs >= 1e11:
        return "ms"
    return "s"
