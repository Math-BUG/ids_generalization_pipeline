"""Reproducible train/validation/test splitting strategies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .config import PipelineConfig
from .utils import ensure_dir, stable_series_to_group_key, write_json


def create_splits(df: pd.DataFrame, config: PipelineConfig) -> dict[str, np.ndarray]:
    if config.split_strategy == "random_stratified":
        return _random_stratified_split(df, config)
    if config.split_strategy == "group_stratified":
        return _group_stratified_split(df, config)
    if config.split_strategy == "temporal":
        return _temporal_split(df, config)
    if config.split_strategy == "temporal_per_class":
        return _temporal_per_class_split(df, config)
    raise ValueError(
        "split_strategy must be one of: random_stratified, group_stratified, temporal, temporal_per_class"
    )


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


def _group_stratified_split(df: pd.DataFrame, config: PipelineConfig) -> dict[str, np.ndarray]:
    if not config.group_cols:
        raise ValueError("group_stratified requires config.group_cols")

    indices = np.arange(len(df))
    y = _split_target(df, config)
    groups = stable_series_to_group_key(df, config.group_cols).astype(str).to_numpy()

    train_val_idx, test_idx = _stratified_group_holdout(
        indices,
        y,
        groups,
        holdout_size=config.test_size,
        random_state=config.random_state,
    )
    train_idx, val_idx = _stratified_group_holdout(
        train_val_idx,
        y[train_val_idx],
        groups[train_val_idx],
        holdout_size=config.val_size,
        random_state=config.random_state + 1,
    )
    splits = _sorted_splits(train_idx, val_idx, test_idx)
    report = group_overlap_report(df, splits, config.group_cols)
    if not report["ok"]:
        raise ValueError(f"group_stratified produced overlapping groups: {report}")
    return splits


def _temporal_split(df: pd.DataFrame, config: PipelineConfig) -> dict[str, np.ndarray]:
    if config.timestamp_col not in df.columns:
        raise ValueError(f"temporal split requires timestamp_col={config.timestamp_col!r}")

    parsed = parse_timestamp_series(df[config.timestamp_col], unit=config.timestamp_unit)
    temporal_frame = pd.DataFrame({"row_index": np.arange(len(df)), "timestamp": parsed})
    if config.temporal_bucket_freq:
        temporal_frame["time_bucket"] = parsed.dt.floor(config.temporal_bucket_freq)
        temporal_frame["time_bucket_key"] = temporal_frame["time_bucket"].astype("string").fillna("<NaT>")
        return _temporal_bucket_split(temporal_frame, config)

    ordered = (
        temporal_frame.sort_values(["timestamp", "row_index"], na_position="last")["row_index"].to_numpy()
    )

    n = len(ordered)
    test_n = _bounded_count(n, config.test_size)
    val_n = _bounded_count(n - test_n, config.val_size)
    train_n = n - val_n - test_n
    if train_n <= 0:
        raise ValueError("temporal split produced an empty train split")

    train_idx = ordered[:train_n]
    val_idx = ordered[train_n : train_n + val_n]
    test_idx = ordered[train_n + val_n :]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def _temporal_per_class_split(df: pd.DataFrame, config: PipelineConfig) -> dict[str, np.ndarray]:
    if config.timestamp_col not in df.columns:
        raise ValueError(f"temporal_per_class split requires timestamp_col={config.timestamp_col!r}")
    target_col = config.type_col if config.type_col in df.columns else config.label_col
    if target_col not in df.columns:
        raise ValueError("temporal_per_class requires type_col or label_col in the dataframe")

    parsed = parse_timestamp_series(df[config.timestamp_col], unit=config.timestamp_unit)
    frame = pd.DataFrame(
        {
            "row_index": np.arange(len(df)),
            "timestamp": parsed,
            "class_value": df[target_col].astype(str).to_numpy(),
        }
    )
    if config.temporal_bucket_freq:
        frame["time_bucket"] = parsed.dt.floor(config.temporal_bucket_freq)
        frame["time_bucket_key"] = frame["time_bucket"].astype("string").fillna("<NaT>")
    else:
        frame["time_bucket_key"] = frame["timestamp"].astype("string").fillna("<NaT>")

    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    ordered = frame.sort_values(["class_value", "time_bucket_key", "timestamp", "row_index"], na_position="last")
    for _, class_frame in ordered.groupby("class_value", sort=True, dropna=False):
        bucket_rows = [
            bucket_frame["row_index"].to_numpy(dtype=int)
            for _, bucket_frame in class_frame.groupby("time_bucket_key", sort=False, dropna=False)
        ]
        class_train, class_val, class_test = _split_ordered_units(bucket_rows, config)
        train_parts.append(class_train)
        val_parts.append(class_val)
        test_parts.append(class_test)

    return {
        "train": np.sort(np.concatenate(train_parts).astype(int)),
        "val": np.sort(np.concatenate(val_parts).astype(int)),
        "test": np.sort(np.concatenate(test_parts).astype(int)),
    }


def _temporal_bucket_split(temporal_frame: pd.DataFrame, config: PipelineConfig) -> dict[str, np.ndarray]:
    ordered = temporal_frame.sort_values(["time_bucket", "timestamp", "row_index"], na_position="last")
    bucket_rows = [
        group["row_index"].to_numpy(dtype=int)
        for _, group in ordered.groupby("time_bucket_key", sort=False, dropna=False)
    ]
    if len(bucket_rows) < 3:
        sample_buckets = ordered["time_bucket_key"].drop_duplicates().head(5).tolist()
        raise ValueError(
            "temporal bucket split requires at least three distinct time buckets. "
            f"Observed {len(bucket_rows)} with timestamp_col={config.timestamp_col!r}, "
            f"timestamp_unit={config.timestamp_unit!r}, temporal_bucket_freq={config.temporal_bucket_freq!r}, "
            f"sample_buckets={sample_buckets}. If the timestamp is numeric Unix time, use timestamp_unit='auto' "
            "or set timestamp_unit explicitly to 's', 'ms', 'us', or 'ns'."
        )

    train_buckets, val_buckets, test_buckets = _split_ordered_bucket_rows(bucket_rows, config)
    return {
        "train": np.concatenate(train_buckets).astype(int),
        "val": np.concatenate(val_buckets).astype(int),
        "test": np.concatenate(test_buckets).astype(int),
    }


def _split_ordered_units(
    ordered_units: list[np.ndarray],
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_units, val_units, test_units = _split_ordered_bucket_rows(ordered_units, config)
    return (
        np.concatenate(train_units).astype(int),
        np.concatenate(val_units).astype(int),
        np.concatenate(test_units).astype(int),
    )


def _split_ordered_bucket_rows(
    bucket_rows: list[np.ndarray],
    config: PipelineConfig,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    if len(bucket_rows) < 3:
        row_units = [np.asarray([row], dtype=int) for rows in bucket_rows for row in rows]
        if len(row_units) < 3:
            raise ValueError("Temporal split requires at least three rows/buckets per class.")
        bucket_rows = row_units

    n = int(sum(len(rows) for rows in bucket_rows))
    test_target = _bounded_count(n, config.test_size)
    val_target = _bounded_count(n - test_target, config.val_size)

    test_units, remaining = _take_buckets_from_end(bucket_rows, test_target)
    val_units, train_units = _take_buckets_from_end(remaining, val_target)
    if not train_units or not val_units or not test_units:
        raise ValueError("Temporal split produced an empty train/val/test split")
    return train_units, val_units, test_units


def _take_buckets_from_end(
    bucket_rows: list[np.ndarray],
    target_rows: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    selected_reversed: list[np.ndarray] = []
    selected_n = 0
    remaining = list(bucket_rows)
    while remaining and selected_n < target_rows:
        rows = remaining.pop()
        selected_reversed.append(rows)
        selected_n += len(rows)
    selected = list(reversed(selected_reversed))
    return selected, remaining


def _bounded_count(n: int, fraction: float) -> int:
    if n <= 0:
        return 0
    count = int(round(n * fraction))
    return max(1, min(count, n - 1))


def _stratified_group_holdout(
    indices: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    holdout_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("group_stratified requires at least two distinct groups")

    n_splits = max(2, int(round(1.0 / max(holdout_size, 1e-6))))
    n_splits = min(n_splits, len(unique_groups))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_pos, holdout_pos = next(splitter.split(indices, y, groups))
    return indices[train_pos], indices[holdout_pos]


def _split_target(df: pd.DataFrame, config: PipelineConfig) -> np.ndarray:
    if config.type_col in df.columns:
        return df[config.type_col].astype(str).to_numpy()
    if config.label_col in df.columns:
        return df[config.label_col].astype(str).to_numpy()
    return np.zeros(len(df), dtype=int)


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


def save_splits(
    splits: dict[str, np.ndarray],
    output_dir: str | Path,
    df: pd.DataFrame | None = None,
    config: PipelineConfig | None = None,
) -> None:
    split_dir = ensure_dir(Path(output_dir) / "splits")
    for name, idx in splits.items():
        pd.DataFrame({"row_index": idx}).to_csv(split_dir / f"{name}_indices.csv", index=False)
    write_json(Path(output_dir) / "split_summary.json", split_summary(splits, config))

    if df is not None and config is not None and config.split_strategy == "group_stratified":
        report = group_overlap_report(df, splits, config.group_cols)
        write_json(split_dir / "group_overlap_report.json", report)
        if not report["ok"]:
            raise ValueError(f"Group overlap detected between splits: {report}")

    if df is not None and config is not None and config.split_strategy == "temporal":
        report = temporal_split_report(df, splits, config)
        write_json(split_dir / "temporal_split_report.json", report)
        if not report["ok"]:
            raise ValueError(f"Temporal split overlap detected: {report}")

    if df is not None and config is not None and config.split_strategy == "temporal_per_class":
        report, per_class = temporal_per_class_report(df, splits, config)
        write_json(split_dir / "temporal_per_class_report.json", report)
        per_class.to_csv(split_dir / "temporal_per_class_report.csv", index=False)
        if not report["ok"]:
            raise ValueError(f"Temporal per-class split issue detected: {report}")

    if df is not None and config is not None:
        target_cols = [c for c in [config.label_col, config.type_col] if c in df.columns]
        if target_cols:
            distribution = target_distribution_table(df, splits, target_cols)
            distribution.to_csv(split_dir / "target_distribution.csv", index=False)
            for target_col in target_cols:
                pivot = target_distribution_pivot(distribution, target_col)
                pivot.to_csv(split_dir / f"target_distribution_{target_col}.csv", index=False)


def split_summary(
    splits: dict[str, np.ndarray],
    config: PipelineConfig | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {name: int(len(idx)) for name, idx in splits.items()}
    if config is not None:
        payload["split_strategy"] = config.split_strategy
        payload["test_size"] = config.test_size
        payload["val_size"] = config.val_size
        payload["timestamp_col"] = config.timestamp_col
        payload["timestamp_unit"] = config.timestamp_unit
        payload["temporal_bucket_freq"] = config.temporal_bucket_freq
    return payload


def group_overlap_report(
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    group_cols: list[str],
) -> dict[str, object]:
    group_keys = stable_series_to_group_key(df, group_cols).astype(str)
    split_groups = {name: set(group_keys.iloc[idx].tolist()) for name, idx in splits.items()}
    train_val = split_groups["train"] & split_groups["val"]
    train_test = split_groups["train"] & split_groups["test"]
    val_test = split_groups["val"] & split_groups["test"]
    return {
        "n_train_groups": len(split_groups["train"]),
        "n_val_groups": len(split_groups["val"]),
        "n_test_groups": len(split_groups["test"]),
        "train_val_overlap": len(train_val),
        "train_test_overlap": len(train_test),
        "val_test_overlap": len(val_test),
        "ok": not train_val and not train_test and not val_test,
    }


def temporal_split_report(
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    config: PipelineConfig,
) -> dict[str, object]:
    parsed = parse_timestamp_series(df[config.timestamp_col], unit=config.timestamp_unit)
    if config.temporal_bucket_freq:
        bucket = parsed.dt.floor(config.temporal_bucket_freq).astype("string").fillna("<NaT>")
    else:
        bucket = parsed.astype("string").fillna("<NaT>")

    split_buckets = {name: set(bucket.iloc[idx].tolist()) for name, idx in splits.items()}
    train_val = split_buckets["train"] & split_buckets["val"]
    train_test = split_buckets["train"] & split_buckets["test"]
    val_test = split_buckets["val"] & split_buckets["test"]
    payload: dict[str, object] = {
        "timestamp_col": config.timestamp_col,
        "timestamp_unit": config.timestamp_unit,
        "timestamp_unit_inferred": infer_timestamp_unit(df[config.timestamp_col]) if config.timestamp_unit in {None, "auto"} else config.timestamp_unit,
        "temporal_bucket_freq": config.temporal_bucket_freq,
        "n_train_time_buckets": len(split_buckets["train"]),
        "n_val_time_buckets": len(split_buckets["val"]),
        "n_test_time_buckets": len(split_buckets["test"]),
        "train_val_bucket_overlap": len(train_val),
        "train_test_bucket_overlap": len(train_test),
        "val_test_bucket_overlap": len(val_test),
        "ok": not train_val and not train_test and not val_test,
    }
    for name, idx in splits.items():
        split_times = parsed.iloc[idx]
        payload[f"{name}_timestamp_min"] = None if split_times.empty else str(split_times.min())
        payload[f"{name}_timestamp_max"] = None if split_times.empty else str(split_times.max())
    return payload


def temporal_per_class_report(
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    config: PipelineConfig,
) -> tuple[dict[str, object], pd.DataFrame]:
    target_col = config.type_col if config.type_col in df.columns else config.label_col
    parsed = parse_timestamp_series(df[config.timestamp_col], unit=config.timestamp_unit)
    if config.temporal_bucket_freq:
        bucket = parsed.dt.floor(config.temporal_bucket_freq).astype("string").fillna("<NaT>")
    else:
        bucket = parsed.astype("string").fillna("<NaT>")

    class_values = df[target_col].astype(str)
    class_bucket = class_values + "||" + bucket
    split_sets = {name: set(class_bucket.iloc[idx].tolist()) for name, idx in splits.items()}
    train_val = split_sets["train"] & split_sets["val"]
    train_test = split_sets["train"] & split_sets["test"]
    val_test = split_sets["val"] & split_sets["test"]

    rows = []
    all_classes = sorted(class_values.unique().tolist())
    missing_train: list[str] = []
    missing_val: list[str] = []
    missing_test: list[str] = []
    for class_value in all_classes:
        row: dict[str, object] = {"class_value": class_value}
        for split, idx in splits.items():
            mask_idx = df.index[idx]
            split_class = class_values.iloc[idx] == class_value
            count = int(split_class.sum())
            split_times = parsed.iloc[idx][split_class.to_numpy()]
            row[f"{split}_count"] = count
            row[f"{split}_timestamp_min"] = None if split_times.empty else str(split_times.min())
            row[f"{split}_timestamp_max"] = None if split_times.empty else str(split_times.max())
            if split == "train" and count == 0:
                missing_train.append(class_value)
            if split == "val" and count == 0:
                missing_val.append(class_value)
            if split == "test" and count == 0:
                missing_test.append(class_value)
        rows.append(row)

    per_class = pd.DataFrame(rows)
    payload: dict[str, object] = {
        "target_col": target_col,
        "timestamp_col": config.timestamp_col,
        "timestamp_unit": config.timestamp_unit,
        "timestamp_unit_inferred": infer_timestamp_unit(df[config.timestamp_col]) if config.timestamp_unit in {None, "auto"} else config.timestamp_unit,
        "temporal_bucket_freq": config.temporal_bucket_freq,
        "n_classes": len(all_classes),
        "classes_missing_train": missing_train,
        "classes_missing_val": missing_val,
        "classes_missing_test": missing_test,
        "train_val_class_bucket_overlap": len(train_val),
        "train_test_class_bucket_overlap": len(train_test),
        "val_test_class_bucket_overlap": len(val_test),
        "ok": not missing_train and not missing_test and not train_val and not train_test and not val_test,
    }
    return payload, per_class


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
    if unit and unit != "auto":
        return pd.to_datetime(pd.to_numeric(series, errors="coerce"), unit=unit, errors="coerce")

    inferred = infer_timestamp_unit(series)
    if inferred is not None:
        numeric = pd.to_numeric(series, errors="coerce")
        return pd.to_datetime(numeric, unit=inferred, errors="coerce")
    return pd.to_datetime(series, errors="coerce")


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
