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
    raise ValueError(
        "split_strategy must be one of: random_stratified, group_stratified, temporal"
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

    parsed = pd.to_datetime(df[config.timestamp_col], errors="coerce")
    ordered = (
        pd.DataFrame({"row_index": np.arange(len(df)), "timestamp": parsed})
        .sort_values(["timestamp", "row_index"], na_position="last")
        ["row_index"]
        .to_numpy()
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


def split_summary(
    splits: dict[str, np.ndarray],
    config: PipelineConfig | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {name: int(len(idx)) for name, idx in splits.items()}
    if config is not None:
        payload["split_strategy"] = config.split_strategy
        payload["test_size"] = config.test_size
        payload["val_size"] = config.val_size
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
