"""Simple reproducible train/validation/test splitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import PipelineConfig
from .utils import ensure_dir, write_json


def create_splits(df: pd.DataFrame, config: PipelineConfig) -> dict[str, np.ndarray]:
    if config.split_strategy != "random_stratified":
        raise ValueError("Minimal version supports only split_strategy=random_stratified.")

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
    return {"train": np.sort(train_idx), "val": np.sort(val_idx), "test": np.sort(test_idx)}


def _safe_stratify(df: pd.DataFrame, label_col: str) -> pd.Series | None:
    if label_col not in df.columns:
        return None
    y = df[label_col]
    counts = y.value_counts(dropna=False)
    return y if len(counts) > 1 and counts.min() >= 2 else None


def save_splits(splits: dict[str, np.ndarray], output_dir: str | Path) -> None:
    split_dir = ensure_dir(Path(output_dir) / "splits")
    for name, idx in splits.items():
        pd.DataFrame({"row_index": idx}).to_csv(split_dir / f"{name}_indices.csv", index=False)
    write_json(Path(output_dir) / "split_summary.json", split_summary(splits))


def split_summary(splits: dict[str, np.ndarray]) -> dict[str, int]:
    return {name: int(len(idx)) for name, idx in splits.items()}
