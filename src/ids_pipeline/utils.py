"""Small utilities used across the pipeline."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logging(output_dir: str | Path, level: str = "INFO") -> None:
    ensure_dir(output_dir)
    log_path = Path(output_dir) / "pipeline.log"
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(numeric_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    stream.setLevel(numeric_level)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(numeric_level)
    root.addHandler(stream)
    root.addHandler(file_handler)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(payload), fh, indent=2, sort_keys=True)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def save_table(df: pd.DataFrame, path_without_suffix: str | Path) -> Path:
    """Save a table as Parquet when possible and CSV as a guaranteed fallback."""

    base = Path(path_without_suffix)
    ensure_dir(base.parent)
    parquet_path = base.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception as exc:  # pragma: no cover - depends on optional engines
        logging.getLogger(__name__).warning(
            "Could not save %s as Parquet (%s). Falling back to CSV.",
            parquet_path,
            exc,
        )
        csv_path = base.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


def stable_series_to_group_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series(np.arange(len(df)), index=df.index, name="row_group")
    values = df[existing].astype("string").fillna("<NA>")
    key = values[existing[0]]
    for col in existing[1:]:
        key = key + "||" + values[col]
    return key.rename("group_key")


def dataframe_from_matrix(matrix: Any, prefix: str = "z") -> pd.DataFrame:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    arr = np.asarray(matrix)
    return pd.DataFrame(arr, columns=[f"{prefix}{i}" for i in range(arr.shape[1])])
