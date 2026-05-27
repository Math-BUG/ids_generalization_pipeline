#!/usr/bin/env python3
"""Generate label/type distribution tables for an existing experiment split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ids_pipeline.data_loading import load_dataset
from ids_pipeline.splitting import target_distribution_pivot, target_distribution_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Describe target distributions across saved splits.")
    parser.add_argument("--data-path", required=True, help="CSV/Parquet dataset path used by the experiment.")
    parser.add_argument("--output-dir", required=True, help="Experiment output directory containing splits/*.csv.")
    parser.add_argument("--label-col", default=None, help="Override binary label column.")
    parser.add_argument("--type-col", default=None, help="Override multiclass type column.")
    parser.add_argument("--compute-backend", default="cpu", choices=["cpu", "gpu", "auto"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    split_dir = output_dir / "splits"
    config = read_optional_json(output_dir / "config_resolved.json")
    label_col = args.label_col or config.get("label_col", "label")
    type_col = args.type_col or config.get("type_col", "type")

    df = load_dataset(args.data_path, compute_backend=args.compute_backend)
    splits = read_splits(split_dir)
    target_cols = [col for col in [label_col, type_col] if col in df.columns]
    if not target_cols:
        raise ValueError(f"No target columns found. Tried {label_col!r} and {type_col!r}.")

    distribution = target_distribution_table(df, splits, target_cols)
    distribution.to_csv(split_dir / "target_distribution.csv", index=False)
    print(f"Wrote {split_dir / 'target_distribution.csv'}")
    for target_col in target_cols:
        pivot = target_distribution_pivot(distribution, target_col)
        path = split_dir / f"target_distribution_{target_col}.csv"
        pivot.to_csv(path, index=False)
        print(f"Wrote {path}")


def read_splits(split_dir: Path) -> dict[str, pd.Index]:
    splits = {}
    for split in ["train", "val", "test"]:
        path = split_dir / f"{split}_indices.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        splits[split] = pd.read_csv(path)["row_index"].to_numpy(dtype=int)
    return splits


def read_optional_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


if __name__ == "__main__":
    main()
