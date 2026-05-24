#!/usr/bin/env python3
"""Convert a directory of CSV files to one Parquet file per CSV.

The converter does a conservative schema pass before writing. This matters for
RAPIDS/cuDF: multi-file Parquet reads require every partition to expose the same
column names and compatible Arrow types.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from pandas.api import types as ptypes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CSV partitions to Parquet partitions.")
    parser.add_argument("--input-dir", required=True, help="Directory containing CSV files.")
    parser.add_argument("--output-dir", required=True, help="Directory where Parquet files will be written.")
    parser.add_argument("--glob", default="*.csv", help="CSV glob relative to input-dir. Default: *.csv")
    parser.add_argument("--compression", default="zstd", help="Parquet compression. Default: zstd")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing Parquet files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.rglob(args.glob))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir} with glob={args.glob!r}")

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "compression": args.compression,
        "files": [],
    }

    print(f"INFERRING common schema from {len(csv_files)} CSV file(s)")
    schema, columns = infer_common_schema(csv_files)
    manifest["schema"] = schema
    manifest["columns"] = columns

    for csv_path in csv_files:
        rel = csv_path.relative_to(input_dir)
        parquet_path = (output_dir / rel).with_suffix(".parquet")
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        if parquet_path.exists() and not args.overwrite:
            print(f"SKIP existing {parquet_path}")
            continue

        print(f"READ {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False)
        df = normalize_to_common_schema(df, schema, columns)
        print(f"WRITE {parquet_path} shape={df.shape}")
        df.to_parquet(parquet_path, index=False, compression=args.compression, engine="pyarrow")
        manifest["files"].append(
            {
                "csv": str(csv_path),
                "parquet": str(parquet_path),
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
            }
        )

    manifest["total_rows"] = int(sum(item["rows"] for item in manifest["files"]))
    manifest["n_files"] = int(len(manifest["files"]))
    manifest_path = output_dir / "_conversion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"WROTE manifest {manifest_path}")


def infer_common_schema(csv_files: list[Path]) -> tuple[dict[str, str], list[str]]:
    """Infer one stable schema for all CSV partitions.

    The policy is intentionally simple:
    - any object/category/string column becomes pandas string;
    - numeric columns stay numeric, with int partitions promoted to float if
      another partition has the same column as float;
    - missing columns are added later as nullable values.
    """

    families_by_col: dict[str, set[str]] = {}
    columns: list[str] = []
    seen: set[str] = set()

    for csv_path in csv_files:
        print(f"SCHEMA {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False)
        for col in df.columns:
            if col not in seen:
                columns.append(col)
                seen.add(col)
            families_by_col.setdefault(col, set()).add(dtype_family(df[col]))

    schema = {col: merge_families(families_by_col.get(col, {"string"})) for col in columns}
    return schema, columns


def dtype_family(series: pd.Series) -> str:
    dtype = series.dtype
    if ptypes.is_bool_dtype(dtype) or ptypes.is_integer_dtype(dtype):
        return "int64"
    if ptypes.is_float_dtype(dtype) or ptypes.is_numeric_dtype(dtype):
        return "float64"
    return "string"


def merge_families(families: set[str]) -> str:
    if "string" in families:
        return "string"
    if "float64" in families:
        return "float64"
    if "int64" in families:
        return "Int64"
    return "string"


def normalize_to_common_schema(
    df: pd.DataFrame,
    schema: dict[str, str],
    columns: list[str],
) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA

    out = out[columns]
    for col, target_dtype in schema.items():
        if target_dtype == "string":
            out[col] = out[col].astype("string")
        elif target_dtype == "float64":
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
        elif target_dtype == "Int64":
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
        else:
            raise ValueError(f"Unsupported target dtype {target_dtype!r} for column {col!r}")
    return out


if __name__ == "__main__":
    main()
