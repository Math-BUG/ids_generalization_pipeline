"""Read-only, bounded-memory audit of TON_IoT partitions; never trains or converts.

Production normalization is reused unchanged. Its exception contains a complete
per-chunk report, so violations do not prevent auditing subsequent chunks/files.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ids_pipeline.dataset_schema import SchemaValidationError, normalize_dataset_schema
from ids_pipeline.schema import TON_IOT_SCHEMA, TON_IOT_SCHEMA_VERSION


COUNT_FIELDS = ("observed_rows", "native_nulls", "registered_sentinels", "recognized_missing",
                "invalid_cells", "negative_nonnegative_domain", "nonintegral",
                "ports_out_of_range", "invalid_label", "unexpected_boolean")


def empty_column():
    return {**dict.fromkeys(COUNT_FIELDS, 0), "physical_dtypes": {}, "pandas_dtypes": {},
            "sentinel_tokens": {}, "issues": {}}


def merge_column(target, source):
    for field in COUNT_FIELDS:
        target[field] += source[field]
    for field in ("physical_dtypes", "pandas_dtypes", "sentinel_tokens"):
        counter = Counter(target[field])
        counter.update(source[field])
        target[field] = dict(counter)
    for kind, issue in source["issues"].items():
        dest = target["issues"].setdefault(kind, {"count": 0, "examples": []})
        dest["count"] += issue["count"]
        for example in issue["examples"]:
            # Prefer different tokens in the bounded global examples.
            if len(dest["examples"]) < 5 and not any(e["value"] == example["value"] for e in dest["examples"]):
                dest["examples"].append(example)


def audit_chunk(frame, physical_types, filename, offset):
    try:
        normalized, report = normalize_dataset_schema(frame)
        del normalized
    except SchemaValidationError as exc:
        report = exc.report
    columns = {}
    for name in frame.columns:
        raw = frame[name]
        spec = TON_IOT_SCHEMA.get(name)
        item = empty_column()
        item["observed_rows"] = len(frame)
        item["physical_dtypes"] = {physical_types[name]: len(frame)}
        item["pandas_dtypes"] = {str(raw.dtype): len(frame)}
        item["native_nulls"] = int(raw.isna().sum())
        if spec is not None:
            info = report["columns"][name]
            item["registered_sentinels"] = info["sentinel_missing_count"]
            item["recognized_missing"] = info["native_missing_count"] + info["sentinel_missing_count"]
            item["sentinel_tokens"] = {token: int(raw.eq(token).fillna(False).sum()) for token in spec.missing_tokens}
            item["invalid_cells"] = info["invalid_count"]
            for kind, issue in info["issues"].items():
                item["issues"][kind] = {
                    "count": issue["count"],
                    "examples": [{"file": filename, "row_0based": offset + e["position"],
                                  "value": e["value"]} for e in issue["examples"]],
                }
            if spec.storage_dtype in {"float64", "Int64"} or spec.semantic_type == "nominal_code":
                values = pd.to_numeric(raw.mask(raw.isna() | raw.isin(spec.missing_tokens)), errors="coerce").astype("float64")
                finite = values.notna() & np.isfinite(values)
                # Audit-only breakdown of the production domain_violation count.
                if spec.minimum is not None and spec.minimum >= 0:
                    item["negative_nonnegative_domain"] = int((finite & (values < 0)).sum())
                if spec.integral:
                    item["nonintegral"] = int((finite & (values % 1 != 0)).sum())
                if spec.semantic_type == "port":
                    item["ports_out_of_range"] = int((finite & ((values < 0) | (values > 65535))).sum())
            if name == "label":
                item["invalid_label"] = info["invalid_count"]
            if spec.semantic_type == "boolean":
                item["unexpected_boolean"] = info["invalid_count"]
        columns[name] = item
    return columns, report["ok"]


def file_signature(path):
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def audit_file(path, chunk_size):
    before = file_signature(path)
    if path.suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        names = parquet.schema_arrow.names
        physical_types = {field.name: str(field.type) for field in parquet.schema_arrow}
        expected_rows = parquet.metadata.num_rows
        chunks = (batch.to_pandas() for batch in parquet.iter_batches(batch_size=chunk_size, use_threads=False))
    else:
        with path.open("r", encoding="utf-8", newline="") as handle:
            names = next(csv.reader(handle))
        physical_types = {name: "CSV text (no intrinsic dtype)" for name in names}
        expected_rows = None
        chunks = pd.read_csv(path, dtype="string", keep_default_na=False, na_filter=False,
                             chunksize=chunk_size, low_memory=False, skip_blank_lines=False)
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate header columns in {path}")
    absent = [name for name in TON_IOT_SCHEMA if name not in names]
    result = {"file": path.name, "path": str(path.resolve()), "source_before": before,
              "columns_present": names, "columns_absent": absent,
              "columns_additional": [name for name in names if name not in TON_IOT_SCHEMA],
              "missing_required_columns": [name for name in absent if not TON_IOT_SCHEMA[name].nullable],
              "rows": 0, "chunks": 0, "expected_rows_from_metadata": expected_rows,
              "columns": {name: empty_column() for name in names}, "values_compatible": True}
    for frame in chunks:
        stats, ok = audit_chunk(frame, physical_types, path.name, result["rows"])
        for name, values in stats.items():
            merge_column(result["columns"][name], values)
        result["rows"] += len(frame)
        result["chunks"] += 1
        result["values_compatible"] &= ok
        if result["chunks"] % 5 == 0:
            print(f"PROGRESS {path.name}: {result['rows']:,} rows", flush=True)
    result["source_after"] = file_signature(path)
    result["source_unchanged_size_mtime"] = before == result["source_after"]
    result["row_count_matches_metadata"] = expected_rows is None or result["rows"] == expected_rows
    result["compatible"] = (result["values_compatible"] and not result["columns_additional"]
                            and not result["missing_required_columns"]
                            and result["source_unchanged_size_mtime"] and result["row_count_matches_metadata"])
    return result


def flatten_column(name, item, filename="ALL"):
    return {"file": filename, "column": name, **{field: item[field] for field in COUNT_FIELDS},
            "conversion_failure": item["issues"].get("conversion_failure", {}).get("count", 0),
            "infinite": item["issues"].get("infinite", {}).get("count", 0),
            "domain_violation": item["issues"].get("domain_violation", {}).get("count", 0),
            "unsafe_integer_precision": item["issues"].get("unsafe_integer_precision", {}).get("count", 0),
            "missing_required": item["issues"].get("missing_required", {}).get("count", 0),
            "physical_dtypes": json.dumps(item["physical_dtypes"]),
            "pandas_dtypes": json.dumps(item["pandas_dtypes"]),
            "sentinel_tokens": json.dumps(item["sentinel_tokens"])}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_audit(input_dir, output_dir, file_format, chunk_size=100000, expected_files=23):
    source = Path(input_dir).resolve()
    output = Path(output_dir).resolve()
    if output == source or source in output.parents:
        raise ValueError("Audit artifacts must be outside the source directory")
    pattern = re.compile(r"Network_dataset_(\d+)\." + re.escape(file_format) + r"$")
    files = [p for p in source.iterdir() if p.is_file() and pattern.fullmatch(p.name)]
    files.sort(key=lambda p: int(pattern.fullmatch(p.name).group(1)))
    if len(files) != expected_files or {int(pattern.fullmatch(p.name).group(1)) for p in files} != set(range(1, expected_files + 1)):
        raise ValueError(f"Expected partitions 1..{expected_files}, found {[p.name for p in files]}")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    output.mkdir(parents=True, exist_ok=False)
    code_dir = Path(__file__).resolve().parents[1]
    result = {"schema_version": TON_IOT_SCHEMA_VERSION, "input_dir": str(source), "format": file_format,
              "started_utc": datetime.now(timezone.utc).isoformat(), "chunk_size": chunk_size,
              "code_sha256": {str(p.relative_to(code_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [Path(__file__).resolve(), code_dir / "src/ids_pipeline/schema.py",
                                        code_dir / "src/ids_pipeline/dataset_schema.py"]},
              "files": [], "columns": {}, "rows": 0, "complete": False}
    for path in files:
        print(f"START {path.name}", flush=True)
        item = audit_file(path, chunk_size)
        (output / (path.stem + ".json")).write_text(json.dumps(item, indent=2), encoding="utf-8")
        result["files"].append(item)
        result["rows"] += item["rows"]
        for name, values in item["columns"].items():
            merge_column(result["columns"].setdefault(name, empty_column()), values)
        print(f"DONE {path.name}: {item['rows']:,} rows; invalid_cells={sum(v['invalid_cells'] for v in item['columns'].values())}", flush=True)
    result["complete"] = True
    result["file_count"] = len(result["files"])
    result["compatible"] = all(item["compatible"] for item in result["files"])
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    for name, item in result["columns"].items():
        item["absent_rows"] = result["rows"] - item["observed_rows"]
    (output / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(output / "by_column.csv", [dict(flatten_column(name, item), absent_rows=item["absent_rows"])
                                         for name, item in result["columns"].items()])
    write_csv(output / "by_file_column.csv", [flatten_column(name, stats, item["file"])
                                              for item in result["files"] for name, stats in item["columns"].items()])
    write_csv(output / "by_file.csv", [{"file": item["file"], "rows": item["rows"], "chunks": item["chunks"],
        "n_columns": len(item["columns_present"]), "absent": json.dumps(item["columns_absent"]),
        "additional": json.dumps(item["columns_additional"]), "compatible": item["compatible"],
        "invalid_cells": sum(v["invalid_cells"] for v in item["columns"].values()),
        "source_unchanged_size_mtime": item["source_unchanged_size_mtime"]} for item in result["files"]])
    print(f"COMPLETE files={len(files)} rows={result['rows']} compatible={result['compatible']}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--format", choices=["csv", "parquet"], required=True)
    parser.add_argument("--chunk-size", type=int, default=100000)
    args = parser.parse_args()
    run_audit(args.input_dir, args.output_dir, args.format, args.chunk_size)
