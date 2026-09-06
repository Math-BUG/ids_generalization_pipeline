"""Read-only, chunked audit of the eligible TON_IoT population after quarantine."""
from __future__ import annotations

import argparse
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
from ids_pipeline.data_quality_policy import (
    DataQualityPopulation, assert_source_unchanged, known_corruption_mask, source_fingerprint,
)
from ids_pipeline.dataset_schema import SchemaValidationError, is_ton_iot, normalize_dataset_schema
from ids_pipeline.schema import TON_IOT_SCHEMA, TON_IOT_SCHEMA_VERSION


def run_audit(input_dir, output_dir, file_format="parquet", chunk_size=100000, expected_files=23):
    source, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    if source == output or source in output.parents:
        raise ValueError("Audit output must be outside the source directory")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    pattern = re.compile(r"Network_dataset_(\d+)\." + re.escape(file_format) + r"$")
    # Same lexicographical file order as the production loader; no split changes.
    files = sorted(p for p in source.iterdir() if p.is_file() and pattern.fullmatch(p.name))
    if len(files) != expected_files or {int(pattern.fullmatch(p.name).group(1)) for p in files} != set(range(1, expected_files+1)):
        raise ValueError(f"Expected exactly partitions 1..{expected_files}")
    output.mkdir(parents=True, exist_ok=False)
    population = DataQualityPopulation()
    repo = Path(__file__).resolve().parents[1]
    result = {"schema_version": TON_IOT_SCHEMA_VERSION, "started_utc": datetime.now(timezone.utc).isoformat(),
              "format": file_format, "chunk_size": chunk_size, "complete": False, "files": [],
              "invalid_cells_after_quarantine": 0, "columns": {},
              "code_sha256": {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [Path(__file__), repo/'src/ids_pipeline/data_quality_policy.py', repo/'src/ids_pipeline/dataset_schema.py', repo/'src/ids_pipeline/schema.py']}}
    for path in files:
        fingerprint = source_fingerprint(path)
        source_id = path.relative_to(source).as_posix()
        population.register_file(source_id, fingerprint, source_path=str(path))
        if file_format == "parquet":
            parquet = pq.ParquetFile(path)
            columns = parquet.schema_arrow.names
            expected_rows = parquet.metadata.num_rows
            chunks = (batch.to_pandas() for batch in parquet.iter_batches(batch_size=chunk_size, use_threads=False))
        else:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                columns = next(csv.reader(handle))
            expected_rows = None
            chunks = pd.read_csv(path, dtype="string", keep_default_na=False, na_filter=False,
                                 chunksize=chunk_size, low_memory=False, skip_blank_lines=False)
        if len(set(columns)) != len(columns) or not is_ton_iot(columns):
            raise ValueError(f"Not a recognized, unique TON_IoT header: {path}")
        item = {"source_id": source_id, "columns": columns, "original_rows": 0, "eligible_rows": 0,
                "quarantined_rows": 0, "invalid_cells_after_quarantine": 0, "issues": {},
                "missing_required_columns": [c for c,spec in TON_IOT_SCHEMA.items() if not spec.nullable and c not in columns],
                "additional_columns": [c for c in columns if c not in TON_IOT_SCHEMA]}
        print(f"START {path.name}", flush=True)
        for raw in chunks:
            offset = item["original_rows"]
            kept = population.apply(raw, source_id=source_id, row_offset=offset)
            accepted_positions = np.flatnonzero(~known_corruption_mask(raw).to_numpy(dtype=bool)) + offset
            try:
                normalized, schema_report = normalize_dataset_schema(kept)
                del normalized
            except SchemaValidationError as exc:
                schema_report = exc.report
            item["original_rows"] += len(raw)
            item["eligible_rows"] += len(kept)
            item["quarantined_rows"] += len(raw) - len(kept)
            for col, info in schema_report["columns"].items():
                target = result["columns"].setdefault(col, {"rows": 0, "invalid_cells": 0, "issues": {}})
                target["rows"] += len(kept)
                target["invalid_cells"] += info["invalid_count"]
                item["invalid_cells_after_quarantine"] += info["invalid_count"]
                for kind, issue in info["issues"].items():
                    target["issues"][kind] = target["issues"].get(kind, 0) + issue["count"]
                    if issue["count"]:
                        dest = item["issues"].setdefault(col, {}).setdefault(kind, {"count":0, "examples":[]})
                        dest["count"] += issue["count"]
                        for e in issue["examples"]:
                            if len(dest["examples"]) < 5:
                                dest["examples"].append({"row_0based": int(accepted_positions[e["position"]]), "value":e["value"]})
            print(f"PROGRESS {path.name}: original={item['original_rows']} eligible={item['eligible_rows']} invalid={item['invalid_cells_after_quarantine']}", flush=True)
        assert_source_unchanged(path, fingerprint)
        item["source_unchanged_size_mtime"] = True
        item["row_count_matches_metadata"] = expected_rows is None or item["original_rows"] == expected_rows
        item["compatible"] = (item["invalid_cells_after_quarantine"] == 0 and item["row_count_matches_metadata"]
                              and not item["missing_required_columns"] and not item["additional_columns"])
        result["files"].append(item)
        result["invalid_cells_after_quarantine"] += item["invalid_cells_after_quarantine"]
        (output/(path.stem+'.json')).write_text(json.dumps(item,indent=2),encoding="utf-8")
        population.save(output/'data_quality_report.json')
    quality = population.save(output/'data_quality_report.json')
    result.update(complete=True, compatible=all(f["compatible"] for f in result["files"]), file_count=len(files),
                  original_rows=quality["original_rows"], quarantined_rows=quality["quarantined_rows"],
                  eligible_rows=quality["eligible_rows"], population_id=quality["population_id"],
                  finished_utc=datetime.now(timezone.utc).isoformat())
    (output/'audit.json').write_text(json.dumps(result,indent=2),encoding="utf-8")
    with (output/'by_file.csv').open('w',encoding='utf-8',newline='') as handle:
        fields=['source_id','original_rows','quarantined_rows','eligible_rows','invalid_cells_after_quarantine','compatible']
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore')
        writer.writeheader(); writer.writerows(result['files'])
    print(json.dumps({k:v for k,v in result.items() if k not in {'files','columns','code_sha256'}},indent=2),flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--format', choices=['parquet','csv'], default='parquet')
    parser.add_argument('--chunk-size', type=int, default=100000)
    args = parser.parse_args()
    result = run_audit(args.input_dir, args.output_dir, args.format, args.chunk_size)
    sys.exit(0 if result['compatible'] else 1)
