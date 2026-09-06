"""Targeted read-only forensics. Writes evidence only below this report directory."""
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import ipaddress
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "src"))
from ids_pipeline.dataset_schema import SchemaValidationError, normalize_dataset_schema
from ids_pipeline.schema import TON_IOT_SCHEMA, TON_IOT_SCHEMA_VERSION


def signature(path):
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


class TrackedLines:
    def __init__(self, handle):
        self.handle, self.lines = handle, []
    def __iter__(self):
        return self
    def __next__(self):
        line = next(self.handle)
        self.lines.append(line)
        return line


def extract_csv(path, expected):
    before = signature(path)
    selected = []
    zero_ip_counts = Counter()
    zero_ip_examples = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        track = TrackedLines(handle)
        reader = csv.reader(track, dialect="excel", strict=True)
        header = next(reader)
        bytes_idx = header.index("src_bytes")
        src_idx, dst_idx = header.index("src_ip"), header.index("dst_ip")
        offset = 0
        while True:
            track.lines = []
            start = reader.line_num + 1
            try:
                fields = next(reader)
            except StopIteration:
                break
            if len(fields) > max(src_idx, dst_idx):
                if fields[src_idx] == "0.0.0.0" or fields[dst_idx] == "0.0.0.0":
                    zero_ip_counts["src_ip"] += fields[src_idx] == "0.0.0.0"
                    zero_ip_counts["dst_ip"] += fields[dst_idx] == "0.0.0.0"
                    if len(zero_ip_examples) < 5:
                        row = dict(zip(header, fields))
                        zero_ip_examples.append({"row_0based": offset, **{k: row.get(k) for k in
                            ["ts", "src_ip", "dst_ip", "src_port", "dst_port", "proto", "service", "src_bytes", "conn_state"]}})
            if len(fields) > bytes_idx and fields[bytes_idx] == "0.0.0.0":
                raw = "".join(track.lines)
                row = dict(zip(header, fields))
                selected.append({"partition": int(path.stem.split("_")[-1]), "row_0based": offset,
                    "physical_line_start": start, "physical_line_end": reader.line_num,
                    "field_count": len(fields), "header_count": len(header),
                    "comma_count": raw.count(","), "quote_count": raw.count('"'),
                    "raw_csv": raw, "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                    "fields": row, "strict_csv_roundtrip": next(csv.reader(io.StringIO(raw), strict=True)) == fields})
            offset += 1
    assert len(selected) == expected, (path, len(selected), expected)
    assert signature(path) == before
    return selected, {"path": str(path), "signature": before, "source_unchanged": True,
        "records_scanned_to_locate_targets": offset, "header": header,
        "zero_ip_counts_in_scanned_partition": dict(zero_ip_counts), "zero_ip_examples": zero_ip_examples}


def parquet_rows_at(path, positions):
    before = signature(path)
    parquet = pq.ParquetFile(path)
    found = {}
    offset = 0
    wanted = sorted(positions)
    groups_read = []
    for group in range(parquet.num_row_groups):
        count = parquet.metadata.row_group(group).num_rows
        targets = [p for p in wanted if offset <= p < offset + count]
        if targets:
            groups_read.append(group)
            batch_offset = offset
            for batch in parquet.iter_batches(batch_size=50000, row_groups=[group], use_threads=False):
                local = [p for p in targets if batch_offset <= p < batch_offset + batch.num_rows]
                if local:
                    subset = batch.take([p - batch_offset for p in local]).to_pylist()
                    found.update(dict(zip(local, subset)))
                batch_offset += batch.num_rows
        offset += count
    assert signature(path) == before
    return found, {"path": str(path), "signature": before, "source_unchanged": True,
                   "row_count": offset, "row_groups_read": groups_read, "matched_positions": len(found)}


def compare_rows(raw_rows, parquet_rows):
    columns = list(raw_rows[0]["fields"])
    csv_df = pd.DataFrame([r["fields"] for r in raw_rows])
    pq_df = pd.DataFrame([parquet_rows[r["row_0based"]] for r in raw_rows])
    common = [c for c in columns if c in pq_df and c != "src_bytes"]
    csv_norm, _ = normalize_dataset_schema(csv_df[common])
    pq_norm, _ = normalize_dataset_schema(pq_df[common])
    mismatches = []
    lexical_differences = Counter()
    unequal = {}
    for col in common:
        a, b = csv_norm[col], pq_norm[col]
        unequal[col] = ~(a.eq(b).fillna(False) | (a.isna() & b.isna()))
        lexical_differences[col] = int(csv_df[col].astype(str).ne(pq_df[col].astype(str)).sum())
    unequal["src_bytes"] = pq_df.src_bytes.astype(str).ne("0.0.0.0")
    mask = pd.DataFrame(unequal)
    for index in np.flatnonzero(mask.any(axis=1)):
        mismatches.append({"row_0based": raw_rows[index]["row_0based"],
                           "columns": mask.columns[mask.iloc[index]].tolist(),
                           "parquet_src_bytes": str(pq_df.src_bytes.iloc[index])})
    return {"rows_compared": len(raw_rows), "common_columns": common + ["src_bytes"],
            "semantic_mismatches": mismatches, "lexical_only_differences_by_column": dict(lexical_differences),
            "extra_parquet_columns": [c for c in pq_df if c not in columns],
            "missing_parquet_columns": [c for c in columns if c not in pq_df]}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    all_records, sources, comparisons = [], [], []
    evidence_path = ROOT / "records_evidence.jsonl"
    cached = [json.loads(line) for line in evidence_path.read_text(encoding="utf-8").splitlines()] if evidence_path.exists() else []
    reinspection_path = ROOT / "csv_reinspection.json"
    verified_sources = json.loads(reinspection_path.read_text(encoding="utf-8")) if reinspection_path.exists() else []
    if cached and not all(any(s["path"] == str(Path(f"D:/IC/Dataset/Network_dataset_{p}.csv"))
                             and s["signature"] == signature(Path(s["path"])) for s in verified_sources)
                          for p in [1, 22, 23]):
        cached = []
    for partition, expected in [(1, 175), (22, 690), (23, 4)]:
        csv_path = Path(f"D:/IC/Dataset/Network_dataset_{partition}.csv")
        if cached:
            records = [r for r in cached if r["partition"] == partition]
            assert len(records) == expected
            source = {"path": str(csv_path), "signature": signature(csv_path),
                      "header": list(records[0]["fields"]), "evidence_reused": True}
        else:
            records, source = extract_csv(csv_path, expected)
        all_records.extend(records)
        sources.append(source)
        for directory in [Path("D:/IC/Dataset/dados/dataset_parquet_total"),
                          Path("D:/IC/Dataset/dataset_parquet"), Path("D:/IC/Dataset")]:
            path = directory / f"Network_dataset_{partition}.parquet"
            if not path.is_file():
                continue
            matches, info = parquet_rows_at(path, [r["row_0based"] for r in records])
            sources.append(info)
            comparisons.append({"partition": partition, "source": str(path), **compare_rows(records, matches)})
        print(f"Partition {partition}: inspected {len(records)} complete CSV records", flush=True)
    (ROOT / "sources_and_comparisons.json").write_text(json.dumps({"sources": sources, "comparisons": comparisons}, indent=2), encoding="utf-8")
    with (ROOT / "records_evidence.jsonl").open("w", encoding="utf-8") as handle:
        for record in all_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    df = pd.DataFrame([{**r["fields"], "partition": r["partition"], "row_0based": r["row_0based"]} for r in all_records])
    try:
        _, report = normalize_dataset_schema(df[list(TON_IOT_SCHEMA.keys() & df.columns)])
    except SchemaValidationError as exc:
        report = exc.report
    def valid_ip(value):
        try:
            ipaddress.ip_address(value)
            return True
        except ValueError:
            return False
    valid_ips = {col: int(df[col].map(valid_ip).sum()) for col in ["src_ip", "dst_ip"]}
    times = pd.to_datetime(pd.to_numeric(df.ts), unit="s", utc=True)
    df["date_utc"] = times.dt.strftime("%Y-%m-%d")
    summaries = {}
    for col in ["partition", "label", "type", "src_ip", "dst_ip", "src_port", "dst_port", "proto", "service", "conn_state", "date_utc"]:
        summaries[col] = [{"value": str(v), "count": int(n)} for v, n in df[col].value_counts(dropna=False).items()]
    pattern_cols = ["partition", "label", "type", "src_ip", "dst_ip", "proto", "service", "conn_state"]
    grouped = df.groupby(pattern_cols, dropna=False, sort=False)
    patterns = []
    for key, group in grouped:
        patterns.append({**dict(zip(pattern_cols, map(str, key))), "count": len(group),
                         "min_ts": int(pd.to_numeric(group.ts).min()), "max_ts": int(pd.to_numeric(group.ts).max()),
                         "example_row_0based": int(group.row_0based.iloc[0])})
    flow_cols = ["src_ip", "src_port", "dst_ip", "dst_port", "proto"]
    flow_groups = df.groupby(flow_cols, dropna=False).size().sort_values(ascending=False)
    output = {"schema_version": TON_IOT_SCHEMA_VERSION, "target_rows": len(df), "sources": sources,
        "parquet_comparisons": comparisons, "schema_report": report, "valid_ip_counts": valid_ips,
        "label_type_consistent": int(((df.label == "0") == (df.type == "normal")).sum()),
        "structural_signatures": [{"signature": str(key), "count": count} for key, count in Counter(
             (r["field_count"], r["header_count"], r["comma_count"], r["quote_count"],
              r["physical_line_end"]-r["physical_line_start"]+1, r["strict_csv_roundtrip"]) for r in all_records).items()],
        "summaries": summaries, "patterns": patterns,
        "time_range_utc": [str(times.min()), str(times.max())],
        "unique_5tuples": len(flow_groups), "unique_ts_5tuples": len(df.drop_duplicates(["ts"] + flow_cols)),
        "top_5tuples": [{"key": list(key), "count": int(count)} for key, count in flow_groups.head(10).items()],
        "neighbor_values": {col: [{"value": str(v), "count": int(n)} for v, n in df[col].value_counts().head(12).items()]
                            for col in ["duration", "src_bytes", "dst_bytes", "conn_state", "src_pkts", "dst_pkts", "src_ip_bytes", "dst_ip_bytes"]}}
    (ROOT / "findings.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(ROOT / "target_rows.csv", index=False)
    pd.DataFrame(patterns).to_csv(ROOT / "patterns.csv", index=False)
    print(json.dumps({k:output[k] for k in ["target_rows", "valid_ip_counts", "structural_signatures", "summaries", "time_range_utc", "unique_5tuples", "neighbor_values"]}, indent=2))


if __name__ == "__main__":
    main()
