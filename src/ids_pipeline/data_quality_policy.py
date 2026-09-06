"""Explicit population eligibility, separate from strict semantic normalization.

No repair, imputation, deduplication, target-dependent selection or schema changes.
The only authorized exclusion is the exact known nonnumeric src_bytes token.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import TON_IOT_SCHEMA_VERSION

DATA_QUALITY_POLICY_VERSION = "ton_iot_known_corruption/1.0.0"
RULE_ID = "quarantine_src_bytes_ipv4_literal"
KNOWN_INVALID_SRC_BYTES = "0.0.0.0"


def known_corruption_mask(frame: pd.DataFrame) -> pd.Series:
    """Exact field/token equality only. All other invalid values reach the schema."""
    if not frame.columns.is_unique:
        raise ValueError("Data quality policy requires unique column names")
    if "src_bytes" not in frame:
        raise ValueError("TON_IoT data quality policy requires src_bytes")
    return frame["src_bytes"].eq(KNOWN_INVALID_SRC_BYTES).fillna(False).astype(bool)


def source_fingerprint(path: str | Path) -> dict:
    """Streaming content hash; never writes or loads an entire source in memory."""
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Source changed while hashing: {path}")
    return {"bytes": before.st_size, "mtime_ns": before.st_mtime_ns, "sha256": digest.hexdigest()}


def assert_source_unchanged(path: str | Path, fingerprint: dict) -> None:
    stat = Path(path).stat()
    if (stat.st_size, stat.st_mtime_ns) != (fingerprint["bytes"], fingerprint["mtime_ns"]):
        raise RuntimeError(f"Source changed while reading: {path}")


class DataQualityPopulation:
    """Accumulate a compact, chunk-independent population manifest.

    Each file is registered once, in input order; chunks must cover contiguous
    original logical positions. Positions do not use (possibly duplicated) pandas
    index labels. Accepted identities are the complement of quarantined positions.
    """
    def __init__(self):
        self.files: list[dict] = []
        self._by_id: dict[str, dict] = {}

    def register_file(self, source_id: str, fingerprint: dict, *, source_path: str = "") -> None:
        if source_id in self._by_id:
            raise ValueError(f"Duplicate source identity: {source_id}")
        item = {"source_id": source_id, "source_path": source_path, "source_fingerprint": fingerprint,
                "original_rows": 0, "quarantined_rows": 0, "eligible_rows": 0,
                "quarantined_row_positions_0based": [], "eligible_target_distributions": {"label": {}, "type": {}}}
        self.files.append(item)
        self._by_id[source_id] = item

    def apply(self, frame: pd.DataFrame, *, source_id: str, row_offset: int) -> pd.DataFrame:
        item = self._by_id[source_id]
        if row_offset != item["original_rows"]:
            raise ValueError("Chunks must be contiguous, without gaps, repeats or overlaps")
        mask = known_corruption_mask(frame).to_numpy(dtype=bool)
        positions = (np.flatnonzero(mask) + row_offset).tolist()
        kept = frame.iloc[np.flatnonzero(~mask)].copy()
        item["original_rows"] += len(frame)
        item["quarantined_rows"] += len(positions)
        item["eligible_rows"] += len(kept)
        item["quarantined_row_positions_0based"].extend(positions)
        # Targets are read strictly AFTER eligibility, for descriptive auditing.
        # Values are reported as observed text, with null in its own counter.
        for target in ("label", "type"):
            if target not in kept:
                continue
            stats = item["eligible_target_distributions"][target]
            values = kept[target]
            counts = Counter(stats.get("values", {}))
            counts.update({str(k): int(v) for k, v in values.dropna().astype(str).value_counts().items()})
            stats.update(values=dict(counts), native_nulls=stats.get("native_nulls", 0) + int(values.isna().sum()))
        return kept

    def report(self) -> dict:
        # Absolute paths and mtimes are audit metadata, not population identity.
        # The ID binds the ordered membership to exact source-file content.
        identity = {"format": "source-content-and-membership/1", "policy_version": DATA_QUALITY_POLICY_VERSION,
                    "schema_version": TON_IOT_SCHEMA_VERSION, "rule_id": RULE_ID,
                    "files": [{"source_id": f["source_id"], "source_sha256": f["source_fingerprint"]["sha256"],
                               "original_rows": f["original_rows"],
                               "quarantined_row_positions_0based": f["quarantined_row_positions_0based"]} for f in self.files]}
        payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        distributions = {}
        for target in ("label", "type"):
            counts, nulls = Counter(), 0
            for item in self.files:
                stats = item["eligible_target_distributions"][target]
                counts.update(stats.get("values", {}))
                nulls += stats.get("native_nulls", 0)
            distributions[target] = {"values": dict(sorted(counts.items())), "native_nulls": nulls}
        return {"policy_version": DATA_QUALITY_POLICY_VERSION, "schema_version": TON_IOT_SCHEMA_VERSION,
                "rule": {"id": RULE_ID, "column": "src_bytes", "operator": "exact_literal_equality",
                         "value": KNOWN_INVALID_SRC_BYTES, "action": "quarantine",
                         "reason": "Known nonnumeric token in a quantitative field; forensic classification D"},
                "original_rows": sum(f["original_rows"] for f in self.files),
                "quarantined_rows": sum(f["quarantined_rows"] for f in self.files),
                "eligible_rows": sum(f["eligible_rows"] for f in self.files),
                "eligible_target_distributions": distributions,
                "population_id": "sha256:" + hashlib.sha256(payload).hexdigest(),
                "population_identity_manifest": identity,
                "identity_definition": "Ordered source-relative file IDs, exact file SHA256, and all original logical row positions except the recorded exclusions. Independent of chunk size; representation-specific; no deduplication.",
                "files": self.files}

    def save(self, path: str | Path | None = None) -> dict:
        report = self.report()
        if path is None:
            path = Path("artifacts/data_quality") / (report["population_id"].split(":", 1)[1] + ".json")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report
