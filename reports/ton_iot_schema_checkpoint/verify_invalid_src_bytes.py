"""Bounded follow-up: exact numeric-failure tokens in affected original CSVs."""
from collections import Counter
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
audit = json.loads((ROOT / "parquet/audit.json").read_text())
results = []
total = Counter()
for item in audit["files"]:
    expected = item["columns"]["src_bytes"]["issues"]["conversion_failure"]["count"]
    if not expected:
        continue
    path = Path("D:/IC/Dataset") / item["file"].replace(".parquet", ".csv")
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    counts = Counter()
    examples = []
    rows = 0
    for frame in pd.read_csv(path, usecols=["src_bytes"], dtype="string", keep_default_na=False,
                             na_filter=False, chunksize=100000, skip_blank_lines=False):
        raw = frame.src_bytes
        parsed = pd.to_numeric(raw, errors="coerce")
        bad = parsed.isna() & ~raw.isin(["", "-"]) & raw.notna()
        counts.update(raw[bad].value_counts().to_dict())
        for idx, value in raw[bad].head(5).items():
            if len(examples) < 5:
                examples.append({"row_0based": int(idx), "value": value})
        rows += len(frame)
    after = (path.stat().st_size, path.stat().st_mtime_ns)
    assert before == after
    assert sum(counts.values()) == expected
    total.update(counts)
    results.append({"file": path.name, "source": str(path), "column": "src_bytes", "rows_checked": rows,
                    "source_signature": before, "source_unchanged_size_mtime": before == after,
                    "conversion_failure_tokens": dict(counts), "examples": examples})
result = {"files": results, "token_counts": dict(total), "complete_for_affected_files": True}
(ROOT / "invalid_token_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
