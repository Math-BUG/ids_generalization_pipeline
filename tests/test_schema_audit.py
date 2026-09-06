import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


spec = importlib.util.spec_from_file_location("audit", Path(__file__).parents[1] / "scripts/audit_ton_iot_schema.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_audit_continues_after_invalid_chunks_and_preserves_sources(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    frame = pd.DataFrame({"src_bytes": ["bad", "-", "3", "bad2"],
                          "label": [0, 1, 2, 0], "type": ["normal"] * 4})
    path = source / "Network_dataset_1.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    original = path.read_bytes()
    result = audit.run_audit(source, output, "csv", chunk_size=2, expected_files=1)
    assert result["complete"] and not result["compatible"]
    assert result["rows"] == 4 and result["files"][0]["chunks"] == 2
    info = result["columns"]["src_bytes"]
    assert info["invalid_cells"] == 2
    assert info["sentinel_tokens"] == {"": 0, "-": 1}
    assert info["recognized_missing"] == 1
    assert info["issues"]["conversion_failure"]["examples"][-1]["row_0based"] == 3
    assert result["columns"]["label"]["invalid_label"] == 1
    assert path.read_bytes() == original


def test_audit_parquet_physical_types_and_domain_breakdown(tmp_path):
    frame = pd.DataFrame({"src_pkts": [-1.0, 1.5, np.inf, np.nan],
                          "src_port": [65536, -1, 80, 53],
                          "dns_AA": ["yes", "T", "F", "-"],
                          "label": [0, 1, 0, 1], "type": ["normal"] * 4})
    path = tmp_path / "Network_dataset_1.parquet"
    frame.to_parquet(path, index=False)
    item = audit.audit_file(path, 2)
    pkts = item["columns"]["src_pkts"]
    assert pkts["physical_dtypes"] == {"double": 4}
    assert pkts["native_nulls"] == 1
    assert pkts["negative_nonnegative_domain"] == 1
    assert pkts["nonintegral"] == 1
    assert pkts["issues"]["infinite"]["count"] == 1
    assert pkts["invalid_cells"] == 3
    assert item["columns"]["src_port"]["ports_out_of_range"] == 2
    assert item["columns"]["dns_AA"]["unexpected_boolean"] == 1
    assert item["row_count_matches_metadata"]
    assert item["source_unchanged_size_mtime"]
