"""Dataset loading helpers for CSV, Parquet, and synthetic debug data."""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .backend import requested_backend
from .dataset_schema import is_ton_iot, normalize_dataset_schema, read_csv_preserving_schema
from .data_quality_policy import DataQualityPopulation, source_fingerprint, assert_source_unchanged

LOGGER = logging.getLogger(__name__)


def load_dataset(
    data_path: str,
    *,
    sample_size: int | None = None,
    random_state: int = 42,
    compute_backend: str = "cpu",
    schema_report_path: str | Path | None = None,
    data_quality_report_path: str | Path | None = None,
) -> pd.DataFrame:
    if data_path.startswith("synthetic://"):
        size_text = data_path.split("://", 1)[1] or "300"
        n_rows = int(size_text)
        return make_synthetic_dataset(n_rows=n_rows, random_state=random_state)

    files = _resolve_files(data_path)
    if not files:
        raise FileNotFoundError(f"No CSV/Parquet files found for data_path={data_path!r}")

    # Read sources independently so original file/row identity survives filtering,
    # including the cuDF path. No provenance columns are exposed as model features.
    population = DataQualityPopulation()
    root = Path(os.path.commonpath([str(path.resolve().parent) for path in files]))
    frames, ton_flags, original_offset = [], [], 0
    for path in files:
        before = path.stat()
        if path.suffix.lower() in {".parquet", ".pq"} and requested_backend(compute_backend) in {"gpu", "auto"}:
            frame = _read_parquet_with_cudf([path], require_cudf=requested_backend(compute_backend) == "gpu")
        else:
            frame = _read_one_file(path)
        raw_count = len(frame)
        if len(files) > 1:
            frame.index = pd.RangeIndex(original_offset, original_offset + raw_count)
        original_offset += raw_count
        ton_flags.append(is_ton_iot(frame.columns))
        if ton_flags[-1]:
            fingerprint = source_fingerprint(path)
            if (before.st_size, before.st_mtime_ns) != (fingerprint["bytes"], fingerprint["mtime_ns"]):
                raise RuntimeError(f"Source changed while reading: {path}")
            source_id = path.resolve().relative_to(root).as_posix()
            population.register_file(source_id, fingerprint, source_path=str(path.resolve()))
            frame = population.apply(frame, source_id=source_id, row_offset=0)
            assert_source_unchanged(path, fingerprint)
        frames.append(frame)
    if any(ton_flags) and not all(ton_flags):
        raise ValueError("Cannot mix TON_IoT and unrecognized partitions in one population")
    df = pd.concat(frames, ignore_index=False) if len(frames) > 1 else frames[0]
    if is_ton_iot(df.columns):
        if data_quality_report_path is None and schema_report_path is not None:
            data_quality_report_path = Path(schema_report_path).with_name("data_quality_report.json")
        quality_report = population.save(data_quality_report_path)
        LOGGER.info("Data quality: original=%d quarantined=%d eligible=%d population=%s",
                    quality_report["original_rows"], quality_report["quarantined_rows"],
                    quality_report["eligible_rows"], quality_report["population_id"])
        df, report = normalize_dataset_schema(df, report_path=schema_report_path)
        df.attrs["schema_version"] = report["schema_version"]
        df.attrs["data_quality_report"] = quality_report
        df.attrs["data_quality_population_id"] = quality_report["population_id"]
        LOGGER.info("Applied semantic schema %s; unknown columns=%s", report["schema_version"], report["unknown_columns"])
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    LOGGER.info("Loaded dataset with shape=%s from %d file(s)", df.shape, len(files))
    return df


def _resolve_files(data_path: str) -> list[Path]:
    path = Path(data_path)
    if any(ch in data_path for ch in "*?[]"):
        return sorted(Path(p) for p in glob.glob(data_path))
    if path.is_dir():
        files = sorted(path.rglob("*.csv")) + sorted(path.rglob("*.parquet"))
        return files
    return [path] if path.exists() else []


def _read_one_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    LOGGER.info("Reading %s", path)
    if suffix == ".csv":
        return read_csv_preserving_schema(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension for {path}. Use CSV or Parquet.")


def _all_parquet(files: list[Path]) -> bool:
    return bool(files) and all(path.suffix.lower() in {".parquet", ".pq"} for path in files)


def _read_parquet_with_cudf(files: list[Path], *, require_cudf: bool) -> pd.DataFrame:
    try:
        import cudf
    except Exception as exc:
        if require_cudf:
            raise RuntimeError(
                "GPU backend requested with Parquet input, but cuDF is not importable. "
                "Install cudf or set IDS_COMPUTE_BACKEND=cpu."
            ) from exc
        LOGGER.info("cuDF is not available. Falling back to pandas.read_parquet.")
        return _read_parquet_with_pandas(files)

    LOGGER.info("Reading %d Parquet file(s) with cuDF", len(files))
    try:
        gpu_df = cudf.read_parquet([str(path) for path in files])
        return gpu_df.to_pandas()
    except Exception as exc:
        LOGGER.warning(
            "cuDF could not read all Parquet files together (%s). "
            "Falling back to pandas for IO; GPU backend can still be used after loading. "
            "For faster cuDF IO, reconvert the CSV partitions with scripts/convert_csv_to_parquet.py.",
            exc,
        )
        return _read_parquet_with_pandas(files)


def _read_parquet_with_pandas(files: list[Path]) -> pd.DataFrame:
    frames = []
    for path in files:
        LOGGER.info("Reading %s", path)
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def make_synthetic_dataset(n_rows: int = 300, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    services = np.array(["http", "dns", "ssh", "mqtt"])
    protos = np.array(["tcp", "udp", "icmp"])
    states = np.array(["S0", "SF", "REJ", "RSTO"])
    attack_types = np.array(["normal", "scan", "dos", "exfiltration"])

    label = rng.binomial(1, 0.35, size=n_rows)
    type_values = np.where(label == 0, "normal", rng.choice(attack_types[1:], size=n_rows, p=[0.45, 0.4, 0.15]))
    src_pkts = rng.poisson(lam=np.where(label == 1, 22, 8), size=n_rows) + 1
    dst_pkts = rng.poisson(lam=np.where(label == 1, 12, 7), size=n_rows) + 1
    src_bytes = src_pkts * rng.integers(40, 500, size=n_rows) + label * rng.integers(0, 2000, size=n_rows)
    dst_bytes = dst_pkts * rng.integers(40, 600, size=n_rows)
    duration = rng.gamma(shape=np.where(label == 1, 1.6, 2.5), scale=1.2, size=n_rows)

    src_hosts = [f"10.0.{i // 255}.{i % 255}" for i in range(1, 80)]
    dst_hosts = [f"172.16.{i // 255}.{i % 255}" for i in range(1, 60)]
    timestamps = pd.date_range("2024-01-01", periods=n_rows, freq="min")

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "dataset_id": rng.choice(["synthetic_a", "synthetic_b"], size=n_rows),
            "src_ip": rng.choice(src_hosts, size=n_rows),
            "dst_ip": rng.choice(dst_hosts, size=n_rows),
            "src_port": rng.integers(1024, 65535, size=n_rows),
            "dst_port": rng.choice([22, 53, 80, 443, 1883, 8080], size=n_rows),
            "proto": rng.choice(protos, size=n_rows),
            "service": rng.choice(services, size=n_rows),
            "conn_state": rng.choice(states, size=n_rows),
            "tcp_flags": rng.choice(["A", "S", "SA", "PA", "R"], size=n_rows),
            "duration": duration,
            "src_bytes": src_bytes,
            "dst_bytes": dst_bytes,
            "src_pkts": src_pkts,
            "dst_pkts": dst_pkts,
            "byte_ratio": src_bytes / np.maximum(dst_bytes, 1),
            "pkt_ratio": src_pkts / np.maximum(dst_pkts, 1),
            "flow_id": [f"flow-{i}" for i in range(n_rows)],
            "uid": [f"uid-{i}" for i in range(n_rows)],
            "label": label,
            "type": type_values,
        }
    )
