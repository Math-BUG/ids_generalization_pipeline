"""Stateless TON_IoT normalization, shared by ingestion and preprocessing."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import TON_IOT_DERIVED_SCHEMA, TON_IOT_SCHEMA, TON_IOT_SCHEMA_VERSION


_REGISTERED_COLUMNS = {**TON_IOT_SCHEMA, **TON_IOT_DERIVED_SCHEMA}


class SchemaValidationError(ValueError):
    """Invalid input; the complete bounded diagnostic remains available."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        bad = [name for name, info in report["columns"].items() if info["invalid_count"]]
        super().__init__(f"{TON_IOT_SCHEMA_VERSION}: invalid values in {bad}; see schema report")


def is_ton_iot(columns) -> bool:
    """Recognize the observed network header, not a directory/config nickname."""
    return {"ts", "src_ip", "dst_ip", "src_bytes", "dns_qtype"}.issubset(columns)


def read_csv_preserving_schema(path: str | Path) -> pd.DataFrame:
    """Preserve raw TON_IoT tokens; pandas' implicit NA vocabulary is disabled."""
    columns = pd.read_csv(path, nrows=0).columns
    if is_ton_iot(columns):
        return pd.read_csv(path, dtype="string", keep_default_na=False, low_memory=False)
    return pd.read_csv(path, low_memory=False)


def normalize_dataset_schema(
    df: pd.DataFrame,
    *,
    report_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize present, registered columns without fitting, filtering or clipping.

    Unknown columns are preserved and reported, never guessed here. Missing
    optional columns are not manufactured. Invalid input always raises, after
    writing the report when requested; no partially coerced frame is returned.
    Native nulls and the exact registered string tokens are the only missing values.
    """
    if not df.columns.is_unique:
        raise ValueError("Semantic normalization requires unique column names")
    out = df.copy()
    report: dict[str, Any] = {
        "schema_version": TON_IOT_SCHEMA_VERSION,
        "rows_before": len(df), "rows_after": len(df),
        "index_preserved": True,
        "unknown_columns": [str(c) for c in df.columns if c not in _REGISTERED_COLUMNS],
        "absent_columns": [c for c in TON_IOT_SCHEMA if c not in df.columns],
        "columns": {}, "ok": True,
    }
    for name in df.columns:
        if name not in _REGISTERED_COLUMNS:
            continue
        spec = _REGISTERED_COLUMNS[name]
        raw = df[name]
        missing = raw.isna() | raw.isin(spec.missing_tokens)
        values = raw.mask(missing)
        masks: dict[str, pd.Series] = {}
        if not spec.nullable:
            masks["missing_required"] = missing
        numeric = spec.storage_dtype in {"float64", "Int64"} or spec.semantic_type == "nominal_code"
        if numeric:
            parsed = pd.to_numeric(values, errors="coerce").astype("float64")
            masks["conversion_failure"] = ~missing & parsed.isna()
            masks["infinite"] = pd.Series(np.isinf(parsed), index=df.index)
            finite = parsed.notna() & ~masks["infinite"]
            domain = pd.Series(False, index=df.index)
            if spec.minimum is not None:
                domain |= finite & (parsed < spec.minimum)
            if spec.maximum is not None:
                domain |= finite & (parsed > spec.maximum)
            if spec.integral:
                domain |= finite & (parsed % 1 != 0)
                # Do not silently round counters/codes beyond float64 exact integers.
                masks["unsafe_integer_precision"] = finite & (parsed.abs() > 2**53 - 1)
            if spec.semantic_type == "timestamp":
                # Validate representability without passing huge floats to pandas'
                # datetime converter (some versions overflow even with coerce).
                domain |= finite & (parsed >= np.iinfo("int64").max / 1e9)
            masks["domain_violation"] = domain
            invalid = _union_masks(masks, df.index)
            parsed = parsed.mask(invalid)
            if spec.semantic_type == "nominal_code":
                out[name] = parsed.astype("Int64").astype("string")
            else:
                out[name] = parsed.astype(spec.storage_dtype)
        elif spec.semantic_type == "boolean":
            # Explicit accepted representations, canonical output T/F; no truthiness.
            tokens = values.astype("string")
            mapping = {"T": "T", "F": "F", "true": "T", "false": "F",
                       "True": "T", "False": "F", "TRUE": "T", "FALSE": "F",
                       "1": "T", "0": "F", "1.0": "T", "0.0": "F"}
            normalized = tokens.map(mapping)
            masks["domain_violation"] = ~missing & normalized.isna()
            invalid = _union_masks(masks, df.index)
            out[name] = normalized.astype("string")
        else:
            # Preserve text, case, whitespace and list serialization verbatim.
            invalid = _union_masks(masks, df.index)
            out[name] = values.astype("string")
        info = {
            **asdict(spec), "source_dtype": str(raw.dtype),
            "normalized_dtype": str(out[name].dtype),
            "native_missing_count": int(raw.isna().sum()),
            "sentinel_missing_count": int((missing & raw.notna()).sum()),
            "missing_after": int(out[name].isna().sum()),
            "invalid_count": int(invalid.sum()),
            "issues": {},
        }
        for kind, mask in masks.items():
            positions = np.flatnonzero(mask.to_numpy(dtype=bool))[:5]
            info["issues"][kind] = {
                "count": int(mask.sum()),
                "examples": [{"position": int(p), "index": str(df.index[p])[:80],
                              "value": str(raw.iloc[p])[:80]} for p in positions],
            }
        report["columns"][name] = info
        report["ok"] = report["ok"] and not bool(invalid.any())
    report["index_preserved"] = out.index.equals(df.index)
    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["ok"]:
        raise SchemaValidationError(report)
    return out, report


def _union_masks(masks: dict[str, pd.Series], index: pd.Index) -> pd.Series:
    result = pd.Series(False, index=index)
    for mask in masks.values():
        result |= mask.fillna(False)
    return result


def model_feature_types(
    df: pd.DataFrame, feature_cols: list[str], *, strict: bool = False,
) -> tuple[list[str], list[str]]:
    """One dispatch for CPU and GPU; dtype fallback only for non-TON extensions."""
    numeric, categorical = [], []
    for name in feature_cols:
        spec = _REGISTERED_COLUMNS.get(name)
        if spec is not None:
            if spec.model_treatment == "target":
                raise ValueError(f"Target {name!r} cannot be a feature")
            treatment = spec.model_treatment
        elif strict:
            raise ValueError(f"Unregistered TON_IoT feature {name!r}; extend the semantic schema explicitly")
        else:
            treatment = "numeric" if pd.api.types.is_numeric_dtype(df[name]) else "categorical"
        (numeric if treatment == "numeric" else categorical).append(name)
    return numeric, categorical
