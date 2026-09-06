"""Anti-leakage checks for the minimal IDS pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import DATASET_ID_COLUMNS, IP_COLUMNS, RAW_IDENTITY_COLUMNS, SERVICE_COLUMNS, TARGET_COLUMNS
from .utils import write_json
from .feature_policy import BEHAVIORAL_STRICT_FEATURES, BEHAVIORAL_STRICT_VERSION


class LeakageError(ValueError):
    """Raised when target or policy-forbidden columns enter model features."""


def normalize_col(name: str) -> str:
    return str(name).strip().lower()


def forbidden_columns_for_policy(
    *,
    feature_policy: str,
    label_col: str = "label",
    type_col: str = "type",
) -> set[str]:
    targets = {normalize_col(c) for c in TARGET_COLUMNS | {label_col, type_col} if c}
    always_forbidden = targets | {normalize_col(c) for c in DATASET_ID_COLUMNS}

    if feature_policy == "all_except_labels":
        return always_forbidden
    if feature_policy == "no_raw_ip":
        stable_ids = {"uid", "flow_id", "id"}
        return always_forbidden | {normalize_col(c) for c in IP_COLUMNS | stable_ids}
    if feature_policy == "no_raw_ip_port":
        return always_forbidden | {normalize_col(c) for c in RAW_IDENTITY_COLUMNS}
    if feature_policy == "behavioral_strict":
        return always_forbidden | {normalize_col(c) for c in RAW_IDENTITY_COLUMNS | SERVICE_COLUMNS}
    raise ValueError(f"Unknown feature_policy={feature_policy!r}")


def check_for_leakage_columns(
    df: pd.DataFrame | None,
    feature_cols: Iterable[str],
    *,
    feature_policy: str = "no_raw_ip_port",
    label_col: str = "label",
    type_col: str = "type",
    context: str = "features",
) -> dict[str, Any]:
    feature_cols = list(feature_cols)
    normalized_features = {normalize_col(c) for c in feature_cols}
    forbidden = forbidden_columns_for_policy(
        feature_policy=feature_policy,
        label_col=label_col,
        type_col=type_col,
    )
    leaked = sorted(normalized_features & forbidden)
    result = {
        "ok": not leaked,
        "context": context,
        "feature_policy": feature_policy,
        "forbidden_found": leaked,
        "n_features_checked": len(feature_cols),
    }
    if leaked:
        raise LeakageError(
            f"Leakage columns detected before {context}: {leaked}. "
            f"feature_policy={feature_policy!r}"
        )

    if feature_policy == "behavioral_strict" and tuple(feature_cols) != BEHAVIORAL_STRICT_FEATURES:
        raise LeakageError(
            f"{BEHAVIORAL_STRICT_VERSION} requires exactly this ordered feature list before {context}: "
            f"{list(BEHAVIORAL_STRICT_FEATURES)}; received {feature_cols}"
        )

    if df is not None:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise LeakageError(f"Feature columns missing before {context}: {missing}")
    return result


def write_forbidden_columns_check(
    path: str | Path,
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    *,
    feature_policy: str,
    label_col: str,
    type_col: str,
    context: str,
) -> dict[str, Any]:
    try:
        result = check_for_leakage_columns(
            df,
            feature_cols,
            feature_policy=feature_policy,
            label_col=label_col,
            type_col=type_col,
            context=context,
        )
    except LeakageError as exc:
        result = {
            "ok": False,
            "context": context,
            "feature_policy": feature_policy,
            "error": str(exc),
        }
        write_json(path, result)
        raise
    write_json(path, result)
    return result
