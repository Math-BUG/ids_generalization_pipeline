"""Feature selection policies for leakage-aware experiments."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .schema import (
    DATASET_ID_COLUMNS,
    IP_COLUMNS,
    PORT_COLUMNS,
    PROHIBITED_FEATURE_COLUMNS,
    SERVICE_COLUMNS,
    TARGET_COLUMNS,
    TON_IOT_SCHEMA,
)

BEHAVIORAL_STRICT_VERSION = "behavioral_strict/2.0.0"
BEHAVIORAL_STRICT_FEATURES = (
    "duration", "src_bytes", "dst_bytes", "conn_state", "missed_bytes",
    "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes",
)


@dataclass(frozen=True)
class FeaturePolicy:
    name: str

    @property
    def version(self) -> str | None:
        return BEHAVIORAL_STRICT_VERSION if self.name == "behavioral_strict" else None

    def selection_metadata(self, found: list[str]) -> dict:
        requested = list(BEHAVIORAL_STRICT_FEATURES) if self.name == "behavioral_strict" else list(found)
        return {"feature_policy_version": self.version, "requested_features": requested,
                "found_features": list(found),
                "semantic_types": {c: TON_IOT_SCHEMA[c].semantic_type for c in found if c in TON_IOT_SCHEMA}}

    @classmethod
    def from_name(cls, name: str) -> "FeaturePolicy":
        valid = {"all_except_labels", "no_raw_ip", "no_raw_ip_port", "behavioral_strict"}
        if name not in valid:
            raise ValueError(f"Unknown feature_policy={name!r}. Valid options: {sorted(valid)}")
        return cls(name=name)

    def select_features(
        self,
        df: pd.DataFrame,
        *,
        label_col: str = "label",
        type_col: str = "type",
        extra_target_cols: list[str] | None = None,
    ) -> list[str]:
        target_like = set(TARGET_COLUMNS) | {label_col, type_col} | set(extra_target_cols or [])
        target_like = {c for c in target_like if c}

        if self.name == "behavioral_strict":
            if not df.columns.is_unique:
                raise ValueError(f"{self.version}: duplicate column names")
            missing = [c for c in BEHAVIORAL_STRICT_FEATURES if c not in df.columns]
            conflicts = [c for c in BEHAVIORAL_STRICT_FEATURES if c.lower() in {t.lower() for t in target_like}]
            if missing or conflicts:
                raise ValueError(f"{self.version}: required features missing={missing}; target conflicts={conflicts}")
            return list(BEHAVIORAL_STRICT_FEATURES)

        if self.name == "all_except_labels":
            remove = target_like | DATASET_ID_COLUMNS
        elif self.name == "no_raw_ip":
            remove = target_like | IP_COLUMNS | DATASET_ID_COLUMNS | {"uid", "flow_id", "id"}
        elif self.name == "no_raw_ip_port":
            remove = target_like | IP_COLUMNS | PORT_COLUMNS | DATASET_ID_COLUMNS | {"uid", "flow_id", "id"}
        else:  # pragma: no cover - guarded by from_name
            raise ValueError(self.name)

        remove_lower = {c.lower() for c in remove}
        features = [c for c in df.columns if c.lower() not in remove_lower]

        return features

    def allowed_leakage_baseline_columns(self) -> set[str]:
        """Columns allowed only when the caller explicitly enables unsafe baselines."""

        if self.name != "all_except_labels":
            return set()
        return {c for c in PROHIBITED_FEATURE_COLUMNS if c not in TARGET_COLUMNS | DATASET_ID_COLUMNS}


def apply_proxy_feature_policies(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    timestamp_col: str = "timestamp",
    timestamp_policy: str = "keep",
    service_policy: str = "keep",
) -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    """Apply configurable proxy controls after the base feature policy.

    Group-only columns such as IPs can remain in the dataframe for splitting,
    while this function controls whether timestamp/service proxies become model
    features.
    """

    out = df.copy()
    selected = list(feature_cols)
    removed_by_timestamp_policy: list[str] = []
    removed_by_service_policy: list[str] = []
    added_by_timestamp_policy: list[str] = []

    if timestamp_policy not in {"keep", "drop", "extract_causal"}:
        raise ValueError("timestamp_policy must be one of: keep, drop, extract_causal")
    if service_policy not in {"keep", "drop"}:
        raise ValueError("service_policy must be one of: keep, drop")

    if timestamp_policy == "drop":
        selected, removed_by_timestamp_policy = _remove_columns(selected, {timestamp_col})
    elif timestamp_policy == "extract_causal" and timestamp_col in out.columns:
        raw_timestamp_was_selected = timestamp_col in selected
        selected, removed_by_timestamp_policy = _remove_columns(selected, {timestamp_col})
        if raw_timestamp_was_selected:
            parsed = pd.to_datetime(out[timestamp_col], errors="coerce")
            hour_col = f"{timestamp_col}_hour"
            day_col = f"{timestamp_col}_day_of_week"
            out[hour_col] = parsed.dt.hour.fillna(-1).astype("int16")
            out[day_col] = parsed.dt.dayofweek.fillna(-1).astype("int16")
            selected.extend([hour_col, day_col])
            added_by_timestamp_policy.extend([hour_col, day_col])

    if service_policy == "drop":
        selected, removed_by_service_policy = _remove_columns(selected, SERVICE_COLUMNS)

    metadata = {
        "timestamp_policy": timestamp_policy,
        "service_policy": service_policy,
        "removed_by_timestamp_policy": removed_by_timestamp_policy,
        "removed_by_service_policy": removed_by_service_policy,
        "added_by_timestamp_policy": added_by_timestamp_policy,
    }
    return out, selected, metadata


def _remove_columns(feature_cols: list[str], remove: set[str]) -> tuple[list[str], list[str]]:
    remove_lower = {name.lower() for name in remove}
    kept = [col for col in feature_cols if col.lower() not in remove_lower]
    removed = [col for col in feature_cols if col.lower() in remove_lower]
    return kept, removed
