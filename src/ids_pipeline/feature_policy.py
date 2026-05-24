"""Feature selection policies for leakage-aware experiments."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .schema import (
    BEHAVIORAL_ALLOWED_HINTS,
    DATASET_ID_COLUMNS,
    IP_COLUMNS,
    PORT_COLUMNS,
    PROHIBITED_FEATURE_COLUMNS,
    RAW_IDENTITY_COLUMNS,
    STABLE_IDENTIFIER_HINTS,
    TARGET_COLUMNS,
)


@dataclass(frozen=True)
class FeaturePolicy:
    name: str

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

        if self.name == "all_except_labels":
            remove = target_like | DATASET_ID_COLUMNS
        elif self.name == "no_raw_ip":
            remove = target_like | IP_COLUMNS | DATASET_ID_COLUMNS | {"uid", "flow_id", "id"}
        elif self.name == "no_raw_ip_port":
            remove = target_like | IP_COLUMNS | PORT_COLUMNS | DATASET_ID_COLUMNS | {"uid", "flow_id", "id"}
        elif self.name == "behavioral_strict":
            remove = (
                target_like
                | RAW_IDENTITY_COLUMNS
                | DATASET_ID_COLUMNS
                | {"service", "svc", "application", "app", "proto_service"}
            )
        else:  # pragma: no cover - guarded by from_name
            raise ValueError(self.name)

        remove_lower = {c.lower() for c in remove}
        features = [c for c in df.columns if c.lower() not in remove_lower]

        if self.name == "behavioral_strict":
            features = [c for c in features if self._is_behavioral_column(c)]

        return features

    @staticmethod
    def _is_behavioral_column(column: str) -> bool:
        name = column.lower()
        if name in PROHIBITED_FEATURE_COLUMNS or name == "service":
            return False
        if any(hint in name for hint in STABLE_IDENTIFIER_HINTS):
            return False
        return any(hint in name for hint in BEHAVIORAL_ALLOWED_HINTS)

    def allowed_leakage_baseline_columns(self) -> set[str]:
        """Columns allowed only when the caller explicitly enables unsafe baselines."""

        if self.name != "all_except_labels":
            return set()
        return {c for c in PROHIBITED_FEATURE_COLUMNS if c not in TARGET_COLUMNS | DATASET_ID_COLUMNS}
