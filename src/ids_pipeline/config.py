"""YAML configuration for the minimal pandas + scikit-learn pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PipelineConfig:
    dataset_name: str = "synthetic_debug"
    synthetic_n: int = 300
    label_col: str = "label"
    type_col: str = "type"
    feature_policy: str = "no_raw_ip_port"
    compute_backend: str = "cpu"

    split_strategy: str = "random_stratified"
    group_cols: list[str] = field(default_factory=list)
    timestamp_col: str = "timestamp"
    timestamp_policy: str = "keep"
    timestamp_unit: str | None = "auto"
    temporal_bucket_freq: str | None = "1s"
    service_policy: str = "keep"
    test_size: float = 0.25
    val_size: float = 0.20

    svd_components: int = 8
    onehot_min_frequency: int | None = 1
    gpu_max_categories_per_col: int = 64
    log1p_numeric: bool = True
    log1p_patterns: list[str] = field(default_factory=lambda: ["bytes", "pkts", "packets", "count", "duration"])
    # log1p_patterns remains parseable for old YAMLs; selection now uses exact names.
    log1p_columns: list[str] | None = None

    selected_k: int = 3
    cluster_batch_size: int = 64
    cluster_n_init: int = 3
    cluster_max_iter: int = 50
    silhouette_sample_size: int = 200

    representative_strategy: str = "full"
    use_representatives_for_supervised: bool = False
    representatives_per_cluster: int = 20
    boundary_per_cluster: int = 5
    targets: list[str] = field(default_factory=lambda: ["label"])

    random_forest_estimators: int = 100
    random_state: int = 42
    # None inherits the legacy random_state independently; no stage seed feeds another.
    split_seed: int | None = None
    clustering_seed: int | None = None
    selection_seed: int | None = None
    model_seed: int | None = None
    selection_budget: int | str | None = None
    selection_method: str | None = None
    n_jobs: int = 1
    log_level: str = "INFO"

    extra: dict[str, Any] = field(default_factory=dict)

    def with_updates(self, **updates: Any) -> "PipelineConfig":
        return replace(self, **updates)

    def seed_for(self, stage: str) -> int:
        if stage not in {"split", "clustering", "selection", "model"}:
            raise ValueError(f"Unknown seed stage: {stage}")
        value = getattr(self, stage + "_seed")
        value = self.random_state if value is None else value
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
            raise ValueError(f"{stage}_seed must be an integer in [0, 2**32)")
        return value

    def for_stage(self, stage: str) -> "PipelineConfig":
        """Adapter for frozen APIs that still consume random_state."""
        seeds = {s + "_seed": self.seed_for(s) for s in ("split", "clustering", "selection", "model")}
        return replace(self, random_state=self.seed_for(stage), **seeds)


def load_config(path: str | Path) -> PipelineConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    known = set(PipelineConfig.__dataclass_fields__)
    config_kwargs = {key: value for key, value in raw.items() if key in known}
    extra = {key: value for key, value in raw.items() if key not in known}
    config_kwargs["extra"] = extra
    return PipelineConfig(**config_kwargs)


def config_to_dict(config: PipelineConfig) -> dict[str, Any]:
    payload = dict(config.__dict__)
    payload["effective_seeds"] = {s: config.seed_for(s) for s in ("split", "clustering", "selection", "model")}
    payload["selection_mode"] = "legacy" if config.selection_budget is None else "total_budget"
    return payload
