"""CLI and orchestration for the minimal IDS experiment pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .backend import resolve_backend
from .clustering import fit_predict_clustering
from .config import PipelineConfig, config_to_dict, load_config
from .data_loading import load_dataset
from .feature_policy import FeaturePolicy, apply_proxy_feature_policies
from .leakage_checks import write_forbidden_columns_check
from .preprocessing import fit_transform_preprocessing, describe_feature_transformations, _select_log_cols
from .schema import TON_IOT_SCHEMA
from .profiling import Profiler
from .representatives import build_cluster_representatives
from .splitting import create_splits, save_splits
from .supervised import train_random_forest
from .utils import ensure_dir, set_seed, setup_logging, write_json

LOGGER = logging.getLogger(__name__)

STAGES = {"prepare", "cluster", "representatives", "supervised", "all"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal leakage-aware IoT/NIDS debug pipeline.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs.")
    parser.add_argument("--stage", default="all", choices=sorted(STAGES), help="Pipeline stage.")
    parser.add_argument(
        "--data-path",
        default=None,
        help="Optional CSV/Parquet path. If omitted, a synthetic dataset is generated.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    setup_logging(args.output_dir, level=config.log_level)
    run_experiment(config, args.output_dir, stage=args.stage, data_path=args.data_path)


def run_experiment(
    config: PipelineConfig,
    output_dir: str | Path,
    *,
    stage: str = "all",
    data_path: str | None = None,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"Unknown stage={stage!r}. Minimal version supports {sorted(STAGES)}.")

    output_dir = ensure_dir(output_dir)
    ensure_dir(output_dir / "artifacts")
    set_seed(config.random_state)
    profiler = Profiler()
    effective_backend = resolve_backend(config.compute_backend)
    profiler.add_metadata(
        backend_requested=config.compute_backend,
        backend_effective=effective_backend,
    )
    write_json(output_dir / "config_resolved.json", config_to_dict(config))

    with profiler.track("data"):
        resolved_data_path = data_path or f"synthetic://{config.synthetic_n}"
        df = load_dataset(
            resolved_data_path,
            random_state=config.random_state,
            compute_backend=config.compute_backend,
            schema_report_path=output_dir / "schema_normalization_report.json",
            data_quality_report_path=output_dir / "data_quality_report.json",
        )
    write_json(
        output_dir / "dataset_info.json",
        {
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "data_path": resolved_data_path,
            "compute_backend": config.compute_backend,
            "data_quality_population_id": df.attrs.get("data_quality_population_id"),
        },
    )

    policy = FeaturePolicy.from_name(config.feature_policy)
    raw_feature_cols = policy.select_features(
        df,
        label_col=config.label_col,
        type_col=config.type_col,
        extra_target_cols=["cluster_id"],
    )
    df, feature_cols, proxy_metadata = apply_proxy_feature_policies(
        df,
        raw_feature_cols,
        timestamp_col=config.timestamp_col,
        timestamp_policy=config.timestamp_policy,
        service_policy=config.service_policy,
    )
    selected_payload = {
        "feature_policy": config.feature_policy,
        "raw_selected_features_before_proxy_policy": raw_feature_cols,
        "selected_features": feature_cols,
        "n_selected_features": len(feature_cols),
        **proxy_metadata,
        **policy.selection_metadata(feature_cols),
    }
    if policy.name == "behavioral_strict":
        numeric_cols = [c for c in feature_cols if TON_IOT_SCHEMA[c].model_treatment == "numeric"]
        log_cols = _select_log_cols(numeric_cols, config.log1p_columns) if config.log1p_numeric else []
        selected_payload.update(
            transformation_status="configured_not_fitted",
            feature_transformations=describe_feature_transformations(feature_cols, log_cols),
        )
    write_json(output_dir / "selected_features.json", selected_payload)
    write_forbidden_columns_check(
        output_dir / "forbidden_columns_check.json",
        df,
        feature_cols,
        feature_policy=config.feature_policy,
        label_col=config.label_col,
        type_col=config.type_col,
        context="selected features",
    )

    with profiler.track("split"):
        from .split_protocols import SplitValidationError
        try:
            splits = create_splits(df, config)
        except SplitValidationError as exc:
            write_json(output_dir / "split_failure_diagnostics.json", exc.diagnostics)
            raise
        save_splits(splits, output_dir, df=df, config=config)

    if stage == "prepare":
        profiling_payload = profiler.to_dict()
        write_json(output_dir / "profiling.json", profiling_payload)
        LOGGER.info("Prepare stage complete. Outputs written to %s", output_dir)
        return {"selected_features": selected_payload}

    with profiler.track("preprocessing_svd"):
        X, _bundle = fit_transform_preprocessing(df, splits, feature_cols, config, output_dir)

    if policy.name == "behavioral_strict":
        fitted = json.loads((output_dir / "artifacts/preprocessing_metadata.json").read_text(encoding="utf-8"))
        selected_payload.update(
            transformation_status="fitted_on_train",
            feature_transformations=fitted["feature_transformations"],
            conn_state_encoding=fitted["conn_state_encoding"],
            quantitative_output_order=fitted["numeric_cols"],
            dimensionality_reduction={"algorithm": fitted["reducer"], "components": fitted["svd_components_effective"]},
            transformed_shapes=fitted["transformed_shapes"],
        )
        write_json(output_dir / "selected_features.json", selected_payload)

    with profiler.track("clustering"):
        clustering = fit_predict_clustering(X, df, splits, feature_cols, config, output_dir)

    if stage == "cluster":
        profiling_payload = profiler.to_dict()
        write_json(output_dir / "profiling.json", profiling_payload)
        LOGGER.info("Cluster stage complete. Outputs written to %s", output_dir)
        return {
            "selected_features": selected_payload,
            "clustering": clustering["metrics"],
            "profiling": profiling_payload,
        }

    with profiler.track("representatives"):
        y_train_dict = _training_targets_for_representatives(df, splits, config, clustering["labels"])
        representatives = build_cluster_representatives(
            X["train"],
            y_train_dict,
            clustering["labels"]["train"],
            clustering["distances"]["train"],
            config,
            effective_backend,
            output_dir,
            train_indices=splits["train"],
        )

    if stage == "representatives":
        profiling_payload = profiler.to_dict()
        write_json(output_dir / "profiling.json", profiling_payload)
        LOGGER.info("Representatives stage complete. Outputs written to %s", output_dir)
        return {
            "selected_features": selected_payload,
            "clustering": clustering["metrics"],
            "representatives": representatives["metadata"],
            "profiling": profiling_payload,
        }

    with profiler.track("supervised_total"):
        supervised_metrics = train_random_forest(
            X,
            df,
            splits,
            feature_cols,
            config,
            output_dir,
            cluster_labels=clustering["labels"],
            representatives=representatives,
            profiler=profiler,
        )

    profiling_payload = profiler.to_dict()
    write_json(output_dir / "profiling.json", profiling_payload)
    LOGGER.info("Pipeline complete. Outputs written to %s", output_dir)
    return {
        "selected_features": selected_payload,
        "clustering": clustering["metrics"],
        "supervised": supervised_metrics,
        "profiling": profiling_payload,
    }


def _training_targets_for_representatives(
    df,
    splits: dict[str, np.ndarray],
    config: PipelineConfig,
    cluster_labels: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    y: dict[str, np.ndarray] = {}
    train_idx = splits["train"]
    for target in config.targets:
        if target == "label":
            y[target] = df.iloc[train_idx][config.label_col].astype(str).to_numpy()
        elif target == "type":
            y[target] = df.iloc[train_idx][config.type_col].astype(str).to_numpy()
        elif target == "cluster_id":
            y[target] = np.asarray(cluster_labels["train"]).astype(str)
        else:
            raise ValueError("targets must contain only: label, type, cluster_id")
    return y


if __name__ == "__main__":
    main()
