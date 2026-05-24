"""CLI and orchestration for the minimal IDS experiment pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from .clustering import fit_predict_clustering
from .config import PipelineConfig, config_to_dict, load_config
from .data_loading import load_dataset
from .feature_policy import FeaturePolicy
from .leakage_checks import write_forbidden_columns_check
from .preprocessing import fit_transform_preprocessing
from .profiling import Profiler
from .splitting import create_splits, save_splits
from .supervised import train_random_forest
from .utils import ensure_dir, set_seed, setup_logging, write_json

LOGGER = logging.getLogger(__name__)

STAGES = {"prepare", "all"}


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
    write_json(output_dir / "config_resolved.json", config_to_dict(config))

    with profiler.track("data"):
        resolved_data_path = data_path or f"synthetic://{config.synthetic_n}"
        df = load_dataset(
            resolved_data_path,
            random_state=config.random_state,
            compute_backend=config.compute_backend,
        )
    write_json(
        output_dir / "dataset_info.json",
        {
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "data_path": resolved_data_path,
            "compute_backend": config.compute_backend,
        },
    )

    policy = FeaturePolicy.from_name(config.feature_policy)
    feature_cols = policy.select_features(df, label_col=config.label_col, type_col=config.type_col)
    selected_payload = {
        "feature_policy": config.feature_policy,
        "selected_features": feature_cols,
        "n_selected_features": len(feature_cols),
    }
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
        splits = create_splits(df, config)
        save_splits(splits, output_dir)

    if stage == "prepare":
        profiling_payload = profiler.to_dict()
        write_json(output_dir / "profiling.json", profiling_payload)
        LOGGER.info("Prepare stage complete. Outputs written to %s", output_dir)
        return {"selected_features": selected_payload}

    with profiler.track("preprocessing_svd"):
        X, _bundle = fit_transform_preprocessing(df, splits, feature_cols, config, output_dir)

    with profiler.track("clustering"):
        clustering = fit_predict_clustering(X, df, splits, feature_cols, config, output_dir)

    with profiler.track("random_forest"):
        supervised_metrics = train_random_forest(X, df, splits, feature_cols, config, output_dir)

    profiling_payload = profiler.to_dict()
    write_json(output_dir / "profiling.json", profiling_payload)
    LOGGER.info("Pipeline complete. Outputs written to %s", output_dir)
    return {
        "selected_features": selected_payload,
        "clustering": clustering["metrics"],
        "supervised": supervised_metrics,
        "profiling": profiling_payload,
    }


if __name__ == "__main__":
    main()
