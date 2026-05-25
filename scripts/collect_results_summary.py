#!/usr/bin/env python3
"""Collect experiment output folders into a comparison CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TOP_LEVEL_STAGE_TIMES = [
    "data",
    "split",
    "preprocessing_svd",
    "clustering",
    "representatives",
    "supervised_total",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize IDS pipeline experiment outputs.")
    parser.add_argument("output_dirs", nargs="+", help="Experiment output directories.")
    parser.add_argument("--output-csv", required=True, help="Path for the summary CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for item in args.output_dirs:
        output_dir = Path(item).expanduser().resolve()
        rows.extend(summarize_experiment(output_dir))
    if not rows:
        raise ValueError("No rows collected. Check output directories.")
    out = pd.DataFrame(rows).sort_values(["experiment", "target"])
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv} with shape={out.shape}")


def summarize_experiment(output_dir: Path) -> list[dict[str, Any]]:
    metrics = read_json(output_dir / "metrics_supervised.json")
    profiling = read_json(output_dir / "profiling.json")
    config = read_json(output_dir / "config_resolved.json")
    clustering = read_json(output_dir / "metrics_clustering.json")
    representatives = read_optional_json(output_dir / "representatives_metadata.json")
    selected = read_optional_json(output_dir / "selected_features.json")

    rows = []
    targets = metrics.get("targets", {})
    for target, target_metrics in targets.items():
        test = target_metrics.get("test", {})
        val = target_metrics.get("val", {})
        row = {
            "experiment": output_dir.name,
            "output_dir": str(output_dir),
            "dataset_name": config.get("dataset_name"),
            "feature_policy": config.get("feature_policy"),
            "timestamp_policy": config.get("timestamp_policy"),
            "service_policy": config.get("service_policy"),
            "split_strategy": config.get("split_strategy"),
            "representative_strategy": metrics.get("representative_strategy", config.get("representative_strategy")),
            "use_representatives_for_supervised": metrics.get(
                "use_representatives_for_supervised",
                config.get("use_representatives_for_supervised"),
            ),
            "target": target,
            "backend_metrics": metrics.get("backend"),
            "backend_effective": profiling.get("backend_effective"),
            "selected_k": clustering.get("selected_k", config.get("selected_k")),
            "n_selected_features": selected.get("n_selected_features"),
            "train_n_samples_original": target_metrics.get("train_n_samples_original"),
            "train_n_samples_used": target_metrics.get("train_n_samples_used"),
            "target_compression_ratio": target_metrics.get("compression_ratio"),
            "representatives_n_original_train": representatives.get("n_original_train"),
            "representatives_n_train": representatives.get("n_representative_train"),
            "representatives_compression_ratio": representatives.get("compression_ratio"),
            "test_accuracy": test.get("accuracy"),
            "test_macro_precision": test.get("macro_precision"),
            "test_macro_recall": test.get("macro_recall"),
            "test_macro_f1": test.get("macro_f1"),
            "test_weighted_f1": test.get("weighted_f1"),
            "test_roc_auc": test.get("roc_auc"),
            "test_pr_auc": test.get("pr_auc"),
            "test_fpr_at_tpr_95": test.get("fpr_at_tpr_95"),
            "test_threshold_at_tpr_95": test.get("threshold_at_tpr_95"),
            "test_positive_label": test.get("positive_label"),
            "val_macro_f1": val.get("macro_f1"),
            "val_pr_auc": val.get("pr_auc"),
            "cluster_silhouette_train": clustering.get("silhouette_train"),
            "cluster_davies_bouldin_train": clustering.get("davies_bouldin_train"),
            "cluster_calinski_harabasz_train": clustering.get("calinski_harabasz_train"),
            "peak_ram_mb": profiling.get("peak_ram_mb"),
            "gpu_vram_current_mb": profiling.get("gpu_vram_current_mb"),
            "total_top_level_seconds": total_top_level_seconds(profiling),
            "supervised_target_seconds": stage_seconds(profiling, f"supervised_{target}"),
        }
        for stage in TOP_LEVEL_STAGE_TIMES:
            row[f"{stage}_seconds"] = stage_seconds(profiling, stage)
            row[f"{stage}_gpu_vram_after_mb"] = stage_vram_after(profiling, stage)
        rows.append(row)
    return rows


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path)


def stage_seconds(profiling: dict[str, Any], stage: str) -> float | None:
    record = profiling.get("stages", {}).get(stage)
    if not record:
        return None
    return record.get("seconds")


def stage_vram_after(profiling: dict[str, Any], stage: str) -> float | None:
    record = profiling.get("stages", {}).get(stage)
    if not record:
        return None
    return record.get("gpu_vram_after_mb")


def total_top_level_seconds(profiling: dict[str, Any]) -> float | None:
    values = [stage_seconds(profiling, stage) for stage in TOP_LEVEL_STAGE_TIMES]
    values = [value for value in values if value is not None]
    return float(sum(values)) if values else None


if __name__ == "__main__":
    main()
