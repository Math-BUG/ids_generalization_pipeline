"""MiniBatchKMeans clustering and metrics for the minimal pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
    v_measure_score,
)

from .backend import resolve_backend, to_gpu_array, to_numpy_array
from .config import PipelineConfig
from .leakage_checks import check_for_leakage_columns
from .utils import ensure_dir, write_json


def fit_predict_clustering(
    X: dict[str, Any],
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    feature_cols: list[str],
    config: PipelineConfig,
    output_dir: str | Path,
) -> dict[str, Any]:
    check_for_leakage_columns(
        df,
        feature_cols,
        feature_policy=config.feature_policy,
        label_col=config.label_col,
        type_col=config.type_col,
        context="MiniBatchKMeans fit",
    )

    output_dir = Path(output_dir)
    artifact_dir = ensure_dir(output_dir / "artifacts")
    backend = resolve_backend(config.compute_backend)

    if backend == "gpu":
        clusterer, labels, distances = _fit_gpu_kmeans(X, config)
        joblib.dump(clusterer, artifact_dir / "cuml_kmeans.joblib")
    else:
        clusterer, labels, distances = _fit_cpu_minibatch_kmeans(X, config)
        joblib.dump(clusterer, artifact_dir / "minibatch_kmeans.joblib")

    if config.export_cluster_assignments:
        assignment_rows = []
        for split, idx in splits.items():
            assignment_rows.append(
                pd.DataFrame(
                    {
                        "split": split,
                        "row_index": idx,
                        "cluster_id": labels[split],
                        "cluster_distance": distances[split],
                    }
                )
            )
        assignments = pd.concat(assignment_rows, ignore_index=True)
        assignments.to_csv(output_dir / "cluster_assignments.csv", index=False)

    metrics = compute_clustering_metrics(X, df, splits, labels, clusterer, config, backend=backend)
    write_json(output_dir / "metrics_clustering.json", metrics)
    return {"clusterer": clusterer, "labels": labels, "distances": distances, "metrics": metrics}


def compute_clustering_metrics(
    X: dict[str, Any],
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    clusterer: MiniBatchKMeans,
    config: PipelineConfig,
    *,
    backend: str,
) -> dict[str, Any]:
    X_train = _as_array(X["train"])
    train_labels = labels["train"]
    metrics: dict[str, Any] = {
        "algorithm": "cuML KMeans" if backend == "gpu" else "MiniBatchKMeans",
        "backend": backend,
        "selected_k": int(config.selected_k),
        "fit_split": "train",
        "fit_n_samples": int(X_train.shape[0]),
        "n_clusters_observed_train": int(len(np.unique(train_labels))),
        "inertia": float(getattr(clusterer, "inertia_", np.nan)),
        "silhouette_train": None,
        "davies_bouldin_train": None,
        "calinski_harabasz_train": None,
    }

    if 1 < len(np.unique(train_labels)) < len(train_labels):
        eval_X, eval_labels = _sample_for_silhouette(
            X_train,
            train_labels,
            sample_size=config.silhouette_sample_size,
            random_state=config.random_state,
        )
        if 1 < len(np.unique(eval_labels)) < len(eval_labels):
            metrics["silhouette_train"] = float(silhouette_score(eval_X, eval_labels))
        metrics["davies_bouldin_train"] = float(davies_bouldin_score(X_train, train_labels))
        metrics["calinski_harabasz_train"] = float(calinski_harabasz_score(X_train, train_labels))

    for split, idx in splits.items():
        split_df = df.iloc[idx]
        cluster_ids = labels[split]
        for target_col in [config.label_col, config.type_col]:
            if target_col not in split_df.columns:
                continue
            y = split_df[target_col].astype(str).to_numpy()
            prefix = f"{split}_{target_col}"
            metrics[f"{prefix}_ami"] = float(adjusted_mutual_info_score(y, cluster_ids))
            metrics[f"{prefix}_v_measure"] = float(v_measure_score(y, cluster_ids))
            metrics[f"{prefix}_purity"] = float(cluster_purity(y, cluster_ids))
    return metrics


def _fit_cpu_minibatch_kmeans(
    X: dict[str, Any],
    config: PipelineConfig,
) -> tuple[MiniBatchKMeans, dict[str, np.ndarray], dict[str, np.ndarray]]:
    clusterer = MiniBatchKMeans(
        n_clusters=config.selected_k,
        batch_size=config.cluster_batch_size,
        random_state=config.random_state,
        n_init=config.cluster_n_init,
        max_iter=config.cluster_max_iter,
    )
    labels = {"train": clusterer.fit_predict(X["train"])}
    labels["val"] = clusterer.predict(X["val"])
    labels["test"] = clusterer.predict(X["test"])
    distances = {split: np.min(clusterer.transform(X_split), axis=1) for split, X_split in X.items()}
    return clusterer, labels, distances


def _fit_gpu_kmeans(
    X: dict[str, Any],
    config: PipelineConfig,
) -> tuple[Any, dict[str, np.ndarray], dict[str, np.ndarray]]:
    import cupy as cp
    from cuml.cluster import KMeans

    X_gpu = {split: to_gpu_array(X_split) for split, X_split in X.items()}
    clusterer = KMeans(
        n_clusters=config.selected_k,
        max_iter=config.cluster_max_iter,
        random_state=config.random_state,
    )
    train_labels = clusterer.fit_predict(X_gpu["train"])
    labels = {"train": to_numpy_array(train_labels).astype(int)}
    labels["val"] = to_numpy_array(clusterer.predict(X_gpu["val"])).astype(int)
    labels["test"] = to_numpy_array(clusterer.predict(X_gpu["test"])).astype(int)

    centers = cp.asarray(clusterer.cluster_centers_)
    distances = {}
    for split, X_split in X_gpu.items():
        assigned_centers = centers[cp.asarray(labels[split])]
        distances[split] = to_numpy_array(cp.linalg.norm(X_split - assigned_centers, axis=1))
    return clusterer, labels, distances


def cluster_purity(y_true: np.ndarray, cluster_ids: np.ndarray) -> float:
    frame = pd.DataFrame({"cluster": cluster_ids, "target": y_true})
    if frame.empty:
        return float("nan")
    correct = frame.groupby("cluster")["target"].agg(lambda s: s.value_counts().iloc[0]).sum()
    return float(correct / len(frame))


def _as_array(X: Any) -> np.ndarray:
    return X.toarray() if hasattr(X, "toarray") else np.asarray(to_numpy_array(X))


def _sample_for_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    sample_size: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(labels) <= sample_size:
        return X, labels
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(labels), size=sample_size, replace=False)
    return X[idx], labels[idx]
