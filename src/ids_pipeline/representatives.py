"""Cluster representative selection for compressed supervised training."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backend import to_numpy_array
from .config import PipelineConfig
from .utils import ensure_dir, write_json


def build_cluster_representatives(
    X_train: Any,
    y_train_dict: dict[str, np.ndarray],
    cluster_ids_train: np.ndarray,
    cluster_distances_train: np.ndarray | None,
    config: PipelineConfig,
    backend: str,
    output_dir: str | Path,
    *,
    train_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    if config.selection_budget is not None:
        raise ValueError('Legacy representatives cannot implement selection_budget; use the real-row budget contract')
    strategy = config.representative_strategy
    if strategy == "full":
        result = _full(X_train, y_train_dict, cluster_ids_train, train_indices)
    elif strategy == "centroid":
        result = _centroid(X_train, y_train_dict, cluster_ids_train, backend)
    elif strategy == "medoid":
        result = _from_indices(
            X_train,
            y_train_dict,
            _medoid_indices(X_train, cluster_ids_train, cluster_distances_train),
            cluster_ids_train,
            train_indices,
        )
    elif strategy == "stratified_sample":
        result = _from_indices(
            X_train,
            y_train_dict,
            _stratified_sample_indices(y_train_dict, cluster_ids_train, config),
            cluster_ids_train,
            train_indices,
        )
    elif strategy == "boundary_sample":
        result = _from_indices(
            X_train,
            y_train_dict,
            _boundary_indices(cluster_ids_train, cluster_distances_train, config),
            cluster_ids_train,
            train_indices,
        )
    elif strategy == "mixed":
        result = _from_indices(
            X_train,
            y_train_dict,
            _mixed_indices(X_train, y_train_dict, cluster_ids_train, cluster_distances_train, config),
            cluster_ids_train,
            train_indices,
        )
    else:
        raise ValueError(
            "representative_strategy must be one of: full, centroid, medoid, "
            "stratified_sample, boundary_sample, mixed"
        )

    metadata = _metadata(result, cluster_ids_train, config, backend)
    result["metadata"] = metadata
    _save_representatives(output_dir, result, cluster_ids_train, train_indices)
    return result


def _full(
    X_train: Any,
    y_train_dict: dict[str, np.ndarray],
    cluster_ids_train: np.ndarray,
    train_indices: np.ndarray | None,
) -> dict[str, Any]:
    indices = np.arange(len(cluster_ids_train), dtype=int)
    return {
        "X": X_train,
        "y": {target: np.asarray(values) for target, values in y_train_dict.items()},
        "sample_weight": None,
        "representative_indices": indices,
        "row_indices": None if train_indices is None else np.asarray(train_indices)[indices],
        "cluster_ids": np.asarray(cluster_ids_train),
        "kind": "real_points",
    }


def _centroid(
    X_train: Any,
    y_train_dict: dict[str, np.ndarray],
    cluster_ids_train: np.ndarray,
    backend: str,
) -> dict[str, Any]:
    clusters = np.sort(np.unique(cluster_ids_train))
    if backend == "gpu":
        import cupy as cp

        X_gpu = cp.asarray(X_train)
        rows = []
        for cluster in clusters:
            mask = cp.asarray(cluster_ids_train == cluster)
            rows.append(cp.mean(X_gpu[mask], axis=0))
        X_rep = cp.stack(rows, axis=0)
    else:
        X_arr = _as_numpy_matrix(X_train)
        X_rep = np.vstack([X_arr[cluster_ids_train == cluster].mean(axis=0) for cluster in clusters])

    y_rep = {}
    for target, values in y_train_dict.items():
        values = np.asarray(values)
        y_rep[target] = np.asarray([_majority(values[cluster_ids_train == cluster]) for cluster in clusters])

    return {
        "X": X_rep,
        "y": y_rep,
        "sample_weight": np.asarray([np.sum(cluster_ids_train == cluster) for cluster in clusters], dtype=float),
        "representative_indices": None,
        "row_indices": None,
        "cluster_ids": clusters,
        "kind": "centroids",
    }


def _from_indices(
    X_train: Any,
    y_train_dict: dict[str, np.ndarray],
    indices: np.ndarray,
    cluster_ids_train: np.ndarray,
    train_indices: np.ndarray | None,
) -> dict[str, Any]:
    indices = np.asarray(sorted(set(int(i) for i in indices)), dtype=int)
    return {
        "X": _slice_matrix(X_train, indices),
        "y": {target: np.asarray(values)[indices] for target, values in y_train_dict.items()},
        "sample_weight": None,
        "representative_indices": indices,
        "row_indices": None if train_indices is None else np.asarray(train_indices)[indices],
        "cluster_ids": np.asarray(cluster_ids_train)[indices],
        "kind": "real_points",
    }


def _medoid_indices(
    X_train: Any,
    cluster_ids_train: np.ndarray,
    cluster_distances_train: np.ndarray | None,
) -> np.ndarray:
    distances = _distances_or_to_centroids(X_train, cluster_ids_train, cluster_distances_train)
    selected = []
    for cluster in np.sort(np.unique(cluster_ids_train)):
        idx = np.flatnonzero(cluster_ids_train == cluster)
        selected.append(idx[np.argmin(distances[idx])])
    return np.asarray(selected, dtype=int)


def _stratified_sample_indices(
    y_train_dict: dict[str, np.ndarray],
    cluster_ids_train: np.ndarray,
    config: PipelineConfig,
) -> np.ndarray:
    rng = np.random.default_rng(config.random_state)
    primary_target = "label" if "label" in y_train_dict else next(iter(y_train_dict))
    y = np.asarray(y_train_dict[primary_target]).astype(str)
    selected: list[int] = []

    for cluster in np.sort(np.unique(cluster_ids_train)):
        cluster_idx = np.flatnonzero(cluster_ids_train == cluster)
        budget = min(config.representatives_per_cluster, len(cluster_idx))
        if budget <= 0:
            continue
        by_class = {klass: cluster_idx[y[cluster_idx] == klass] for klass in np.unique(y[cluster_idx])}
        picks: list[int] = []
        for klass_idx in by_class.values():
            n_class = max(1, int(round(budget * len(klass_idx) / len(cluster_idx))))
            n_class = min(n_class, len(klass_idx), max(0, budget - len(picks)))
            if n_class:
                picks.extend(rng.choice(klass_idx, size=n_class, replace=False).tolist())
        if len(picks) < budget:
            remaining = np.setdiff1d(cluster_idx, np.asarray(picks, dtype=int), assume_unique=False)
            fill_n = min(budget - len(picks), len(remaining))
            if fill_n:
                picks.extend(rng.choice(remaining, size=fill_n, replace=False).tolist())
        selected.extend(picks[:budget])
    return np.asarray(selected, dtype=int)


def _boundary_indices(
    cluster_ids_train: np.ndarray,
    cluster_distances_train: np.ndarray | None,
    config: PipelineConfig,
) -> np.ndarray:
    if cluster_distances_train is None:
        raise ValueError("boundary_sample requires cluster_distances_train")
    distances = np.asarray(cluster_distances_train)
    selected = []
    for cluster in np.sort(np.unique(cluster_ids_train)):
        idx = np.flatnonzero(cluster_ids_train == cluster)
        take = min(config.boundary_per_cluster, len(idx))
        if take:
            selected.extend(idx[np.argsort(distances[idx])[-take:]].tolist())
    return np.asarray(selected, dtype=int)


def _mixed_indices(
    X_train: Any,
    y_train_dict: dict[str, np.ndarray],
    cluster_ids_train: np.ndarray,
    cluster_distances_train: np.ndarray | None,
    config: PipelineConfig,
) -> np.ndarray:
    pieces = [
        _medoid_indices(X_train, cluster_ids_train, cluster_distances_train),
        _boundary_indices(cluster_ids_train, cluster_distances_train, config),
        _stratified_sample_indices(y_train_dict, cluster_ids_train, config),
    ]
    return np.asarray(sorted(set(np.concatenate(pieces).astype(int).tolist())), dtype=int)


def _distances_or_to_centroids(
    X_train: Any,
    cluster_ids_train: np.ndarray,
    cluster_distances_train: np.ndarray | None,
) -> np.ndarray:
    if cluster_distances_train is not None:
        return np.asarray(cluster_distances_train)
    X_arr = _as_numpy_matrix(X_train)
    distances = np.zeros(len(cluster_ids_train), dtype=float)
    for cluster in np.unique(cluster_ids_train):
        idx = np.flatnonzero(cluster_ids_train == cluster)
        centroid = X_arr[idx].mean(axis=0)
        distances[idx] = np.linalg.norm(X_arr[idx] - centroid, axis=1)
    return distances


def _majority(values: np.ndarray) -> object:
    counts = Counter(pd.Series(values).astype(str).tolist())
    return counts.most_common(1)[0][0]


def _slice_matrix(X: Any, indices: np.ndarray) -> Any:
    return X[indices]


def _as_numpy_matrix(X: Any) -> np.ndarray:
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(to_numpy_array(X))


def _metadata(
    result: dict[str, Any],
    cluster_ids_train: np.ndarray,
    config: PipelineConfig,
    backend: str,
) -> dict[str, object]:
    n_original = int(len(cluster_ids_train))
    n_rep = int(_matrix_n_rows(result["X"]))
    return {
        "strategy": config.representative_strategy,
        "n_original_train": n_original,
        "n_representative_train": n_rep,
        "compression_ratio": float(n_rep / n_original) if n_original else None,
        "n_clusters": int(len(np.unique(cluster_ids_train))),
        "representatives_per_cluster": int(config.representatives_per_cluster),
        "boundary_per_cluster": int(config.boundary_per_cluster),
        "backend": backend,
    }


def _matrix_n_rows(X: Any) -> int:
    return int(X.shape[0])


def _save_representatives(
    output_dir: str | Path,
    result: dict[str, Any],
    cluster_ids_train: np.ndarray,
    train_indices: np.ndarray | None,
) -> None:
    output_dir = Path(output_dir)
    write_json(output_dir / "representatives_metadata.json", result["metadata"])

    indices = result.get("representative_indices")
    if indices is not None and result["metadata"]["strategy"] != "full":
        frame = pd.DataFrame(
            {
                "train_position": indices,
                "row_index": result.get("row_indices") if result.get("row_indices") is not None else indices,
                "cluster_id": result["cluster_ids"],
            }
        )
        frame.to_csv(output_dir / "representatives_indices.csv", index=False)

    summary = _cluster_summary(result, cluster_ids_train, train_indices)
    summary.to_csv(output_dir / "representatives_cluster_summary.csv", index=False)


def _cluster_summary(
    result: dict[str, Any],
    cluster_ids_train: np.ndarray,
    train_indices: np.ndarray | None,
) -> pd.DataFrame:
    original = pd.Series(cluster_ids_train, name="cluster_id").value_counts().sort_index()
    represented = pd.Series(result["cluster_ids"], name="cluster_id").value_counts().sort_index()
    frame = pd.DataFrame(
        {
            "cluster_id": original.index.astype(int),
            "original_n": original.values.astype(int),
            "representative_n": [int(represented.get(cluster, 0)) for cluster in original.index],
        }
    )
    frame["cluster_compression_ratio"] = frame["representative_n"] / frame["original_n"]
    return frame
