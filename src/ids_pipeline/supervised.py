"""RandomForest supervised baselines for multiple targets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder

from .backend import resolve_backend, to_gpu_array, to_numpy_array
from .config import PipelineConfig
from .leakage_checks import check_for_leakage_columns
from .profiling import Profiler
from .utils import ensure_dir, write_json


def train_random_forest(
    X: dict[str, Any],
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    feature_cols: list[str],
    config: PipelineConfig,
    output_dir: str | Path,
    *,
    cluster_labels: dict[str, np.ndarray] | None = None,
    representatives: dict[str, Any] | None = None,
    profiler: Profiler | None = None,
) -> dict[str, Any]:
    check_for_leakage_columns(
        df,
        feature_cols,
        feature_policy=config.feature_policy,
        label_col=config.label_col,
        type_col=config.type_col,
        context="RandomForest fit",
    )

    backend = resolve_backend(config.compute_backend)
    output_dir = Path(output_dir)
    artifact_dir = ensure_dir(output_dir / "artifacts")

    target_metrics: dict[str, Any] = {}
    for target in config.targets:
        timer_name = f"supervised_{target}"
        if profiler is None:
            metrics = _train_one_target(
                target,
                X,
                df,
                splits,
                config,
                output_dir,
                artifact_dir,
                backend,
                cluster_labels,
                representatives,
            )
        else:
            with profiler.track(timer_name):
                metrics = _train_one_target(
                    target,
                    X,
                    df,
                    splits,
                    config,
                    output_dir,
                    artifact_dir,
                    backend,
                    cluster_labels,
                    representatives,
                )
        target_metrics[target] = metrics

    payload: dict[str, Any] = {
        "backend": backend,
        "use_representatives_for_supervised": bool(config.use_representatives_for_supervised),
        "representative_strategy": config.representative_strategy,
        "targets": target_metrics,
    }

    # Backward-compatible aliases for the original single-label debug pipeline.
    if "label" in target_metrics:
        payload["target"] = config.label_col
        payload["val"] = target_metrics["label"]["val"]
        payload["test"] = target_metrics["label"]["test"]

    write_json(output_dir / "metrics_supervised.json", payload)
    return payload


def _train_one_target(
    target: str,
    X: dict[str, Any],
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    config: PipelineConfig,
    output_dir: Path,
    artifact_dir: Path,
    backend: str,
    cluster_labels: dict[str, np.ndarray] | None,
    representatives: dict[str, Any] | None,
) -> dict[str, Any]:
    y = _target_arrays(target, df, splits, config, cluster_labels)
    X_train = X["train"]
    y_train = y["train"]

    train_n_original = int(len(y_train))
    if config.use_representatives_for_supervised:
        if representatives is None:
            raise ValueError("use_representatives_for_supervised=true but representatives were not provided")
        X_train = representatives["X"]
        y_train = np.asarray(representatives["y"][target])

    train_n_used = int(len(y_train))
    compression_ratio = float(train_n_used / train_n_original) if train_n_original else None

    model_bundle, X_eval = _fit_model(X_train, y_train, X, config, backend)
    joblib.dump(model_bundle, artifact_dir / f"random_forest_{target}.joblib")

    val_metrics = _evaluate_split(model_bundle, X_eval["val"], y["val"], split="val", backend=backend)
    test_metrics = _evaluate_split(model_bundle, X_eval["test"], y["test"], split="test", backend=backend)
    y_pred_test = _predict_decoded(model_bundle, X_eval["test"], backend=backend)

    _write_report_and_confusion(output_dir, target, y["test"], y_pred_test)
    if target == "label":
        (output_dir / "classification_report.txt").write_text(
            classification_report(_as_str(y["test"]), y_pred_test, zero_division=0),
            encoding="utf-8",
        )
        _confusion_dataframe(y["test"], y_pred_test).to_csv(output_dir / "confusion_matrix.csv")

    return {
        "train_n_samples_original": train_n_original,
        "train_n_samples_used": train_n_used,
        "compression_ratio": compression_ratio,
        "val": val_metrics,
        "test": test_metrics,
    }


def _target_arrays(
    target: str,
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    config: PipelineConfig,
    cluster_labels: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    if target == "label":
        source = {split: df.iloc[idx][config.label_col].to_numpy() for split, idx in splits.items()}
    elif target == "type":
        if config.type_col not in df.columns:
            raise ValueError(f"Target 'type' requested but {config.type_col!r} is not in dataframe")
        source = {split: df.iloc[idx][config.type_col].to_numpy() for split, idx in splits.items()}
    elif target == "cluster_id":
        if cluster_labels is None:
            raise ValueError("Target 'cluster_id' requested but cluster labels were not provided")
        source = {split: np.asarray(cluster_labels[split]) for split in ["train", "val", "test"]}
    else:
        raise ValueError("targets must contain only: label, type, cluster_id")
    return {split: _as_str(values) for split, values in source.items()}


def _fit_model(
    X_train: Any,
    y_train: np.ndarray,
    X_eval_source: dict[str, Any],
    config: PipelineConfig,
    backend: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(_as_str(y_train)).astype(np.int32)

    if backend == "gpu":
        model, X_eval = _fit_gpu_random_forest(X_train, y_encoded, X_eval_source, config)
        model_name = "cuML RandomForestClassifier"
    else:
        model = RandomForestClassifier(
            n_estimators=config.random_forest_estimators,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
            class_weight="balanced_subsample",
        )
        model.fit(X_train, y_encoded)
        X_eval = X_eval_source
        model_name = "RandomForestClassifier"

    return {"model": model, "label_encoder": encoder, "model_name": model_name}, X_eval


def _fit_gpu_random_forest(
    X_train: Any,
    y_train_encoded: np.ndarray,
    X_eval_source: dict[str, Any],
    config: PipelineConfig,
) -> tuple[Any, dict[str, Any]]:
    import cupy as cp
    from cuml.ensemble import RandomForestClassifier as CuMLRandomForestClassifier

    X_gpu = {split: to_gpu_array(X_split) for split, X_split in X_eval_source.items()}
    X_train_gpu = to_gpu_array(X_train)
    y_gpu = cp.asarray(y_train_encoded, dtype=cp.int32)
    model = CuMLRandomForestClassifier(
        n_estimators=config.random_forest_estimators,
        random_state=config.random_state,
        n_streams=max(1, min(int(config.n_jobs), 16)),
    )
    model.fit(X_train_gpu, y_gpu)
    return model, X_gpu


def _predict_encoded(model_bundle: dict[str, Any], X: Any, *, backend: str) -> np.ndarray:
    model = model_bundle["model"]
    pred = to_numpy_array(model.predict(X)) if backend == "gpu" else model.predict(X)
    return np.asarray(pred, dtype=np.int32)


def _predict_decoded(model_bundle: dict[str, Any], X: Any, *, backend: str) -> np.ndarray:
    encoder: LabelEncoder = model_bundle["label_encoder"]
    encoded = _predict_encoded(model_bundle, X, backend=backend)
    return encoder.inverse_transform(encoded)


def _predict_proba(model_bundle: dict[str, Any], X: Any, *, backend: str) -> np.ndarray | None:
    model = model_bundle["model"]
    if not hasattr(model, "predict_proba"):
        return None
    try:
        proba = to_numpy_array(model.predict_proba(X)) if backend == "gpu" else model.predict_proba(X)
        return np.asarray(proba)
    except Exception:
        return None


def _model_classes_encoded(model_bundle: dict[str, Any], y_true: np.ndarray, *, backend: str) -> list[int]:
    model = model_bundle["model"]
    if hasattr(model, "classes_"):
        classes = to_numpy_array(model.classes_) if backend == "gpu" else model.classes_
        return [int(c) for c in np.asarray(classes).tolist()]
    encoder: LabelEncoder = model_bundle["label_encoder"]
    return list(range(len(encoder.classes_)))


def _evaluate_split(
    model_bundle: dict[str, Any],
    X: Any,
    y_true: np.ndarray,
    *,
    split: str,
    backend: str,
) -> dict[str, Any]:
    y_true = _as_str(y_true)
    y_pred = _predict_decoded(model_bundle, X, backend=backend)
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    metrics: dict[str, Any] = {
        "split": split,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
    }
    metrics.update(_binary_metrics(model_bundle, X, y_true, backend=backend))
    return metrics


def _binary_metrics(
    model_bundle: dict[str, Any],
    X: Any,
    y_true: np.ndarray,
    *,
    backend: str,
) -> dict[str, Any]:
    encoder: LabelEncoder = model_bundle["label_encoder"]
    classes = list(encoder.classes_)
    payload = {
        "roc_auc": None,
        "pr_auc": None,
        "fpr_at_tpr_95": None,
        "threshold_at_tpr_95": None,
        "positive_label": None,
    }
    if len(np.unique(y_true)) != 2 or len(classes) != 2:
        return payload

    positive_label = _choose_positive_label(classes)
    proba = _predict_proba(model_bundle, X, backend=backend)
    if proba is None:
        return payload

    try:
        positive_encoded = int(encoder.transform([positive_label])[0])
        model_classes = _model_classes_encoded(model_bundle, y_true, backend=backend)
        positive_col = model_classes.index(positive_encoded)
        scores = proba[:, positive_col]
        y_binary = (y_true == positive_label).astype(int)
        payload["roc_auc"] = float(roc_auc_score(y_binary, scores))
        payload["pr_auc"] = float(average_precision_score(y_binary, scores))
        fpr_payload = compute_fpr_at_tpr(y_binary, scores, target_tpr=0.95)
        if fpr_payload is not None:
            payload.update(fpr_payload)
        payload["positive_label"] = str(positive_label)
    except Exception:
        return payload
    return payload


def compute_fpr_at_tpr(
    y_true_binary: np.ndarray,
    y_score: np.ndarray,
    target_tpr: float = 0.95,
) -> dict[str, float] | None:
    y_true_binary = np.asarray(y_true_binary).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true_binary)) != 2:
        return None
    try:
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_score)
    except Exception:
        return None
    candidates = np.flatnonzero(tpr >= target_tpr)
    if len(candidates) == 0:
        return None
    best = candidates[np.argmin(fpr[candidates])]
    return {
        "fpr_at_tpr_95": float(fpr[best]),
        "threshold_at_tpr_95": float(thresholds[best]),
    }


def _choose_positive_label(classes: list[str]) -> str:
    if "1" in classes:
        return "1"
    lowered = {str(c).lower(): c for c in classes}
    for key in ["attack", "malicious", "anomaly", "true"]:
        if key in lowered:
            return str(lowered[key])
    return str(classes[-1])


def _write_report_and_confusion(
    output_dir: Path,
    target: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    report_text = classification_report(_as_str(y_true), _as_str(y_pred), zero_division=0)
    (output_dir / f"classification_report_{target}.txt").write_text(report_text, encoding="utf-8")
    _confusion_dataframe(y_true, y_pred).to_csv(output_dir / f"confusion_matrix_{target}.csv")


def _confusion_dataframe(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    y_true = _as_str(y_true)
    y_pred = _as_str(y_pred)
    labels = sorted(pd.Series(np.concatenate([y_true, y_pred])).dropna().astype(str).unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"true_{label}" for label in labels], columns=[f"pred_{label}" for label in labels])


def _as_str(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).astype(str).to_numpy()
