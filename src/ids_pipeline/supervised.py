"""RandomForest supervised baseline for the minimal pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .backend import resolve_backend, to_gpu_array, to_numpy_array
from .config import PipelineConfig
from .leakage_checks import check_for_leakage_columns
from .utils import ensure_dir, write_json


def train_random_forest(
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
        context="RandomForest fit",
    )

    y_train = df.iloc[splits["train"]][config.label_col].to_numpy()
    y_val = df.iloc[splits["val"]][config.label_col].to_numpy()
    y_test = df.iloc[splits["test"]][config.label_col].to_numpy()
    backend = resolve_backend(config.compute_backend)

    output_dir = Path(output_dir)
    artifact_dir = ensure_dir(output_dir / "artifacts")
    if backend == "gpu":
        model, X_eval = _fit_gpu_random_forest(X, y_train, config)
        joblib.dump(model, artifact_dir / "cuml_random_forest.joblib")
        model_name = "cuML RandomForestClassifier"
    else:
        model = RandomForestClassifier(
            n_estimators=config.random_forest_estimators,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
            class_weight="balanced_subsample",
        )
        model.fit(X["train"], y_train)
        X_eval = X
        joblib.dump(model, artifact_dir / "random_forest.joblib")
        model_name = "RandomForestClassifier"

    val_metrics = _evaluate_split(model, X_eval["val"], y_val, split="val", backend=backend)
    test_metrics = _evaluate_split(model, X_eval["test"], y_test, split="test", backend=backend)
    metrics = {
        "model": model_name,
        "backend": backend,
        "target": config.label_col,
        "train_n_samples": int(len(y_train)),
        "val": val_metrics,
        "test": test_metrics,
    }
    write_json(output_dir / "metrics_supervised.json", metrics)

    y_pred_test = _predict(model, X_eval["test"], backend=backend)
    report_text = classification_report(y_test, y_pred_test, zero_division=0)
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    labels = sorted(pd.Series(np.concatenate([y_test, y_pred_test])).dropna().astype(str).unique())
    cm = confusion_matrix(y_test.astype(str), y_pred_test.astype(str), labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{label}" for label in labels], columns=[f"pred_{label}" for label in labels])
    cm_df.to_csv(output_dir / "confusion_matrix.csv")
    return metrics


def _fit_gpu_random_forest(
    X: dict[str, Any],
    y_train: np.ndarray,
    config: PipelineConfig,
) -> tuple[Any, dict[str, Any]]:
    import cupy as cp
    from cuml.ensemble import RandomForestClassifier as CuMLRandomForestClassifier

    X_gpu = {split: to_gpu_array(X_split) for split, X_split in X.items()}
    y_gpu = cp.asarray(y_train)
    model = CuMLRandomForestClassifier(
        n_estimators=config.random_forest_estimators,
        random_state=config.random_state,
        n_streams=max(1, min(int(config.n_jobs), 16)),
    )
    model.fit(X_gpu["train"], y_gpu)
    return model, X_gpu


def _predict(model: Any, X: Any, *, backend: str) -> np.ndarray:
    return to_numpy_array(model.predict(X)) if backend == "gpu" else model.predict(X)


def _predict_proba(model: Any, X: Any, *, backend: str) -> np.ndarray:
    return to_numpy_array(model.predict_proba(X)) if backend == "gpu" else model.predict_proba(X)


def _model_classes(model: Any, y_true: np.ndarray, *, backend: str) -> list[Any]:
    if hasattr(model, "classes_"):
        return list(to_numpy_array(model.classes_)) if backend == "gpu" else list(model.classes_)
    return sorted(np.unique(y_true).tolist())


def _evaluate_split(model: Any, X: Any, y_true: np.ndarray, *, split: str, backend: str) -> dict[str, Any]:
    y_pred = _predict(model, X, backend=backend)
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
        "roc_auc": None,
    }
    if len(np.unique(y_true)) == 2 and hasattr(model, "predict_proba"):
        classes = _model_classes(model, y_true, backend=backend)
        positive_label = 1 if 1 in classes else classes[-1]
        scores = _predict_proba(model, X, backend=backend)[:, classes.index(positive_label)]
        metrics["roc_auc"] = float(roc_auc_score((y_true == positive_label).astype(int), scores))
        metrics["positive_label"] = str(positive_label)
    return metrics
