"""Leakage-safe preprocessing and dimensionality reduction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .backend import resolve_backend, to_numpy_array
from .config import PipelineConfig
from .dataset_schema import is_ton_iot, model_feature_types, normalize_dataset_schema
from .leakage_checks import check_for_leakage_columns
from .schema import DEFAULT_LOG1P_COLUMNS, TON_IOT_SCHEMA_VERSION, TON_IOT_SCHEMA
from .nominal_encoding import ConnStateOneHot
from .utils import ensure_dir, write_json

LOGGER = logging.getLogger(__name__)


@dataclass
class PreprocessingBundle:
    feature_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    log_cols: list[str]
    preprocessor: Any
    svd: Any | None
    backend: str = "cpu"


def fit_transform_preprocessing(
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    feature_cols: list[str],
    config: PipelineConfig,
    output_dir: str | Path,
) -> tuple[dict[str, Any], PreprocessingBundle]:
    check_for_leakage_columns(
        df,
        feature_cols,
        feature_policy=config.feature_policy,
        label_col=config.label_col,
        type_col=config.type_col,
        context="preprocessing fit",
    )

    # Validate direct callers too. This is stateless; all fitted statistics remain
    # below, after the split. Never use validation/test values to infer model roles.
    features, schema_report = normalize_dataset_schema(
        df[feature_cols], report_path=Path(output_dir) / "preprocessing_schema_report.json",
    )
    numeric_cols, categorical_cols = model_feature_types(
        features, feature_cols, strict=is_ton_iot(df.columns),
    )
    X_train_df = features.iloc[splits["train"]].copy()
    X_val_df = features.iloc[splits["val"]].copy()
    X_test_df = features.iloc[splits["test"]].copy()
    log_cols = _select_log_cols(numeric_cols, config.log1p_columns) if config.log1p_numeric else []
    if config.log1p_numeric and config.log1p_columns is None:
        LOGGER.info("Using explicit log1p column list; legacy log1p_patterns is ignored")

    X_train_df = apply_log1p(X_train_df, log_cols)
    X_val_df = apply_log1p(X_val_df, log_cols)
    X_test_df = apply_log1p(X_test_df, log_cols)

    backend = resolve_backend(config.compute_backend)
    if backend == "gpu":
        return _fit_transform_gpu_preprocessing(
            X_train_df,
            X_val_df,
            X_test_df,
            feature_cols,
            numeric_cols,
            categorical_cols,
            log_cols,
            config,
            output_dir,
        )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols, config.onehot_min_frequency)
    check_for_leakage_columns(
        df,
        feature_cols,
        feature_policy=config.feature_policy,
        label_col=config.label_col,
        type_col=config.type_col,
        context="encoder/scaler fit",
    )
    X_train = preprocessor.fit_transform(X_train_df)
    X_val = preprocessor.transform(X_val_df)
    X_test = preprocessor.transform(X_test_df)

    svd = None
    requested = int(config.svd_components or 0)
    max_components = max(0, min(X_train.shape[0] - 1, X_train.shape[1] - 1))
    n_components = min(requested, max_components)
    if n_components >= 1:
        check_for_leakage_columns(
            df,
            feature_cols,
            feature_policy=config.feature_policy,
            label_col=config.label_col,
            type_col=config.type_col,
            context="SVD fit",
        )
        svd = TruncatedSVD(n_components=n_components, random_state=config.random_state)
        X_train = svd.fit_transform(X_train)
        X_val = svd.transform(X_val)
        X_test = svd.transform(X_test)
        LOGGER.info("Fitted TruncatedSVD with n_components=%d", n_components)
    else:
        LOGGER.info("Skipping SVD because requested=%s and max_components=%s", requested, max_components)

    bundle = PreprocessingBundle(
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        log_cols=log_cols,
        preprocessor=preprocessor,
        svd=svd,
        backend="cpu",
    )
    artifact_dir = ensure_dir(Path(output_dir) / "artifacts")
    joblib.dump(bundle, artifact_dir / "preprocessing_bundle.joblib")
    write_json(
        artifact_dir / "preprocessing_metadata.json",
        {
            "schema_version": TON_IOT_SCHEMA_VERSION,
            "unregistered_columns": schema_report["unknown_columns"],
            "feature_cols": feature_cols,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "log_cols": log_cols,
            "conn_state_encoding": (preprocessor.named_transformers_["conn_state"].metadata()
                                    if "conn_state" in categorical_cols else None),
            "feature_transformations": describe_feature_transformations(feature_cols, log_cols),
            "reducer": None if svd is None else "TruncatedSVD",
            "svd_components_effective": None if svd is None else int(svd.n_components),
            "transformed_shapes": {k: list(v.shape) for k, v in {"train": X_train, "val": X_val, "test": X_test}.items()},
        },
    )
    return {"train": X_train, "val": X_val, "test": X_test}, bundle


def _fit_transform_gpu_preprocessing(
    X_train_df: pd.DataFrame,
    X_val_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    log_cols: list[str],
    config: PipelineConfig,
    output_dir: str | Path,
) -> tuple[dict[str, Any], PreprocessingBundle]:
    import cupy as cp

    parts_train = []
    parts_val = []
    parts_test = []
    gpu_metadata: dict[str, Any] = {"categorical_maps": {}, "numeric_stats": {}}

    if numeric_cols:
        train_num, val_num, test_num, numeric_stats = _gpu_numeric_arrays(
            X_train_df,
            X_val_df,
            X_test_df,
            numeric_cols,
        )
        parts_train.append(train_num)
        parts_val.append(val_num)
        parts_test.append(test_num)
        gpu_metadata["numeric_stats"] = numeric_stats

    ordinal_cols = [c for c in categorical_cols if c != "conn_state"]
    if ordinal_cols:
        train_cat, val_cat, test_cat, categorical_maps = _gpu_categorical_code_arrays(
            X_train_df,
            X_val_df,
            X_test_df,
            ordinal_cols,
            config.onehot_min_frequency,
            config.gpu_max_categories_per_col,
        )
        parts_train.append(train_cat)
        parts_val.append(val_cat)
        parts_test.append(test_cat)
        gpu_metadata["categorical_maps"] = categorical_maps
        gpu_metadata["categorical_output_columns"] = list(ordinal_cols)

    if "conn_state" in categorical_cols:
        encoder = ConnStateOneHot().fit(X_train_df[["conn_state"]])
        parts_train.append(encoder.transform_gpu(X_train_df[["conn_state"]]))
        parts_val.append(encoder.transform_gpu(X_val_df[["conn_state"]]))
        parts_test.append(encoder.transform_gpu(X_test_df[["conn_state"]]))
        gpu_metadata["conn_state_encoder"] = encoder
        gpu_metadata["conn_state_encoding"] = encoder.metadata()
        gpu_metadata.setdefault("categorical_output_columns", []).extend(encoder.get_feature_names_out().tolist())

    if not parts_train:
        raise ValueError("No feature columns available after applying FeaturePolicy.")

    X_train = cp.concatenate(parts_train, axis=1) if len(parts_train) > 1 else parts_train[0]
    X_val = cp.concatenate(parts_val, axis=1) if len(parts_val) > 1 else parts_val[0]
    X_test = cp.concatenate(parts_test, axis=1) if len(parts_test) > 1 else parts_test[0]

    reducer = None
    requested = int(config.svd_components or 0)
    max_components = max(0, min(X_train.shape[0] - 1, X_train.shape[1] - 1))
    n_components = min(requested, max_components)
    reducer_name = None
    if n_components >= 1:
        reducer, reducer_name = _make_gpu_reducer(n_components, config.random_state)
        X_train = cp.asarray(reducer.fit_transform(X_train), dtype=cp.float32)
        X_val = cp.asarray(reducer.transform(X_val), dtype=cp.float32)
        X_test = cp.asarray(reducer.transform(X_test), dtype=cp.float32)
        LOGGER.info("Fitted GPU %s with n_components=%d", reducer_name, n_components)
    else:
        LOGGER.info("Skipping GPU dimensionality reduction because requested=%s and max_components=%s", requested, max_components)

    bundle = PreprocessingBundle(
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        log_cols=log_cols,
        preprocessor=gpu_metadata,
        svd=reducer,
        backend="gpu",
    )
    artifact_dir = ensure_dir(Path(output_dir) / "artifacts")
    try:
        joblib.dump(bundle, artifact_dir / "preprocessing_bundle.joblib")
    except Exception as exc:
        LOGGER.warning("Could not serialize GPU preprocessing bundle with joblib: %s", exc)
    write_json(
        artifact_dir / "preprocessing_metadata.json",
        {
            "backend": "gpu",
            "schema_version": TON_IOT_SCHEMA_VERSION,
            "feature_cols": feature_cols,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "log_cols": log_cols,
            "onehot_min_frequency": config.onehot_min_frequency,
            "gpu_categorical_encoding": "one_hot" if categorical_cols == ["conn_state"] else "per_column",
            "legacy_ordinal_columns": ordinal_cols,
            "conn_state_encoding": gpu_metadata.get("conn_state_encoding"),
            "feature_transformations": describe_feature_transformations(feature_cols, log_cols),
            "gpu_max_categories_per_col": config.gpu_max_categories_per_col,
            "categorical_output_columns": gpu_metadata.get("categorical_output_columns", []),
            "reducer": reducer_name,
            "svd_components_effective": None if reducer is None else int(n_components),
            "transformed_shapes": {
                "train": list(X_train.shape),
                "val": list(X_val.shape),
                "test": list(X_test.shape),
            },
        },
    )
    return {"train": X_train, "val": X_val, "test": X_test}, bundle


def _gpu_numeric_arrays(
    X_train_df: pd.DataFrame,
    X_val_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    numeric_cols: list[str],
) -> tuple[Any, Any, Any, dict[str, Any]]:
    import cupy as cp

    train = X_train_df[numeric_cols].astype("float64")
    val = X_val_df[numeric_cols].astype("float64")
    test = X_test_df[numeric_cols].astype("float64")

    medians = train.median(numeric_only=True).fillna(0.0)
    train = train.fillna(medians)
    val = val.fillna(medians)
    test = test.fillna(medians)

    means = train.mean(numeric_only=True).fillna(0.0)
    stds = train.std(numeric_only=True).replace(0.0, 1.0).fillna(1.0)

    train_arr = cp.asarray(((train - means) / stds).to_numpy(dtype=np.float32), dtype=cp.float32)
    val_arr = cp.asarray(((val - means) / stds).to_numpy(dtype=np.float32), dtype=cp.float32)
    test_arr = cp.asarray(((test - means) / stds).to_numpy(dtype=np.float32), dtype=cp.float32)
    stats = {
        "median": {k: float(v) for k, v in medians.items()},
        "mean": {k: float(v) for k, v in means.items()},
        "std": {k: float(v) for k, v in stds.items()},
    }
    return train_arr, val_arr, test_arr, stats


def _gpu_categorical_code_arrays(
    X_train_df: pd.DataFrame,
    X_val_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    categorical_cols: list[str],
    onehot_min_frequency: int | None,
    max_categories_per_col: int,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    import cupy as cp

    if "conn_state" in categorical_cols:
        raise ValueError("conn_state must use ConnStateOneHot, never ordinal model coordinates")

    min_frequency = max(1, int(onehot_min_frequency or 1))
    max_categories = max(1, int(max_categories_per_col or 64))
    train_arrays = []
    val_arrays = []
    test_arrays = []
    maps: dict[str, Any] = {}

    for col in categorical_cols:
        train_s = _normalize_category_series(X_train_df[col])
        mode = _category_mode(train_s)
        filled_train = train_s.fillna(mode)
        counts = filled_train.value_counts(dropna=False)
        kept_index = counts[counts >= min_frequency].head(max_categories).index
        keep_values = [str(v) for v in kept_index]
        if not keep_values:
            keep_values = [str(mode)]
        code_map = {value: code + 1 for code, value in enumerate(keep_values)}
        train_codes = _category_codes(train_s, mode, code_map)
        val_codes = _category_codes(_normalize_category_series(X_val_df[col]), mode, code_map)
        test_codes = _category_codes(_normalize_category_series(X_test_df[col]), mode, code_map)

        mean = float(train_codes.mean())
        std = float(train_codes.std())
        if std == 0.0 or np.isnan(std):
            std = 1.0
        train_arrays.append(cp.asarray(((train_codes - mean) / std).to_numpy(dtype=np.float32))[:, None])
        val_arrays.append(cp.asarray(((val_codes - mean) / std).to_numpy(dtype=np.float32))[:, None])
        test_arrays.append(cp.asarray(((test_codes - mean) / std).to_numpy(dtype=np.float32))[:, None])
        maps[col] = {
            "mode": str(mode),
            "n_kept": int(len(code_map)),
            "min_frequency": int(min_frequency),
            "max_categories": int(max_categories),
            "unknown_code": 0,
            "mean": mean,
            "std": std,
        }

    train_arr = cp.concatenate(train_arrays, axis=1).astype(cp.float32)
    val_arr = cp.concatenate(val_arrays, axis=1).astype(cp.float32)
    test_arr = cp.concatenate(test_arrays, axis=1).astype(cp.float32)
    return train_arr, val_arr, test_arr, maps


def _normalize_category_series(series: pd.Series) -> pd.Series:
    out = series.astype("object").copy()
    missing = out.isna()
    out.loc[~missing] = out.loc[~missing].map(str)
    out.loc[missing] = np.nan
    return out


def _category_mode(series: pd.Series) -> str:
    modes = series.dropna().mode()
    if modes.empty:
        return "__MISSING__"
    return str(modes.iloc[0])


def _category_codes(series: pd.Series, mode: str, code_map: dict[str, int]) -> pd.Series:
    filled = series.fillna(mode).map(str)
    return filled.map(code_map).fillna(0).astype("float32")


def _make_gpu_reducer(n_components: int, random_state: int) -> tuple[Any, str]:
    try:
        from cuml.decomposition import TruncatedSVD as CuMLTruncatedSVD

        return CuMLTruncatedSVD(n_components=n_components, random_state=random_state), "TruncatedSVD"
    except Exception:
        from cuml.decomposition import PCA

        return PCA(n_components=n_components, random_state=random_state), "PCA"


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    onehot_min_frequency: int | None,
) -> ColumnTransformer:
    transformers = []
    if numeric_cols:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipeline, numeric_cols))
    other_categorical_cols = [c for c in categorical_cols if c != "conn_state"]
    if other_categorical_cols:
        categorical_pipeline = Pipeline(
            steps=[
                ("to_string", FunctionTransformer(_categoricals_to_string, validate=False)),
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", _make_onehot(onehot_min_frequency)),
            ]
        )
        transformers.append(("cat", categorical_pipeline, other_categorical_cols))
    if "conn_state" in categorical_cols:
        transformers.append(("conn_state", ConnStateOneHot(), ["conn_state"]))
    if not transformers:
        raise ValueError("No feature columns available after applying FeaturePolicy.")
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)


def _categoricals_to_string(X: Any) -> Any:
    """Convert categorical values to strings while preserving missing values."""

    if isinstance(X, pd.DataFrame):
        out = X.copy()
        for col in out.columns:
            missing = out[col].isna()
            out[col] = out[col].astype("object")
            out.loc[~missing, col] = out.loc[~missing, col].map(str)
            out.loc[missing, col] = np.nan
        return out

    arr = np.asarray(X, dtype=object).copy()
    missing = pd.isna(arr)
    arr[~missing] = arr[~missing].astype(str)
    arr[missing] = np.nan
    return arr


def _make_onehot(onehot_min_frequency: int | None) -> OneHotEncoder:
    kwargs = {"handle_unknown": "ignore"}
    if onehot_min_frequency:
        kwargs["min_frequency"] = onehot_min_frequency
    try:
        return OneHotEncoder(**kwargs, sparse_output=True)
    except TypeError:  # scikit-learn < 1.2
        kwargs.pop("min_frequency", None)
        return OneHotEncoder(**kwargs, sparse=True)


def _select_log_cols(numeric_cols: list[str], columns: list[str] | None = None) -> list[str]:
    requested = set(DEFAULT_LOG1P_COLUMNS if columns is None else columns)
    # Explicit overrides also cannot log a nominal code or a timestamp by accident.
    invalid = requested - set(DEFAULT_LOG1P_COLUMNS)
    if invalid:
        raise ValueError(f"log1p_columns contains columns without a declared log1p domain: {sorted(invalid)}")
    return [c for c in numeric_cols if c in requested]


def describe_feature_transformations(feature_cols: list[str], log_cols: list[str]) -> dict:
    """Describe registered transformations without learning from any split."""
    result = {}
    for col in feature_cols:
        spec = TON_IOT_SCHEMA.get(col)
        if spec is None:
            continue
        if col == "conn_state":
            transforms = ["semantic_normalization", "training_mode_imputation", "one_hot_train_vocabulary_unknown_all_zero"]
        elif spec.model_treatment == "numeric":
            transforms = ["semantic_normalization"] + (["log1p"] if col in log_cols else [])
            transforms += ["training_median_imputation", "training_standardization"]
        else:
            transforms = ["semantic_normalization", "backend_specific_categorical_encoding"]
        result[col] = {"semantic_type": spec.semantic_type, "model_treatment": spec.model_treatment,
                       "transformations": transforms}
    return result


def apply_log1p(df: pd.DataFrame, log_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in log_cols:
        values = pd.to_numeric(out[col], errors="raise").astype("float64")
        if np.isinf(values).any() or (values < 0).any():
            raise ValueError(f"log1p requires finite nonnegative values or missing values: {col}")
        out[col] = np.log1p(values)
    return out
