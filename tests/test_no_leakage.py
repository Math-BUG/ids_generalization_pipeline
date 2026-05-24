import pytest

from ids_pipeline.config import PipelineConfig
from ids_pipeline.data_loading import make_synthetic_dataset
from ids_pipeline.feature_policy import FeaturePolicy
from ids_pipeline.leakage_checks import LeakageError, check_for_leakage_columns
from ids_pipeline.preprocessing import fit_transform_preprocessing
from ids_pipeline.splitting import create_splits


def test_no_raw_ip_port_fails_when_ip_or_port_enters_features():
    df = make_synthetic_dataset(80, random_state=3)
    unsafe_features = ["src_ip", "dst_port", "duration", "src_bytes"]
    with pytest.raises(LeakageError):
        check_for_leakage_columns(
            df,
            unsafe_features,
            feature_policy="no_raw_ip_port",
            label_col="label",
            type_col="type",
        )


def test_behavioral_strict_fails_when_port_enters_features():
    df = make_synthetic_dataset(80, random_state=4)
    unsafe_features = ["src_port", "duration", "src_bytes"]
    with pytest.raises(LeakageError):
        check_for_leakage_columns(
            df,
            unsafe_features,
            feature_policy="behavioral_strict",
            label_col="label",
            type_col="type",
        )


def test_label_and_type_always_fail_as_features():
    df = make_synthetic_dataset(80, random_state=5)
    for policy in ["all_except_labels", "no_raw_ip", "no_raw_ip_port", "behavioral_strict"]:
        with pytest.raises(LeakageError):
            check_for_leakage_columns(
                df,
                ["duration", "label", "type"],
                feature_policy=policy,
                label_col="label",
                type_col="type",
            )


def test_safe_policy_reaches_preprocessing_without_leakage(tmp_path):
    df = make_synthetic_dataset(80, random_state=6)
    config = PipelineConfig(svd_components=3, random_state=6, feature_policy="no_raw_ip_port")
    splits = create_splits(df, config)
    features = FeaturePolicy.from_name(config.feature_policy).select_features(
        df, label_col=config.label_col, type_col=config.type_col
    )
    X, bundle = fit_transform_preprocessing(df, splits, features, config, tmp_path)
    assert X["train"].shape[0] == len(splits["train"])
    assert "src_ip" not in bundle.feature_cols
    assert "dst_port" not in bundle.feature_cols


def test_preprocessing_handles_mixed_type_categoricals(tmp_path):
    df = make_synthetic_dataset(80, random_state=7)
    df["mixed_category"] = ["1" if i % 2 else 1 for i in range(len(df))]
    config = PipelineConfig(svd_components=3, random_state=7, feature_policy="no_raw_ip_port")
    splits = create_splits(df, config)
    features = FeaturePolicy.from_name(config.feature_policy).select_features(
        df, label_col=config.label_col, type_col=config.type_col
    )

    X, bundle = fit_transform_preprocessing(df, splits, features, config, tmp_path)

    assert "mixed_category" in bundle.categorical_cols
    assert X["train"].shape[0] == len(splits["train"])
