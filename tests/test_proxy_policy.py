from ids_pipeline.config import PipelineConfig
from ids_pipeline.data_loading import make_synthetic_dataset
from ids_pipeline.feature_policy import FeaturePolicy, apply_proxy_feature_policies


def test_timestamp_policy_drop_removes_timestamp():
    df = make_synthetic_dataset(50, random_state=31)
    config = PipelineConfig(timestamp_col="timestamp", timestamp_policy="drop")
    raw_features = FeaturePolicy.from_name("no_raw_ip_port").select_features(
        df,
        label_col=config.label_col,
        type_col=config.type_col,
    )

    _, features, metadata = apply_proxy_feature_policies(
        df,
        raw_features,
        timestamp_col=config.timestamp_col,
        timestamp_policy=config.timestamp_policy,
        service_policy="keep",
    )

    assert "timestamp" not in features
    assert metadata["removed_by_timestamp_policy"] == ["timestamp"]


def test_timestamp_policy_extract_causal_adds_simple_time_features():
    df = make_synthetic_dataset(50, random_state=32)
    raw_features = FeaturePolicy.from_name("no_raw_ip_port").select_features(
        df,
        label_col="label",
        type_col="type",
    )

    transformed, features, metadata = apply_proxy_feature_policies(
        df,
        raw_features,
        timestamp_col="timestamp",
        timestamp_policy="extract_causal",
        service_policy="keep",
    )

    assert "timestamp" not in features
    assert "timestamp_hour" in features
    assert "timestamp_day_of_week" in features
    assert "timestamp_hour" in transformed.columns
    assert metadata["added_by_timestamp_policy"] == ["timestamp_hour", "timestamp_day_of_week"]


def test_service_policy_drop_removes_service_aliases():
    df = make_synthetic_dataset(50, random_state=33)
    df["svc"] = df["service"]
    raw_features = FeaturePolicy.from_name("no_raw_ip_port").select_features(
        df,
        label_col="label",
        type_col="type",
    )

    _, features, metadata = apply_proxy_feature_policies(
        df,
        raw_features,
        timestamp_col="timestamp",
        timestamp_policy="keep",
        service_policy="drop",
    )

    assert "service" not in features
    assert "svc" not in features
    assert set(metadata["removed_by_service_policy"]) == {"service", "svc"}
