import numpy as np
import pandas as pd

from ids_pipeline.config import PipelineConfig
from ids_pipeline.data_loading import make_synthetic_dataset
from ids_pipeline.feature_policy import FeaturePolicy
from ids_pipeline.splitting import (
    create_splits,
    group_overlap_report,
    infer_timestamp_unit,
    parse_timestamp_series,
    target_distribution_pivot,
    target_distribution_table,
    temporal_per_class_report,
    temporal_split_report,
)


def test_group_stratified_no_overlap():
    df = make_synthetic_dataset(240, random_state=21)
    config = PipelineConfig(
        split_strategy="group_stratified",
        group_cols=["src_ip"],
        test_size=0.25,
        val_size=0.25,
        random_state=21,
    )

    splits = create_splits(df, config)
    report = group_overlap_report(df, splits, config.group_cols)

    assert report["ok"] is True
    assert report["train_val_overlap"] == 0
    assert report["train_test_overlap"] == 0
    assert report["val_test_overlap"] == 0


def test_temporal_split_order():
    df = make_synthetic_dataset(90, random_state=22)
    df = df.sample(frac=1.0, random_state=22).reset_index(drop=True)
    config = PipelineConfig(
        split_strategy="temporal",
        timestamp_col="timestamp",
        test_size=0.2,
        val_size=0.2,
        random_state=22,
    )

    splits = create_splits(df, config)
    train_max = pd.to_datetime(df.iloc[splits["train"]]["timestamp"]).max()
    val_min = pd.to_datetime(df.iloc[splits["val"]]["timestamp"]).min()
    val_max = pd.to_datetime(df.iloc[splits["val"]]["timestamp"]).max()
    test_min = pd.to_datetime(df.iloc[splits["test"]]["timestamp"]).min()

    assert train_max <= val_min
    assert val_max <= test_min


def test_temporal_split_keeps_same_time_bucket_together():
    rows = []
    for second in range(8):
        for offset_ms in [1, 100, 900]:
            rows.append(
                {
                    "timestamp": pd.Timestamp("2024-01-01 00:00:00")
                    + pd.Timedelta(seconds=second, milliseconds=offset_ms),
                    "label": second % 2,
                    "type": "normal" if second % 2 == 0 else "attack",
                    "duration": float(second),
                    "src_bytes": second + offset_ms,
                }
            )
    df = pd.DataFrame(rows)
    config = PipelineConfig(
        split_strategy="temporal",
        timestamp_col="timestamp",
        temporal_bucket_freq="1s",
        test_size=0.25,
        val_size=0.25,
        random_state=24,
    )

    splits = create_splits(df, config)
    report = temporal_split_report(df, splits, config)

    assert report["ok"] is True
    assert report["train_val_bucket_overlap"] == 0
    assert report["train_test_bucket_overlap"] == 0
    assert report["val_test_bucket_overlap"] == 0


def test_temporal_split_parses_numeric_unix_seconds():
    base = 1_700_000_000.0
    df = pd.DataFrame(
        {
            "ts": [base + i + offset for i in range(8) for offset in [0.001, 0.100, 0.900]],
            "label": [i % 2 for i in range(8) for _ in range(3)],
            "type": ["normal" if i % 2 == 0 else "attack" for i in range(8) for _ in range(3)],
            "duration": np.arange(24),
        }
    )
    config = PipelineConfig(
        split_strategy="temporal",
        timestamp_col="ts",
        timestamp_unit="auto",
        temporal_bucket_freq="1s",
        test_size=0.25,
        val_size=0.25,
    )

    parsed = parse_timestamp_series(df["ts"], unit=config.timestamp_unit)
    splits = create_splits(df, config)
    report = temporal_split_report(df, splits, config)

    assert infer_timestamp_unit(df["ts"]) == "s"
    assert parsed.dt.year.min() >= 2023
    assert report["ok"] is True


def test_temporal_per_class_keeps_each_type_in_train_and_test():
    rows = []
    for attack_type in ["normal", "ddos", "xss"]:
        for i in range(12):
            rows.append(
                {
                    "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=i),
                    "label": 0 if attack_type == "normal" else 1,
                    "type": attack_type,
                    "duration": float(i),
                    "src_bytes": i + 1,
                }
            )
    df = pd.DataFrame(rows)
    config = PipelineConfig(
        split_strategy="temporal_per_class",
        timestamp_col="timestamp",
        temporal_bucket_freq="1s",
        test_size=0.25,
        val_size=0.25,
    )

    splits = create_splits(df, config)
    report, per_class = temporal_per_class_report(df, splits, config)

    assert report["ok"] is True
    assert report["classes_missing_train"] == []
    assert report["classes_missing_test"] == []
    assert set(per_class["class_value"]) == {"normal", "ddos", "xss"}

    for attack_type in ["normal", "ddos", "xss"]:
        train_times = pd.to_datetime(df.iloc[splits["train"]].query("type == @attack_type")["timestamp"])
        val_times = pd.to_datetime(df.iloc[splits["val"]].query("type == @attack_type")["timestamp"])
        test_times = pd.to_datetime(df.iloc[splits["test"]].query("type == @attack_type")["timestamp"])
        assert train_times.max() <= val_times.min()
        assert val_times.max() <= test_times.min()


def test_group_cols_not_forced_into_features():
    df = make_synthetic_dataset(80, random_state=23)
    config = PipelineConfig(
        feature_policy="no_raw_ip_port",
        split_strategy="group_stratified",
        group_cols=["src_ip", "dst_ip", "service", "proto"],
    )
    features = FeaturePolicy.from_name(config.feature_policy).select_features(
        df,
        label_col=config.label_col,
        type_col=config.type_col,
    )

    assert "src_ip" not in features
    assert "dst_ip" not in features
    assert "src_port" not in features
    assert "dst_port" not in features
    assert "proto" in features
    assert "service" in features


def test_target_distribution_table_and_pivot():
    df = make_synthetic_dataset(60, random_state=25)
    config = PipelineConfig(random_state=25)
    splits = create_splits(df, config)

    distribution = target_distribution_table(df, splits, ["label", "type"])
    pivot = target_distribution_pivot(distribution, "type")

    assert {"split", "target", "class_value", "count", "fraction"}.issubset(distribution.columns)
    assert {"class_value", "train_count", "val_count", "test_count"}.issubset(pivot.columns)
    assert int(pivot[["train_count", "val_count", "test_count"]].to_numpy().sum()) == len(df)
