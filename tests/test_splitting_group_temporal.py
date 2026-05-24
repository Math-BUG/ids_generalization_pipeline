import numpy as np
import pandas as pd

from ids_pipeline.config import PipelineConfig
from ids_pipeline.data_loading import make_synthetic_dataset
from ids_pipeline.feature_policy import FeaturePolicy
from ids_pipeline.splitting import create_splits, group_overlap_report


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
