import pandas as pd

from ids_pipeline.feature_policy import FeaturePolicy


def _df():
    return pd.DataFrame(
        {
            "src_ip": ["10.0.0.1"],
            "dst_ip": ["10.0.0.2"],
            "src_port": [1234],
            "dst_port": [80],
            "service": ["http"],
            "proto": ["tcp"],
            "duration": [1.2],
            "src_bytes": [100],
            "dst_pkts": [4],
            "conn_state": ["SF"],
            "flow_id": ["abc"],
            "uid": ["u1"],
            "dataset_id": ["synthetic"],
            "label": [0],
            "type": ["normal"],
        }
    )


def test_supported_feature_policies_select_expected_columns():
    df = _df()

    all_except = FeaturePolicy.from_name("all_except_labels").select_features(df)
    assert "label" not in all_except
    assert "type" not in all_except
    assert "src_ip" in all_except
    assert "src_port" in all_except

    no_raw_ip = FeaturePolicy.from_name("no_raw_ip").select_features(df)
    assert "src_ip" not in no_raw_ip
    assert "dst_ip" not in no_raw_ip
    assert "src_port" in no_raw_ip
    assert "dst_port" in no_raw_ip


def test_no_raw_ip_port_removes_ip_ports_and_targets():
    df = _df()
    cols = FeaturePolicy.from_name("no_raw_ip_port").select_features(df)
    forbidden = {"src_ip", "dst_ip", "src_port", "dst_port", "label", "type"}
    assert forbidden.isdisjoint(cols)
    assert "proto" in cols
    assert "service" in cols


def test_behavioral_strict_removes_service_and_identity_columns():
    df = _df()
    cols = FeaturePolicy.from_name("behavioral_strict").select_features(df)
    forbidden = {"src_ip", "dst_ip", "src_port", "dst_port", "service", "label", "type", "flow_id", "uid"}
    assert forbidden.isdisjoint(cols)
    assert "duration" in cols
    assert "src_bytes" in cols
    assert "conn_state" in cols
