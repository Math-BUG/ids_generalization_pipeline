import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from ids_pipeline.config import PipelineConfig
from ids_pipeline.data_loading import load_dataset
from ids_pipeline.dataset_schema import (
    SchemaValidationError, model_feature_types, normalize_dataset_schema,
    read_csv_preserving_schema,
)
from ids_pipeline.schema import TON_IOT_LOG1P_COLUMNS, TON_IOT_SCHEMA, TON_IOT_SCHEMA_VERSION
from ids_pipeline import preprocessing as prep


def ton_frame(n=8):
    return pd.DataFrame({
        "ts": np.arange(n) + 1554198358,
        "src_ip": ["10.0.0.1"] * n, "dst_ip": ["10.0.0.2"] * n,
        "src_bytes": [str(i * 10) for i in range(n)],
        "dns_qtype": [1, 28] * (n // 2),
        "http_trans_depth": ["1", "-"] * (n // 2),
        "dns_AA": ["T", "F"] * (n // 2),
        "label": [0, 1] * (n // 2), "type": ["normal", "ddos"] * (n // 2),
    })


def test_schema_covers_observed_47_column_header():
    # Header union inspected without loading full datasets; uid is in partition 6.
    columns = """ts uid src_ip src_port dst_ip dst_port proto service duration
    src_bytes dst_bytes conn_state missed_bytes src_pkts src_ip_bytes dst_pkts
    dst_ip_bytes dns_query dns_qclass dns_qtype dns_rcode dns_AA dns_RD dns_RA
    dns_rejected ssl_version ssl_cipher ssl_resumed ssl_established ssl_subject
    ssl_issuer http_trans_depth http_method http_uri http_referrer http_version
    http_request_body_len http_response_body_len http_status_code http_user_agent
    http_orig_mime_types http_resp_mime_types weird_name weird_addl weird_notice label type""".split()
    assert set(TON_IOT_SCHEMA) == set(columns)
    assert len(TON_IOT_SCHEMA) == 47
    assert TON_IOT_SCHEMA["ts"].semantic_type == "timestamp"
    assert TON_IOT_SCHEMA["ts"].unit == "Unix seconds"


def test_src_bytes_numeric_strings_and_registered_missing():
    frame = pd.DataFrame({"src_bytes": [10, "20", "3e2", "-", "", None, np.nan, pd.NA]})
    normalized, report = normalize_dataset_schema(frame)
    assert normalized.src_bytes.dtype == np.dtype("float64")
    assert normalized.src_bytes.iloc[:3].tolist() == [10, 20, 300]
    assert normalized.src_bytes.iloc[3:].isna().all()
    assert report["columns"]["src_bytes"]["semantic_type"] == "count"
    assert report["columns"]["src_bytes"]["sentinel_missing_count"] == 2
    assert model_feature_types(normalized, ["src_bytes"]) == (["src_bytes"], [])


@pytest.mark.parametrize("column", TON_IOT_LOG1P_COLUMNS)
def test_declared_quantities_are_numeric_independent_of_physical_type(column):
    numeric, _ = normalize_dataset_schema(pd.DataFrame({column: [1, 2]}))
    strings, _ = normalize_dataset_schema(pd.DataFrame({column: ["1", "2"]}))
    pd.testing.assert_frame_equal(numeric, strings)
    assert model_feature_types(strings, [column]) == ([column], [])


@pytest.mark.parametrize("column", ["dns_qclass", "dns_qtype", "dns_rcode", "http_status_code"])
def test_numeric_codes_are_canonical_categories(column):
    frame, _ = normalize_dataset_schema(pd.DataFrame({column: [1, "1", 1.0, "01", "0", "-"]}))
    assert frame[column].iloc[:5].tolist() == ["1", "1", "1", "1", "0"]
    assert pd.isna(frame[column].iloc[5])
    assert model_feature_types(frame, [column]) == ([], [column])


@pytest.mark.parametrize("column", ["dns_AA", "dns_RD", "dns_RA", "dns_rejected",
                                    "ssl_resumed", "ssl_established", "weird_notice"])
def test_explicit_indicator_normalization(column):
    raw = pd.DataFrame({column: [True, False, "T", "F", "true", "false", 1, 0, "-"]})
    frame, _ = normalize_dataset_schema(raw)
    assert frame[column].iloc[:8].tolist() == ["T", "F"] * 4
    assert pd.isna(frame[column].iloc[8])
    assert model_feature_types(frame, [column]) == ([], [column])


@pytest.mark.parametrize("token", ["not-a-number", "NA", "NULL", "nan", "None"])
def test_invalid_token_is_not_a_missing_sentinel(token, tmp_path):
    report_path = tmp_path / "schema.json"
    with pytest.raises(SchemaValidationError) as caught:
        normalize_dataset_schema(pd.DataFrame({"src_bytes": [token]}), report_path=report_path)
    report = json.loads(report_path.read_text())
    assert not report["ok"]
    assert caught.value.report["columns"]["src_bytes"]["invalid_count"] == 1
    assert report["columns"]["src_bytes"]["issues"]["conversion_failure"]["count"] == 1
    assert report["columns"]["src_bytes"]["issues"]["conversion_failure"]["examples"][0]["position"] == 0


@pytest.mark.parametrize("value", [np.inf, -np.inf, "inf", "-inf"])
def test_infinite_values_are_reported(value):
    with pytest.raises(SchemaValidationError) as caught:
        normalize_dataset_schema(pd.DataFrame({"duration": [value]}))
    assert caught.value.report["columns"]["duration"]["issues"]["infinite"]["count"] == 1


@pytest.mark.parametrize("column,value", [("src_bytes", -1), ("src_pkts", 1.5),
    ("src_port", 65536), ("label", 2), ("dns_AA", "yes"), ("ts", -1), ("ts", 1e20)])
def test_domain_violations_are_reported(column, value):
    with pytest.raises(SchemaValidationError) as caught:
        normalize_dataset_schema(pd.DataFrame({column: [value]}))
    assert caught.value.report["columns"][column]["issues"]["domain_violation"]["count"] == 1


def test_large_integers_do_not_silently_lose_precision():
    with pytest.raises(SchemaValidationError) as caught:
        normalize_dataset_schema(pd.DataFrame({"src_bytes": ["9007199254740993"]}))
    assert caught.value.report["columns"]["src_bytes"]["issues"]["unsafe_integer_precision"]["count"] == 1


def test_rows_index_and_input_are_preserved_including_duplicate_indices():
    frame = pd.DataFrame({"src_bytes": ["1", "-", "3"], "uid": ["a", "b", "c"],
                          "extra": [9, 8, 7]}, index=pd.Index([4, 4, 1], name="row"))
    original = frame.copy(deep=True)
    normalized, report = normalize_dataset_schema(frame)
    pd.testing.assert_frame_equal(frame, original)
    assert normalized.index.equals(frame.index)
    assert normalized.uid.tolist() == frame.uid.tolist()
    assert len(normalized) == 3
    assert normalized.extra.equals(frame.extra)
    assert report["unknown_columns"] == ["extra"]
    assert report["index_preserved"]
    again, _ = normalize_dataset_schema(normalized)
    pd.testing.assert_frame_equal(normalized, again)


def test_targets_never_enter_model_and_missing_targets_fail():
    for name in ["label", "type"]:
        with pytest.raises(ValueError, match="Target"):
            model_feature_types(pd.DataFrame({name: [1]}), [name])
        with pytest.raises(SchemaValidationError):
            normalize_dataset_schema(pd.DataFrame({name: [None]}))


def test_unknown_ton_feature_cannot_be_inferred_silently():
    with pytest.raises(ValueError, match="Unregistered"):
        model_feature_types(pd.DataFrame({"new_counter": [1]}), ["new_counter"], strict=True)


def test_existing_timestamp_derived_features_remain_supported(tmp_path):
    from ids_pipeline.feature_policy import apply_proxy_feature_policies

    frame, cols, _ = apply_proxy_feature_policies(
        ton_frame(), ["ts", "src_bytes"], timestamp_col="ts", timestamp_policy="extract_causal",
    )
    splits = {"train": np.arange(4), "val": np.array([4, 5]), "test": np.array([6, 7])}
    _, bundle = prep.fit_transform_preprocessing(
        frame, splits, cols, PipelineConfig(svd_components=0), tmp_path,
    )
    assert bundle.numeric_cols == ["src_bytes", "ts_hour", "ts_day_of_week"]
    assert bundle.log_cols == ["src_bytes"]


def test_all_registered_raw_columns_normalize_and_preserve_text():
    # Explicit valid fixture spanning every field family, including nulls.
    raw = {}
    for name, spec in TON_IOT_SCHEMA.items():
        if spec.semantic_type == "boolean":
            raw[name] = [True, "-"]
        elif spec.model_treatment == "target":
            raw[name] = [0, 1] if name == "label" else ["normal", "ddos"]
        elif spec.storage_dtype in {"float64", "Int64"} or spec.semantic_type == "nominal_code":
            raw[name] = ["1", "-"]
        else:
            raw[name] = ["NA", "-"]
    frame, report = normalize_dataset_schema(pd.DataFrame(raw))
    assert report["ok"] and not report["unknown_columns"] and not report["absent_columns"]
    assert frame.shape == (2, 47)
    assert frame.http_orig_mime_types.iloc[0] == "NA"
    assert frame.dtypes.astype(str).to_dict() == {name: spec.storage_dtype for name, spec in TON_IOT_SCHEMA.items()}


def test_log1p_is_explicit_and_never_clips():
    assert prep._select_log_cols(["src_bytes", "ssl_resumed", "invented_bytes", "dns_qtype"]) == ["src_bytes"]
    assert prep._select_log_cols(["src_bytes"], []) == []
    with pytest.raises(ValueError):
        prep._select_log_cols(["dns_qtype"], ["dns_qtype"])
    for value in [-1, np.inf, -np.inf, "broken"]:
        with pytest.raises(ValueError):
            prep.apply_log1p(pd.DataFrame({"src_bytes": [value]}), ["src_bytes"])


def test_csv_loading_preserves_tokens_and_writes_failure_report(tmp_path):
    frame = ton_frame()
    frame.loc[0, "src_bytes"] = "NA"
    path = tmp_path / "partition.csv"
    frame.to_csv(path, index=False)
    assert read_csv_preserving_schema(path).src_bytes.iloc[0] == "NA"
    with pytest.raises(SchemaValidationError):
        load_dataset(str(path), schema_report_path=tmp_path / "report.json")
    assert not json.loads((tmp_path / "report.json").read_text())["ok"]


def test_old_parquet_is_normalized_without_reconversion(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "partition.parquet"
    ton_frame().to_parquet(path, index=False)
    frame = load_dataset(str(path), compute_backend="cpu", schema_report_path=tmp_path / "report.json")
    assert frame.src_bytes.dtype == np.dtype("float64")
    assert str(frame.dns_qtype.dtype) == "string"
    assert frame.attrs["schema_version"] == TON_IOT_SCHEMA_VERSION


def test_validation_test_do_not_change_cpu_fitted_statistics(tmp_path):
    frame = ton_frame()
    frame.loc[1, "src_bytes"] = "-"
    splits = {"train": np.arange(4), "val": np.array([4, 5]), "test": np.array([6, 7])}
    cols = ["src_bytes", "dns_qtype", "http_trans_depth", "dns_AA"]
    config = PipelineConfig(svd_components=2, compute_backend="cpu")
    x1, b1 = prep.fit_transform_preprocessing(frame, splits, cols, config, tmp_path / "one")
    changed = frame.copy()
    changed.loc[4:, "src_bytes"] = "999999"
    changed.loc[4:, "dns_qtype"] = 99
    x2, b2 = prep.fit_transform_preprocessing(changed, splits, cols, config, tmp_path / "two")
    for step, attr in [("imputer", "statistics_"), ("scaler", "mean_"), ("scaler", "scale_")]:
        a = getattr(b1.preprocessor.named_transformers_["num"].named_steps[step], attr)
        b = getattr(b2.preprocessor.named_transformers_["num"].named_steps[step], attr)
        np.testing.assert_allclose(a, b)
    enc1 = b1.preprocessor.named_transformers_["cat"].named_steps["onehot"]
    enc2 = b2.preprocessor.named_transformers_["cat"].named_steps["onehot"]
    for a, b in zip(enc1.categories_, enc2.categories_):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(b1.svd.components_, b2.svd.components_)
    np.testing.assert_allclose(x1["train"], x2["train"])


def test_cpu_and_gpu_dispatch_identical_semantic_columns(monkeypatch, tmp_path):
    frame = ton_frame()
    cols = ["src_bytes", "dns_qtype", "http_trans_depth", "dns_AA"]
    splits = {"train": np.arange(4), "val": np.array([4, 5]), "test": np.array([6, 7])}
    config = PipelineConfig(svd_components=0)
    _, cpu = prep.fit_transform_preprocessing(frame, splits, cols, config, tmp_path / "cpu")
    captured = {}
    def gpu_stub(train, val, test, features, numeric, categorical, log, config, output):
        captured.update(numeric=numeric, categorical=categorical, log=log)
        assert pd.api.types.is_numeric_dtype(train.src_bytes)
        assert train.dns_qtype.tolist() == ["1", "28", "1", "28"]
        return {}, None
    monkeypatch.setattr(prep, "resolve_backend", lambda backend: "gpu")
    monkeypatch.setattr(prep, "_fit_transform_gpu_preprocessing", gpu_stub)
    prep.fit_transform_preprocessing(frame, splits, cols, config, tmp_path / "gpu")
    assert captured == {"numeric": cpu.numeric_cols, "categorical": cpu.categorical_cols, "log": cpu.log_cols}


def test_gpu_helpers_fit_statistics_only_on_training_with_numpy_stub(monkeypatch):
    # Exercises host-side GPU preparation without claiming a CUDA execution.
    monkeypatch.setitem(sys.modules, "cupy", np)
    train = pd.DataFrame({"src_bytes": [1.0, np.nan, 3.0], "dns_qtype": ["1", "28", "1"]})
    first = train.copy()
    changed = pd.DataFrame({"src_bytes": [1e9, 2e9, 3e9], "dns_qtype": ["99"] * 3})
    a = prep._gpu_numeric_arrays(train, first, first, ["src_bytes"])
    b = prep._gpu_numeric_arrays(train, changed, changed, ["src_bytes"])
    assert a[3] == b[3]
    np.testing.assert_array_equal(a[0], b[0])
    a = prep._gpu_categorical_code_arrays(train, first, first, ["dns_qtype"], 1, 64)
    b = prep._gpu_categorical_code_arrays(train, changed, changed, ["dns_qtype"], 1, 64)
    assert a[3] == b[3]
    np.testing.assert_array_equal(a[0], b[0])


def test_converter_and_loader_share_schema_across_partitions(tmp_path):
    pytest.importorskip("pyarrow")
    spec = importlib.util.spec_from_file_location("converter", Path(__file__).parents[1] / "scripts/convert_csv_to_parquet.py")
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    first, second = ton_frame(), ton_frame()
    second["src_bytes"] = np.arange(len(second))
    second["uid"] = ["id-" + str(i) for i in range(len(second))]
    paths = [tmp_path / "first.csv", tmp_path / "second.csv"]
    for frame, path in zip([first, second], paths):
        frame.to_csv(path, index=False)
    schema, cols = converter.infer_common_schema(paths)
    assert schema["src_bytes"] == "float64"
    assert schema["dns_qtype"] == "string"
    normalized = [converter.normalize_to_common_schema(read_csv_preserving_schema(p), schema, cols) for p in paths]
    assert normalized[0].dtypes.equals(normalized[1].dtypes)
    assert normalized[0].uid.isna().all()
    output = tmp_path / "converted"
    output.mkdir()
    for i, frame in enumerate(normalized):
        frame.to_parquet(output / f"part{i}.parquet", index=False)
    loaded = load_dataset(str(output), data_quality_report_path=tmp_path / "quality.json")
    assert len(loaded) == 16
    assert loaded.src_bytes.dtype == np.dtype("float64")
    assert loaded.dns_qtype.iloc[0] == "1"
