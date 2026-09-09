"""Small export regression checks; no real dataset or existing CSV is read."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from ids_pipeline import clustering
from ids_pipeline.config import PipelineConfig, config_to_dict, load_config


@pytest.mark.parametrize("setting, expected", [(None, True), ("true", True), ("false", False)])
def test_export_config_is_backward_compatible(tmp_path, setting, expected):
    path = tmp_path / "config.yaml"
    path.write_text("{}" if setting is None else f"export_cluster_assignments: {setting}\n")
    config = load_config(path)
    assert config.export_cluster_assignments is expected
    assert config_to_dict(config)["export_cluster_assignments"] is expected
    assert "export_cluster_assignments" not in config.extra


def test_full_ton_iot_gpu_disables_assignment_export():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/ton_iot_behavioral_strict_gpu.yaml")
    assert config.export_cluster_assignments is False
    assert config.compute_backend == "gpu"


@pytest.mark.parametrize("backend", ["cpu", "gpu_stub"])
def test_export_toggle_preserves_clustering_and_metrics(tmp_path, monkeypatch, backend):
    # GPU routing is tested with a CPU fit substitute, not claimed as CUDA validation.
    monkeypatch.delenv("IDS_COMPUTE_BACKEND", raising=False)
    if backend == "gpu_stub":
        monkeypatch.setattr(clustering, "resolve_backend", lambda _: "gpu")
        monkeypatch.setattr(clustering, "_fit_gpu_kmeans", clustering._fit_cpu_minibatch_kmeans)

    rng = np.random.default_rng(42)
    classes = np.tile(np.arange(3), 30)
    values = rng.normal(scale=0.1, size=(90, 2)) + classes[:, None] * 5
    df = pd.DataFrame({"duration": values[:, 0], "label": classes != 0, "type": classes})
    splits = {"train": np.arange(60), "val": np.arange(60, 75), "test": np.arange(75, 90)}
    X = {split: values[idx] for split, idx in splits.items()}
    config = PipelineConfig(cluster_n_init=1, cluster_max_iter=10, silhouette_sample_size=60)

    assignment_frames = []
    concatenated = []
    allow_export = True

    def frame(data, *args, **kwargs):
        if isinstance(data, dict) and "cluster_distance" in data:
            assert allow_export, "Disabled export must not allocate assignment DataFrames"
            assignment_frames.append(data)
        return pd.DataFrame(data, *args, **kwargs)

    def concat(*args, **kwargs):
        assert allow_export, "Disabled export must not concatenate assignment DataFrames"
        result = pd.concat(*args, **kwargs)
        concatenated.append(result)
        return result

    # Replace only this module's pandas reference, preserving pandas internals.
    monkeypatch.setattr(clustering, "pd", SimpleNamespace(DataFrame=frame, concat=concat))
    results = {}
    with threadpool_limits(limits=1):
        for enabled in (True, False):
            allow_export = enabled
            output = tmp_path / str(enabled)
            result = clustering.fit_predict_clustering(
                X, df, splits, ["duration"], config.with_updates(export_cluster_assignments=enabled), output
            )
            results[enabled] = result
            csv = output / "cluster_assignments.csv"
            assert csv.exists() is enabled
            if enabled:
                assert csv.stat().st_size > 0
            metrics = json.loads((output / "metrics_clustering.json").read_text())
            assert metrics == result["metrics"]
            assert metrics["fit_n_samples"] == 60
            assert metrics["n_clusters_observed_train"] == 3
            for key in ("inertia", "silhouette_train", "davies_bouldin_train", "calinski_harabasz_train"):
                assert np.isfinite(metrics[key])
            for split in splits:
                for target in ("label", "type"):
                    for metric in ("ami", "v_measure", "purity"):
                        assert np.isfinite(metrics[f"{split}_{target}_{metric}"])
            model = "cuml_kmeans.joblib" if backend == "gpu_stub" else "minibatch_kmeans.joblib"
            assert (output / "artifacts" / model).is_file()

    assert len(assignment_frames) == 3
    assert len(concatenated) == 1
    exported = concatenated[0]
    assert list(exported.columns) == ["split", "row_index", "cluster_id", "cluster_distance"]
    np.testing.assert_array_equal(exported["row_index"], np.concatenate(list(splits.values())))
    assert exported["split"].tolist() == [split for split, idx in splits.items() for _ in idx]
    np.testing.assert_array_equal(exported["cluster_id"], np.concatenate(list(results[True]["labels"].values())))
    np.testing.assert_array_equal(
        exported["cluster_distance"], np.concatenate(list(results[True]["distances"].values()))
    )
    assert results[True].keys() == results[False].keys() == {"clusterer", "labels", "distances", "metrics"}
    assert results[True]["metrics"] == results[False]["metrics"]
    np.testing.assert_array_equal(results[True]["clusterer"].cluster_centers_, results[False]["clusterer"].cluster_centers_)
    for key in ("labels", "distances"):
        for split in splits:
            np.testing.assert_array_equal(results[True][key][split], results[False][key][split])
