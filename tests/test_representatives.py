import numpy as np

from ids_pipeline.config import PipelineConfig
from ids_pipeline.representatives import build_cluster_representatives


def _toy_inputs():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.1],
            [0.3, 0.1],
            [3.0, 3.0],
            [3.1, 3.0],
            [3.2, 3.1],
            [3.3, 3.1],
            [9.0, 9.0],
            [9.2, 9.0],
            [9.3, 9.1],
            [9.4, 9.1],
        ],
        dtype=float,
    )
    cluster_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    distances = np.array([0.10, 0.20, 0.30, 0.40, 0.30, 0.10, 0.20, 0.40, 0.40, 0.10, 0.20, 0.30])
    y = {
        "label": np.array(["0", "0", "0", "0", "1", "1", "1", "1", "1", "1", "1", "1"]),
        "type": np.array(
            ["normal", "normal", "normal", "normal", "dos", "dos", "dos", "dos", "scan", "scan", "scan", "scan"]
        ),
        "cluster_id": cluster_ids.astype(str),
    }
    return X, y, cluster_ids, distances


def test_centroid_representatives_compress_by_cluster(tmp_path):
    X, y, cluster_ids, distances = _toy_inputs()
    config = PipelineConfig(representative_strategy="centroid")

    result = build_cluster_representatives(X, y, cluster_ids, distances, config, "cpu", tmp_path)

    assert result["X"].shape == (3, 2)
    assert result["metadata"]["compression_ratio"] == 0.25
    assert result["representative_indices"] is None


def test_medoid_representatives_select_real_points(tmp_path):
    X, y, cluster_ids, distances = _toy_inputs()
    config = PipelineConfig(representative_strategy="medoid")

    result = build_cluster_representatives(X, y, cluster_ids, distances, config, "cpu", tmp_path)

    assert result["X"].shape[0] == 3
    assert result["metadata"]["compression_ratio"] == 0.25
    assert result["representative_indices"].tolist() == [0, 5, 9]


def test_mixed_representatives_compress_real_points(tmp_path):
    X, y, cluster_ids, distances = _toy_inputs()
    config = PipelineConfig(
        representative_strategy="mixed",
        representatives_per_cluster=1,
        boundary_per_cluster=1,
    )

    result = build_cluster_representatives(X, y, cluster_ids, distances, config, "cpu", tmp_path)

    assert 0 < result["X"].shape[0] < X.shape[0]
    assert result["metadata"]["compression_ratio"] < 1.0
    assert result["representative_indices"] is not None
