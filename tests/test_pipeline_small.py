import json

from ids_pipeline.cli import run_experiment
from ids_pipeline.config import PipelineConfig


def test_pipeline_stage_all_runs_and_writes_required_outputs(tmp_path):
    output_dir = tmp_path / "debug_small"
    config = PipelineConfig(
        dataset_name="synthetic_test",
        synthetic_n=120,
        feature_policy="no_raw_ip_port",
        svd_components=5,
        selected_k=3,
        cluster_batch_size=32,
        cluster_n_init=2,
        cluster_max_iter=20,
        silhouette_sample_size=50,
        random_forest_estimators=20,
        n_jobs=1,
        random_state=11,
    )
    result = run_experiment(config, output_dir, stage="all")
    assert "clustering" in result
    assert "supervised" in result

    required = [
        "metrics_clustering.json",
        "metrics_supervised.json",
        "profiling.json",
        "selected_features.json",
        "forbidden_columns_check.json",
        "cluster_assignments.csv",
        "classification_report.txt",
        "confusion_matrix.csv",
    ]
    for filename in required:
        assert (output_dir / filename).exists()

    selected = json.loads((output_dir / "selected_features.json").read_text())
    forbidden = {"src_ip", "dst_ip", "src_port", "dst_port", "label", "type"}
    assert forbidden.isdisjoint(set(selected["selected_features"]))

    leakage_check = json.loads((output_dir / "forbidden_columns_check.json").read_text())
    assert leakage_check["ok"] is True

    supervised = json.loads((output_dir / "metrics_supervised.json").read_text())
    assert "test" in supervised
    assert "macro_f1" in supervised["test"]
