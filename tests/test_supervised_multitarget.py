import json
import numpy as np

from ids_pipeline.config import PipelineConfig
from ids_pipeline.cli import run_experiment
from ids_pipeline.data_loading import make_synthetic_dataset
from ids_pipeline.representatives import build_cluster_representatives
from ids_pipeline.splitting import create_splits
from ids_pipeline.supervised import train_random_forest


def test_supervised_trains_label_type_and_cluster_id(tmp_path):
    df = make_synthetic_dataset(140, random_state=41)
    config = PipelineConfig(
        targets=["label", "type", "cluster_id"],
        selected_k=3,
        representative_strategy="mixed",
        use_representatives_for_supervised=True,
        representatives_per_cluster=4,
        boundary_per_cluster=2,
        random_forest_estimators=10,
        n_jobs=1,
        random_state=41,
    )
    splits = create_splits(df, config)
    rng = np.random.default_rng(41)
    X = {split: rng.normal(size=(len(idx), 5)) for split, idx in splits.items()}
    cluster_labels = {split: rng.integers(0, 3, size=len(idx)) for split, idx in splits.items()}
    distances = {split: rng.random(size=len(idx)) for split, idx in splits.items()}
    y_train = {
        "label": df.iloc[splits["train"]][config.label_col].astype(str).to_numpy(),
        "type": df.iloc[splits["train"]][config.type_col].astype(str).to_numpy(),
        "cluster_id": cluster_labels["train"].astype(str),
    }
    reps = build_cluster_representatives(
        X["train"],
        y_train,
        cluster_labels["train"],
        distances["train"],
        config,
        "cpu",
        tmp_path,
        train_indices=splits["train"],
    )

    metrics = train_random_forest(
        X,
        df,
        splits,
        ["duration", "src_bytes", "dst_bytes"],
        config,
        tmp_path,
        cluster_labels=cluster_labels,
        representatives=reps,
    )

    assert set(metrics["targets"]) == {"label", "type", "cluster_id"}
    assert (tmp_path / "artifacts" / "random_forest_label.joblib").exists()
    assert (tmp_path / "artifacts" / "random_forest_type.joblib").exists()
    assert (tmp_path / "artifacts" / "random_forest_cluster_id.joblib").exists()
    saved = json.loads((tmp_path / "metrics_supervised.json").read_text())
    assert "pr_auc" in saved["targets"]["label"]["test"]
    assert "fpr_at_tpr_95" in saved["targets"]["label"]["test"]


def test_pipeline_runs_multitarget_with_representatives(tmp_path):
    config = PipelineConfig(
        dataset_name="synthetic_multitarget",
        synthetic_n=120,
        targets=["label", "type", "cluster_id"],
        selected_k=3,
        svd_components=5,
        representative_strategy="mixed",
        use_representatives_for_supervised=True,
        representatives_per_cluster=3,
        boundary_per_cluster=1,
        random_forest_estimators=10,
        silhouette_sample_size=40,
        n_jobs=1,
        random_state=42,
    )

    result = run_experiment(config, tmp_path / "experiment", stage="all")

    assert set(result["supervised"]["targets"]) == {"label", "type", "cluster_id"}
    assert "representatives" in result["profiling"]["stages"]
    assert "supervised_cluster_id" in result["profiling"]["stages"]
