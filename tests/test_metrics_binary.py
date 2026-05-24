import numpy as np

from ids_pipeline.supervised import compute_fpr_at_tpr


def test_compute_fpr_at_tpr_returns_threshold():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.05, 0.20, 0.40, 0.70, 0.90, 0.95])

    result = compute_fpr_at_tpr(y_true, scores, target_tpr=0.95)

    assert result is not None
    assert 0.0 <= result["fpr_at_tpr_95"] <= 1.0
    assert 0.0 <= result["threshold_at_tpr_95"] <= 1.0


def test_compute_fpr_at_tpr_returns_none_for_one_class():
    y_true = np.array([0, 0, 0])
    scores = np.array([0.1, 0.2, 0.3])

    assert compute_fpr_at_tpr(y_true, scores, target_tpr=0.95) is None
