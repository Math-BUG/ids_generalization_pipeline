"""Regression for job 5444's nvforest input limit; no real CUDA or large data."""
import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

from ids_pipeline import supervised


class DeviceMatrix:
    """Input double: host conversion of features is forbidden."""
    def __init__(self, values):
        self.values = values
        self.shape = values.shape

    def __getitem__(self, index):
        return DeviceMatrix(self.values[index])

    def __array__(self, *args, **kwargs):
        raise AssertionError('GPU input must not be copied to CPU for inference')


class LimitedForest:
    def __init__(self, limit=4, classes=2):
        self.limit = limit
        self.classes_ = np.arange(classes)
        self.calls = []

    def _values(self, X, method):
        if X.shape[0] > self.limit:
            raise RuntimeError(f'Input size too large! Input should be at most {self.limit}.')
        self.calls.append((method, X.shape[0]))
        return X.values if isinstance(X, DeviceMatrix) else X

    def predict(self, X):
        values = self._values(X, 'predict')
        return (values[:, 0].astype(int) % len(self.classes_)).astype(np.int32)

    def predict_proba(self, X):
        values = self._values(X, 'predict_proba')
        raw = 1 + (values[:, :1] + np.arange(len(self.classes_))) % 7
        return raw / raw.sum(axis=1, keepdims=True)


def bundle(model):
    encoder = LabelEncoder().fit([str(i) for i in model.classes_])
    return dict(model=model, label_encoder=encoder)


@pytest.mark.parametrize('rows', [0, 1, 4, 5, 8, 11])
@pytest.mark.parametrize('classes', [2, 10, 30])
def test_all_rows_order_and_probabilities_match_one_call(monkeypatch, rows, classes):
    monkeypatch.setattr(supervised, 'GPU_PREDICTION_BATCH_ROWS', 4)
    values = np.arange(rows, dtype=np.float32)[:, None]
    original = values.copy()
    gpu = bundle(LimitedForest(classes=classes))
    cpu = bundle(LimitedForest(limit=100, classes=classes))
    if rows > 4:
        with pytest.raises(RuntimeError, match='Input size too large'):
            gpu['model'].predict(DeviceMatrix(values))
    for method in (supervised._predict_encoded, supervised._predict_decoded, supervised._predict_proba):
        actual = method(gpu, DeviceMatrix(values), backend='gpu')
        expected = method(cpu, values, backend='cpu')
        np.testing.assert_array_equal(actual, expected)
    sizes = [min(4, rows-start) for start in range(0, rows, 4)] or [0]
    assert gpu['model'].calls == [('predict', n) for n in sizes] * 2 + [('predict_proba', n) for n in sizes]
    np.testing.assert_array_equal(values, original)


def test_metrics_are_computed_globally_including_auc_ap_fpr(monkeypatch):
    monkeypatch.setattr(supervised, 'GPU_PREDICTION_BATCH_ROWS', 4)
    values = np.arange(11, dtype=np.float32)[:, None]
    labels = np.array(['0'] * 7 + ['1'] * 4)
    gpu = supervised._evaluate_split(bundle(LimitedForest()), DeviceMatrix(values), labels, split='val', backend='gpu')
    cpu = supervised._evaluate_split(bundle(LimitedForest(limit=100)), values, labels, split='val', backend='cpu')
    assert gpu == cpu
    assert all(gpu[key] is not None for key in ('roc_auc', 'pr_auc', 'fpr_at_tpr_95'))


@pytest.mark.parametrize('method', ['predict', 'predict_proba'])
def test_gpu_failure_is_not_hidden_or_replaced_with_partial_output(monkeypatch, method):
    monkeypatch.setattr(supervised, 'GPU_PREDICTION_BATCH_ROWS', 4)
    model = LimitedForest()
    original = getattr(model, method)
    def fail_second(X):
        if model.calls:
            raise RuntimeError('device failure')
        return original(X)
    setattr(model, method, fail_second)
    call = supervised._predict_encoded if method == 'predict' else supervised._predict_proba
    with pytest.raises(RuntimeError, match='device failure'):
        call(bundle(model), DeviceMatrix(np.arange(9)[:, None]), backend='gpu')


@pytest.mark.parametrize('bad', ['rows', 'columns', 'dtype'])
def test_malformed_batch_is_rejected(monkeypatch, bad):
    monkeypatch.setattr(supervised, 'GPU_PREDICTION_BATCH_ROWS', 4)
    model = LimitedForest()
    original = model.predict_proba
    def malformed(X):
        output = original(X)
        if len(model.calls) == 1:
            return output
        if bad == 'rows':
            return output[:-1]
        if bad == 'columns':
            return output[:, :1]
        return output.astype(np.float32)
    model.predict_proba = malformed
    with pytest.raises(ValueError, match='GPU predict_proba'):
        supervised._predict_proba(bundle(model), DeviceMatrix(np.arange(9)[:, None]), backend='gpu')


def test_cpu_predict_is_not_batched(monkeypatch):
    monkeypatch.setattr(supervised, 'GPU_PREDICTION_BATCH_ROWS', 4)
    model = LimitedForest(limit=100)
    supervised._predict_encoded(bundle(model), np.arange(11)[:, None], backend='cpu')
    assert model.calls == [('predict', 11)]
