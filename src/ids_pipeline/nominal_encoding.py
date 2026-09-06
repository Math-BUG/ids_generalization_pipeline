"""Shared train-fitted nominal representation for the low-cardinality conn_state."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class ConnStateOneHot(TransformerMixin, BaseEstimator):
    """One indicator per observed training state, never ordinal coordinates.

    Missing values use the training mode (lexical tie break); all-missing training
    uses the explicit local encoder fallback __MISSING__. Unknown states use an
    all-zero vector. No frequency grouping, dropping or validation vocabulary fit.
    Input must already have passed the semantic normalizer.
    """
    @staticmethod
    def _series(X):
        if isinstance(X, pd.DataFrame):
            if X.shape[1] != 1 or X.columns[0] != "conn_state":
                raise ValueError("ConnStateOneHot expects only conn_state")
            series = X.iloc[:, 0]
        else:
            arr = np.asarray(X, dtype=object)
            if arr.ndim != 2 or arr.shape[1] != 1:
                raise ValueError("ConnStateOneHot expects a single-column matrix")
            series = pd.Series(arr[:, 0])
        return series.astype("string")

    def fit(self, X, y=None):
        series = self._series(X)
        if len(series) == 0:
            raise ValueError("Cannot learn conn_state vocabulary from empty training")
        modes = series.dropna().mode()
        self.mode_ = str(modes.iloc[0]) if len(modes) else "__MISSING__"
        self.categories_ = np.asarray(sorted(set(series.fillna(self.mode_))), dtype=object)
        self.n_features_in_ = 1
        self.feature_names_in_ = np.asarray(["conn_state"], dtype=object)
        return self

    def category_indices(self, X):
        """Internal scatter indices only; never returned as model features."""
        check_is_fitted(self, ["categories_", "mode_"])
        series = self._series(X).fillna(self.mode_)
        return pd.Categorical(series, categories=self.categories_).codes.astype(np.int64)

    def transform(self, X):
        codes = self.category_indices(X)
        rows = np.flatnonzero(codes >= 0)
        return sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, codes[rows])),
                                 shape=(len(codes), len(self.categories_)), dtype=np.float32)

    def transform_gpu(self, X):
        import cupy as cp
        codes = self.category_indices(X)
        rows = np.flatnonzero(codes >= 0)
        # Allocate only the final low-cardinality GPU block; no dense host N*C copy.
        out = cp.zeros((len(codes), len(self.categories_)), dtype=cp.float32)
        out[cp.asarray(rows), cp.asarray(codes[rows])] = 1.0
        return out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "categories_")
        return np.asarray(["conn_state=" + json.dumps(c, ensure_ascii=False) for c in self.categories_], dtype=object)

    def metadata(self):
        check_is_fitted(self, "categories_")
        return {"encoding": "one_hot", "vocabulary": self.categories_.tolist(),
                "output_columns": self.get_feature_names_out().tolist(), "fit_split": "train",
                "missing": "training_mode", "mode": self.mode_, "all_missing_training_fallback": "__MISSING__",
                "unknown": "all_zero", "drop": None, "frequency_grouping": False,
                "indicator_scaling": "none"}
