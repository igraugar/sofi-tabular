"""Adapters that decouple SOFI from the estimator interface and from the
container type (``pandas`` versus ``numpy``) used to fit the model."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

try:  # pragma: no cover - scikit-learn is an optional runtime dependency
    from sklearn.base import is_classifier, is_regressor
except ImportError:  # pragma: no cover
    is_classifier = is_regressor = None


__all__ = ["ModelWrapper"]


def _first_attr(estimator, name):
    """Return ``name`` from the estimator or from the last step of a pipeline."""
    if hasattr(estimator, name):
        return getattr(estimator, name)
    steps = getattr(estimator, "steps", None)
    if steps:
        last = steps[-1][1]
        if hasattr(last, name):
            return getattr(last, name)
    return None


class ModelWrapper:
    """Uniform prediction interface for classifiers and regressors.

    The wrapper solves two recurrent problems. It infers the learning task
    without asking the user, and it converts whatever array or frame SOFI
    produces into the container the estimator was fitted with. A model fitted
    on a ``DataFrame`` therefore keeps receiving named columns in the original
    order, while a model fitted on a plain array keeps receiving an array.

    Parameters
    ----------
    model : object
        Fitted estimator exposing ``predict`` and, for classification,
        ``predict_proba``.
    task : {"auto", "classification", "regression"}, default="auto"
        Learning task. The default resolves the task from the estimator.
    predict_fn, proba_fn : callable, optional
        Escape hatches for estimators outside the scikit-learn ecosystem. Both
        receive the model input already adapted and must return arrays.
    output_feature_names : sequence of str, optional
        Names of the columns produced by the encoding stage. They are used when
        the estimator requires named columns but SOFI holds a bare array.
    """

    def __init__(
        self,
        model,
        task: str = "auto",
        predict_fn: Optional[Callable] = None,
        proba_fn: Optional[Callable] = None,
        output_feature_names=None,
    ):
        self.model = model
        self._predict_fn = predict_fn
        self._proba_fn = proba_fn
        self.task = self._resolve_task(task)

        names = _first_attr(model, "feature_names_in_")
        self.expected_names_ = None if names is None else list(names)
        if self.expected_names_ is None and output_feature_names is not None:
            self.output_feature_names_ = list(output_feature_names)
        else:
            self.output_feature_names_ = (
                None if output_feature_names is None else list(output_feature_names)
            )

        self.classes_ = _first_attr(model, "classes_")
        if self.task == "classification" and self.classes_ is not None:
            self._class_position = {c: i for i, c in enumerate(self.classes_)}
        else:
            self._class_position = {}

    # ------------------------------------------------------------------ task
    def _resolve_task(self, task: str) -> str:
        if task in ("classification", "regression"):
            return task
        if task != "auto":
            raise ValueError(
                "task must be one of 'auto', 'classification' or 'regression'."
            )
        if self._proba_fn is not None:
            return "classification"
        if is_classifier is not None and is_classifier(self.model):
            return "classification"
        if is_regressor is not None and is_regressor(self.model):
            return "regression"
        if hasattr(self.model, "predict_proba"):
            return "classification"
        return "regression"

    # ------------------------------------------------------------- adaptation
    def adapt(self, data):
        """Return ``data`` in the container the estimator expects."""
        if self.expected_names_ is None:
            if isinstance(data, pd.DataFrame):
                return data.to_numpy()
            return np.asarray(data)

        if isinstance(data, pd.DataFrame):
            if list(data.columns) == self.expected_names_:
                return data
            missing = [c for c in self.expected_names_ if c not in data.columns]
            if missing:
                raise ValueError(
                    "The estimator was fitted with columns that are absent from the "
                    f"data handled by SOFI. First missing column, {missing[0]}."
                )
            return data.loc[:, self.expected_names_]

        array = np.asarray(data)
        if array.shape[1] != len(self.expected_names_):
            raise ValueError(
                f"The estimator expects {len(self.expected_names_)} columns while "
                f"the encoded data has {array.shape[1]}."
            )
        columns = self.output_feature_names_ or self.expected_names_
        frame = pd.DataFrame(array, columns=list(columns))
        return frame.loc[:, self.expected_names_]

    # ------------------------------------------------------------ prediction
    def predict(self, data) -> np.ndarray:
        adapted = self.adapt(data)
        if self._predict_fn is not None:
            return np.asarray(self._predict_fn(adapted))
        return np.asarray(self.model.predict(adapted))

    def predict_proba(self, data) -> np.ndarray:
        adapted = self.adapt(data)
        if self._proba_fn is not None:
            return np.asarray(self._proba_fn(adapted))
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(
                "The classifier does not expose predict_proba, so SOFI cannot "
                "measure probability degradation. Provide proba_fn instead."
            )
        return np.asarray(self.model.predict_proba(adapted))

    def class_positions(self, labels) -> np.ndarray:
        """Map class labels onto column positions of the probability matrix."""
        if not self._class_position:
            return np.asarray(labels, dtype=int)
        return np.array([self._class_position[label] for label in labels], dtype=int)
