"""Estimation and application of marginalization values.

Marginalizing a feature means replacing its columns with values that carry no
instance level information. Two variants are available. The constant variant
substitutes a single statistic estimated on the training data, which keeps the
number of model queries equal to the number of features. The background variant
approximates an expectation over the training distribution, replacing the
feature with values drawn from a fixed background sample.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .encoding import FeatureSpace

__all__ = ["Marginalizer"]

_NUMERIC_STRATEGIES = ("mean", "median", "mode")
_INTEGER_STRATEGIES = ("mode", "median", "mean")
_CATEGORICAL_STRATEGIES = ("mode",)


def _mode_of(series: pd.Series):
    modes = series.mode(dropna=True)
    if len(modes) == 0:
        return series.iloc[0]
    return modes.iloc[0]


def _cast_to_column_dtype(value, series: pd.Series, column):
    """Return ``value`` expressed in the dtype of ``series`` when possible."""
    dtype = series.dtype
    if isinstance(value, float) and pd.api.types.is_integer_dtype(dtype):
        if not float(value).is_integer():
            warnings.warn(
                f"The marginalization value of the integer column '{column}' was "
                "rounded to preserve the original dtype. Use integer_strategy="
                "'mean' on a float column if the exact value matters.",
                stacklevel=3,
            )
        return type(series.iloc[0])(round(value))
    if pd.api.types.is_float_dtype(dtype):
        return float(value)
    return value


class Marginalizer:
    """Compute and apply the values that neutralize a logical feature.

    Parameters
    ----------
    space : FeatureSpace
        Description of the columns that form every logical feature.
    X_train : DataFrame
        Training data, which is the only source of marginalization values. The
        validation data explained afterwards never contributes statistics.
    strategy : {"constant", "background"}, default="constant"
        Marginalization variant.
    numeric_strategy : {"mean", "median", "mode"}, default="mean"
        Statistic used for floating point columns.
    integer_strategy : {"mode", "median", "mean"}, default="mode"
        Statistic used for integer columns. The default preserves the dtype.
    categorical_strategy : {"mode"}, default="mode"
        Statistic used for nominal, boolean and categorical columns.
    n_background : int, default=25
        Size of the background sample of the background variant.
    random_state : int or Generator, optional
        Controls the background sample, which is drawn once and then reused, so
        that competing rankings are compared under identical conditions.
    """

    def __init__(
        self,
        space: FeatureSpace,
        X_train: pd.DataFrame,
        strategy: str = "background",
        numeric_strategy: str = "mean",
        integer_strategy: str = "mode",
        categorical_strategy: str = "mode",
        n_background: int = 25,
        random_state=None,
    ):
        if strategy not in ("constant", "background"):
            raise ValueError("strategy must be either 'constant' or 'background'.")
        if numeric_strategy not in _NUMERIC_STRATEGIES:
            raise ValueError(f"numeric_strategy must be one of {_NUMERIC_STRATEGIES}.")
        if integer_strategy not in _INTEGER_STRATEGIES:
            raise ValueError(f"integer_strategy must be one of {_INTEGER_STRATEGIES}.")
        if categorical_strategy not in _CATEGORICAL_STRATEGIES:
            raise ValueError(
                f"categorical_strategy must be one of {_CATEGORICAL_STRATEGIES}."
            )
        if strategy == "background" and n_background < 1:
            raise ValueError("n_background must be a positive integer.")

        self.space = space
        self.strategy = strategy
        self.numeric_strategy = numeric_strategy
        self.integer_strategy = integer_strategy
        self.categorical_strategy = categorical_strategy
        self.n_background = int(n_background) if strategy == "background" else 1
        self.rng = np.random.default_rng(random_state)

        self.values_: Dict[str, Dict] = {}
        self.background_: Optional[pd.DataFrame] = None
        self._fit(X_train)

    # ------------------------------------------------------------------- fit
    def _fit(self, X_train: pd.DataFrame) -> None:
        if len(X_train) == 0:
            raise ValueError("The training data used by SOFI cannot be empty.")

        for feature, columns in self.space.groups.items():
            if len(columns) > 1:
                self.values_[feature] = self._joint_mode(X_train, columns)
            else:
                column = columns[0]
                series = X_train[column]
                value = self._column_value(series)
                self.values_[feature] = {
                    column: _cast_to_column_dtype(value, series, column)
                }

        if self.strategy == "background":
            size = min(self.n_background, len(X_train))
            positions = self.rng.choice(len(X_train), size=size, replace=False)
            self.background_ = X_train.iloc[np.sort(positions)].reset_index(drop=True)
            self.n_background = size

    def _column_value(self, series: pd.Series):
        dtype = series.dtype
        if pd.api.types.is_bool_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
            return _mode_of(series)
        if pd.api.types.is_integer_dtype(dtype):
            strategy = self.integer_strategy
        elif pd.api.types.is_float_dtype(dtype):
            strategy = self.numeric_strategy
        else:
            return _mode_of(series)

        if strategy == "mean":
            return float(series.mean())
        if strategy == "median":
            return float(series.median())
        return _mode_of(series)

    @staticmethod
    def _joint_mode(X_train: pd.DataFrame, columns: List) -> Dict:
        """Most frequent joint pattern of a multi column feature.

        A one-hot block is replaced as a whole, so the substituted pattern is
        always a valid category rather than an average of indicator columns.
        """
        block = X_train.loc[:, columns]
        counts = block.value_counts(dropna=False)
        pattern = counts.index[0]
        if not isinstance(pattern, tuple):
            pattern = (pattern,)
        return dict(zip(columns, pattern))

    # ----------------------------------------------------------------- apply
    @property
    def n_replicas(self) -> int:
        """Number of copies of the validation data required per evaluation."""
        return self.n_background if self.strategy == "background" else 1

    def tile(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replicate the validation data once per background draw."""
        if self.strategy == "constant":
            return X.reset_index(drop=True)
        replicas = [X.reset_index(drop=True)] * self.n_background
        return pd.concat(replicas, axis=0, ignore_index=True)

    def fill(self, X_work: pd.DataFrame, feature: str, n_rows: int) -> None:
        """Marginalize ``feature`` in place on a tiled working frame.

        ``n_rows`` is the number of validation instances, so that the tiled
        frame holds ``n_rows`` times the number of background draws.
        """
        columns = self.space.columns_of(feature)
        constant = self.strategy == "constant"
        values = self.values_[feature] if constant else None

        for column in columns:
            dtype = X_work[column].dtype
            if constant:
                X_work[column] = values[column]
            else:
                draws = self.background_[column].to_numpy()
                X_work[column] = np.repeat(draws, n_rows)
            if X_work[column].dtype != dtype:
                try:
                    X_work[column] = X_work[column].astype(dtype)
                except (TypeError, ValueError):  # pragma: no cover
                    pass
