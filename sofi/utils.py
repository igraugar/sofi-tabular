"""Helpers that support the recommended usage protocol."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .model import ModelWrapper

__all__ = ["select_reliable_instances"]


def select_reliable_instances(
    model,
    X,
    y,
    encoder=None,
    task: str = "auto",
    quantile: float = 0.5,
    tolerance: Optional[float] = None,
    return_mask: bool = False,
):
    """Keep the instances on which the model performs well.

    SOFI assumes that the response of the model deteriorates as features are
    marginalized, and that premise only holds for instances that were correctly
    classified or estimated with a small error. The selection is left outside
    the explainer so that the user stays in control of the validation set.

    Parameters
    ----------
    model : object
        Fitted estimator, optionally a pipeline that carries its encoder.
    X : DataFrame
        Validation data expressed in the space consumed by the estimator or,
        when ``encoder`` is supplied, in the original feature space.
    y : array like
        Ground truth targets of the validation data.
    encoder : transformer, optional
        Fitted encoder applied before the estimator receives the data.
    task : {"auto", "classification", "regression"}, default="auto"
        Learning task, resolved from the estimator by default.
    quantile : float, default=0.5
        Fraction of regression instances retained, ordered by absolute error.
    tolerance : float, optional
        Absolute error threshold that supersedes ``quantile`` when supplied.
    return_mask : bool, default=False
        Whether the boolean mask is returned together with the subset.

    Returns
    -------
    DataFrame or tuple
        Selected instances and, on request, the mask that produced them.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a DataFrame so that the subset keeps its columns.")

    wrapper = ModelWrapper(
        model,
        task=task,
        output_feature_names=None
        if encoder is None
        else list(encoder.get_feature_names_out()),
    )
    data = X if encoder is None else encoder.transform(X)
    predictions = wrapper.predict(data)
    truth = np.asarray(y)

    if wrapper.task == "classification":
        mask = predictions == truth
    else:
        errors = np.abs(np.asarray(predictions, dtype=float) - truth.astype(float))
        if tolerance is None:
            threshold = float(np.quantile(errors, quantile))
        else:
            threshold = float(tolerance)
        mask = errors <= threshold

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError(
            "No instance satisfied the selection criterion, so the validation set "
            "would be empty."
        )
    subset = X.loc[mask]
    return (subset, mask) if return_mask else subset
