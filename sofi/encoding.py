"""Resolution of the space in which features are marginalized.

SOFI perturbs whole features, never isolated columns of a one-hot block. The
classes below make that guarantee explicit through a mapping from a logical
feature to the set of columns that represent it, together with the
transformation that turns a perturbed frame into model input.

Three usage paths are supported.

1. A ``Pipeline`` that carries the encoder and the estimator. Perturbation
   happens in the original feature space and the encoding never leaks into the
   explainer. This path is the recommended one.
2. A fitted encoder passed apart from the estimator. Perturbation still happens
   in the original space, and SOFI applies the encoder before each prediction.
3. Data that is already encoded, together with a ``feature_groups`` mapping
   that states which columns belong to the same logical feature.
"""

from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["FeatureSpace", "build_feature_space", "feature_groups_from_encoder"]


class FeatureSpace:
    """Mapping between logical features and the columns SOFI perturbs.

    Attributes
    ----------
    feature_names : list of str
        Names of the logical features, in the order used internally.
    groups : dict
        Maps every logical feature onto the list of columns that represent it.
    transform : callable
        Turns a perturbed frame into the container consumed by the estimator.
    output_feature_names : list of str or None
        Column names produced by ``transform`` when it returns a bare array.
    """

    def __init__(
        self,
        groups: Dict[str, List],
        transform: Optional[Callable] = None,
        output_feature_names: Optional[Sequence] = None,
        path: str = "identity",
    ):
        self.groups = {str(k): list(v) for k, v in groups.items()}
        self.feature_names = list(self.groups.keys())
        self._transform = transform
        self.output_feature_names = (
            None if output_feature_names is None else list(output_feature_names)
        )
        self.path = path

    def __len__(self) -> int:
        return len(self.feature_names)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def columns_of(self, feature) -> List:
        if isinstance(feature, (int, np.integer)):
            feature = self.feature_names[int(feature)]
        return self.groups[feature]

    def transform(self, X: pd.DataFrame):
        if self._transform is None:
            return X
        return self._transform(X)

    def describe(self) -> pd.DataFrame:
        """Return one row per logical feature with its block width."""
        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "n_columns": [len(self.groups[f]) for f in self.feature_names],
                "columns": [list(self.groups[f]) for f in self.feature_names],
            }
        )


def _is_pipeline(model) -> bool:
    try:
        from sklearn.pipeline import Pipeline
    except ImportError:  # pragma: no cover
        return False
    return isinstance(model, Pipeline)


def _one_hot_widths(transformer, columns) -> Optional[List[int]]:
    """Number of output columns produced per input column by a one-hot encoder."""
    widths = getattr(transformer, "_n_features_outs", None)
    if isinstance(widths, list) and len(widths) == len(columns):
        return [int(w) for w in widths]

    categories = getattr(transformer, "categories_", None)
    if categories is None or len(categories) != len(columns):
        return None

    drop_idx = getattr(transformer, "drop_idx_", None)
    widths = []
    for position, values in enumerate(categories):
        width = len(values)
        if drop_idx is not None and drop_idx[position] is not None:
            width -= 1
        widths.append(int(width))
    return widths


def feature_groups_from_encoder(encoder) -> Dict[str, List[str]]:
    """Recover logical feature blocks from a fitted ``ColumnTransformer``.

    The mapping is derived from the transformer structure rather than from
    column name patterns, so renaming conventions such as
    ``verbose_feature_names_out`` do not affect the result. Use it when the
    estimator was fitted on already encoded data.
    """
    if not hasattr(encoder, "transformers_"):
        raise TypeError(
            "feature_groups_from_encoder expects a fitted ColumnTransformer."
        )

    output_names = list(encoder.get_feature_names_out())
    groups: Dict[str, List[str]] = {}
    cursor = 0

    for name, transformer, columns in encoder.transformers_:
        if transformer == "drop" or transformer is None:
            continue
        if isinstance(columns, str):
            columns = [columns]
        columns = list(columns)

        if transformer == "passthrough":
            widths = [1] * len(columns)
        else:
            widths = _one_hot_widths(transformer, columns)
            if widths is None:
                try:
                    produced = list(transformer.get_feature_names_out(columns))
                except Exception:  # pragma: no cover - permissive fallback
                    produced = []
                if len(produced) == len(columns):
                    widths = [1] * len(columns)
                else:
                    warnings.warn(
                        f"The outputs of transformer '{name}' cannot be attributed to "
                        "individual input columns, so the whole block is treated as a "
                        "single logical feature.",
                        stacklevel=2,
                    )
                    total = len(produced) if produced else len(output_names) - cursor
                    groups[name] = output_names[cursor : cursor + total]
                    cursor += total
                    continue

        for column, width in zip(columns, widths):
            groups[str(column)] = output_names[cursor : cursor + width]
            cursor += width

    if cursor != len(output_names):
        warnings.warn(
            "Some encoded columns were not attributed to a logical feature. "
            "Pass feature_groups explicitly if the mapping matters.",
            stacklevel=2,
        )
    return groups


def build_feature_space(
    model,
    X: pd.DataFrame,
    encoder=None,
    feature_groups: Optional[Dict] = None,
) -> FeatureSpace:
    """Select the marginalization space that matches the supplied objects."""
    columns = list(X.columns)

    if feature_groups is not None:
        if feature_groups == "auto":
            if encoder is None:
                raise ValueError(
                    "feature_groups='auto' requires the fitted encoder that produced "
                    "the columns of X."
                )
            feature_groups = feature_groups_from_encoder(encoder)

        groups: Dict[str, List] = {}
        assigned = set()
        for feature, block in feature_groups.items():
            block = [block] if isinstance(block, str) else list(block)
            unknown = [c for c in block if c not in X.columns]
            if unknown:
                raise ValueError(
                    f"Feature group '{feature}' refers to columns that are absent "
                    f"from X. First unknown column, {unknown[0]}."
                )
            overlap = assigned.intersection(block)
            if overlap:
                raise ValueError(
                    f"Column '{sorted(overlap)[0]}' belongs to more than one feature "
                    "group, which makes the marginalization order ambiguous."
                )
            groups[str(feature)] = block
            assigned.update(block)

        for column in columns:
            if column not in assigned:
                groups[str(column)] = [column]
        return FeatureSpace(groups, transform=None, path="feature_groups")

    if encoder is not None:
        if not hasattr(encoder, "transform"):
            raise TypeError("The encoder must be fitted and expose a transform method.")
        try:
            output_names = list(encoder.get_feature_names_out())
        except Exception:  # pragma: no cover - encoders without name support
            output_names = None
        groups = {str(c): [c] for c in columns}
        return FeatureSpace(
            groups,
            transform=encoder.transform,
            output_feature_names=output_names,
            path="encoder",
        )

    groups = {str(c): [c] for c in columns}
    path = "pipeline" if _is_pipeline(model) else "identity"
    return FeatureSpace(groups, transform=None, path=path)
