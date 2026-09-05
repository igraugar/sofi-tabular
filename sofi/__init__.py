"""SOFI, Sparseness Optimized Feature Importance.

A model agnostic post hoc explainer that searches for a ranking of features
whose cumulative marginalization degrades the response of the model as fast as
possible. The objective is the degradation score, namely the area between the
LeRF and the MoRF perturbation curves, which rewards sparsity and correctness
at once. The search works on the raw response of the model, and normalization
is applied afterwards, for reporting and for drawing alone.
Classification and regression are both supported, and the search is a hill
climbing procedure whose operator swaps two randomly selected ranking
positions.
"""

from .encoding import FeatureSpace, feature_groups_from_encoder
from .explainer import SOFIExplainer
from .explanation import SOFIExplanation, aggregate_explanations, noise_onset
from .marginalization import Marginalizer
from .objective import Objective, RankingEvaluation, curve_auc, degradation_score
from .plotting import (
    LERF_COLOR,
    MORF_COLOR,
    plot_degradation_curve,
    plot_explanation_grid,
    set_curve_colors,
    set_plot_style,
)
from .search import hill_climbing
from .utils import select_reliable_instances

__version__ = "1.0.0"

__all__ = [
    "SOFIExplainer",
    "SOFIExplanation",
    "FeatureSpace",
    "Marginalizer",
    "Objective",
    "aggregate_explanations",
    "RankingEvaluation",
    "curve_auc",
    "degradation_score",
    "feature_groups_from_encoder",
    "hill_climbing",
    "noise_onset",
    "plot_degradation_curve",
    "set_curve_colors",
    "plot_explanation_grid",
    "set_plot_style",
    "select_reliable_instances",
    "__version__",
]
