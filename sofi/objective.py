"""Perturbation curves and the objective optimized by SOFI.

A ranking is assessed through two perturbation curves obtained after
cumulative marginalization. The MoRF curve removes the most relevant features
first and should collapse immediately, while the LeRF curve removes the least
relevant features first and should stay flat for as long as possible. The
degradation score is the integral between the two, following Schulz et al., and
SOFI maximizes it. The MoRF branch rewards sparsity, since a sparse explanation
concentrates the response of the model in a handful of features, and the LeRF
branch rewards correctness, since the features declared irrelevant must be the
ones whose removal leaves the response untouched.

The search works on the raw response of the model and on nothing else.
Classification records the probability assigned to the class predicted before
any perturbation, as in region perturbation. Regression records the mean
absolute deviation with respect to the unperturbed estimate, negated so that
both quantities fall as evidence is destroyed. No ground truth label takes part
in either computation, which keeps explanations free from the bias induced by
the error of the model.

Normalization never enters the optimization. It is applied afterwards, for
reporting and for drawing, and it follows the convention of the literature. The
response of the unperturbed input is placed at one and the response of the
fully perturbed input at zero. Both anchors are properties of the model and the
data rather than of a ranking, so every ranking evaluated on the same data
shares them, the reported score is the optimized score divided by a positive
constant, and no ordering can change. The anchors are read off a reference sweep that also
covers the fully perturbed input, so in the ordinary case they are exactly the
two responses the convention prescribes. Should the fully perturbed response
fail to be the lowest one, which happens when a feature is replaced with a
single statistic that lands the instance in a confident region, the anchors
widen to the extremes the sweep observed and a warning names the cause.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .encoding import FeatureSpace
from .marginalization import Marginalizer
from .model import ModelWrapper

__all__ = ["Objective", "RankingEvaluation", "curve_auc", "degradation_score"]

_EPSILON = 1e-12


def curve_auc(scores: Sequence[float]) -> float:
    """Area under a perturbation curve, divided by the number of steps.

    The division turns the area into the mean height of the curve, so datasets
    with different numbers of features stay comparable.
    """
    values = np.asarray(scores, dtype=float)
    if values.size < 2:
        return float(values.sum())
    area = np.trapezoid(values) if hasattr(np, "trapezoid") else np.trapz(values)
    return float(area / (values.size - 1))


def degradation_score(morf: Sequence[float], lerf: Sequence[float]) -> float:
    """Integral between the LeRF and the MoRF perturbation curves.

    Larger values denote explanations that are simultaneously sparser and more
    faithful. On the reported scale both areas belong to the unit interval, so
    the score runs from minus one to one and a sound ranking keeps it positive.
    """
    return curve_auc(lerf) - curve_auc(morf)


@dataclass
class RankingEvaluation:
    """Both raw curves of a ranking together with its degradation score."""

    order: List[int]
    morf: List[float] = field(repr=False)
    lerf: List[float] = field(repr=False)
    ds: float = 0.0


class Objective:
    """Evaluate feature rankings through cumulative marginalization."""

    def __init__(
        self,
        wrapper: ModelWrapper,
        space: FeatureSpace,
        marginalizer: Marginalizer,
        X_eval: pd.DataFrame,
        chunk_size: int = 20000,
    ):
        self.wrapper = wrapper
        self.space = space
        self.marginalizer = marginalizer
        self.chunk_size = int(chunk_size)
        self.n_calls_ = 0

        self.X_eval = X_eval.reset_index(drop=True)
        self.n_rows = len(self.X_eval)
        if self.n_rows == 0:
            raise ValueError("The data to be explained cannot be empty.")

        self.template = marginalizer.tile(self.X_eval)
        self.n_replicas = marginalizer.n_replicas

        if wrapper.task == "classification":
            proba = wrapper.predict_proba(space.transform(self.X_eval))
            self.n_calls_ += 1
            self.reference_ = np.argmax(proba, axis=1)
            self.raw_baseline_ = float(
                np.mean(proba[np.arange(self.n_rows), self.reference_])
            )
            self.baseline_ = self.raw_baseline_
        else:
            self.reference_ = wrapper.predict(space.transform(self.X_eval)).astype(float)
            self.n_calls_ += 1
            self.raw_baseline_ = float(np.mean(self.reference_))
            self.baseline_ = 0.0

        self._reference_tiled = np.tile(self.reference_, self.n_replicas)
        self._reference_sweep()

    # -------------------------------------------------------- reference sweep
    def _reference_sweep(self) -> None:
        """Marginalize every feature alone and then follow the greedy ordering.

        The sweep serves three purposes. It orders the features by the
        degradation each of them causes on its own, which is the greedy start
        of the search, it produces the response of the fully perturbed input,
        which anchors the floor of the reported scale, and it establishes
        whether the response reacts to marginalization at all.
        """
        alone = np.asarray(self._sweep(), dtype=float)
        self.single_feature_drops_ = self.baseline_ - alone
        self.greedy_order_ = [
            int(position)
            for position in np.argsort(-self.single_feature_drops_, kind="stable")
        ]
        self.reference_curve_ = self.curve(self.greedy_order_)
        self.full_score_ = float(self.reference_curve_[-1])

        observed = list(alone) + list(self.reference_curve_) + [self.baseline_]
        self.anchor_high_ = float(self.baseline_)
        self.anchor_low_ = float(min(observed))

        # the unperturbed response should be the highest one. A marginal excursion
        # above it is absorbed by the clipping, while a substantial one widens the
        # scale, so that a curve is never flattened against a false ceiling
        overshoot = float(max(observed)) - self.anchor_high_
        span = self.anchor_high_ - self.anchor_low_
        systematic = overshoot > max(0.1 * span, _EPSILON)

        if span <= _EPSILON:
            warnings.warn(
                "The response of the model never lowers when features are "
                "marginalized, so the degradation score carries no information. "
                "Check that the validation set holds instances the model handles "
                "well and that the marginalization values neutralize the features.",
                stacklevel=3,
            )
        elif systematic:
            self.anchor_high_ = float(max(observed))
            warnings.warn(
                "Marginalization leaves the model substantially more confident than the "
                "unperturbed input, so the response it reaches is not the most degraded "
                "one and the reported curves cannot start at one. This points at the "
                "marginalization values rather than at the ranking, since replacing a "
                "feature with one statistic can land the instance in a region the model "
                "reads with confidence. Use marginalization='background', which averages "
                "the response over a sample of the training data instead.",
                stacklevel=3,
            )

    def _sweep(self) -> List[float]:
        """Raw response after marginalizing each feature on its own."""
        states = []
        for position in range(self.space.n_features):
            state = self.template.copy()
            self.marginalizer.fill(state, self.space.feature_names[position], self.n_rows)
            states.append(state)
        return self._predict_states(states)

    # --------------------------------------------------------- normalization
    @property
    def anchors_(self) -> Tuple[float, float]:
        """Responses placed at one and at zero when a curve is reported."""
        return self.anchor_high_, self.anchor_low_

    def normalize(self, scores: Sequence[float]) -> List[float]:
        """Express a raw curve on the reported scale.

        The transformation is affine and shared by every ranking, so it serves
        reading and drawing alone. It never takes part in the search.
        """
        span = self.anchor_high_ - self.anchor_low_
        if span <= _EPSILON:
            return [1.0 for _ in scores]
        return [
            float(min(max((float(score) - self.anchor_low_) / span, 0.0), 1.0))
            for score in scores
        ]

    # --------------------------------------------------------------- scoring
    def _score_predictions(self, output: np.ndarray) -> float:
        """Raw response of the model over the instances being explained."""
        if self.wrapper.task == "classification":
            rows = np.arange(output.shape[0])
            return float(np.mean(output[rows, self._reference_tiled]))
        deviation = np.abs(np.asarray(output, dtype=float) - self._reference_tiled)
        return -float(np.mean(deviation))

    def _predict_states(self, states: List[pd.DataFrame]) -> List[float]:
        """Score a sequence of marginalized states with batched model queries."""
        if not states:
            return []

        rows_per_state = len(states[0])
        per_chunk = max(1, self.chunk_size // max(rows_per_state, 1))
        scores: List[float] = []

        for start in range(0, len(states), per_chunk):
            block = states[start : start + per_chunk]
            stacked = pd.concat(block, axis=0, ignore_index=True)
            model_input = self.space.transform(stacked)
            if self.wrapper.task == "classification":
                output = self.wrapper.predict_proba(model_input)
            else:
                output = self.wrapper.predict(model_input)
            self.n_calls_ += 1
            for position in range(len(block)):
                lower = position * rows_per_state
                upper = lower + rows_per_state
                scores.append(self._score_predictions(output[lower:upper]))
        return scores

    # ----------------------------------------------------------------- curve
    def curve(
        self,
        order: Sequence[int],
        prefix_length: int = 0,
        prefix_scores: Optional[Sequence[float]] = None,
    ) -> List[float]:
        """Raw perturbation curve of a marginalization order.

        A swap between two positions leaves every step before the first of them
        untouched, so those scores are inherited through ``prefix_scores``
        instead of being recomputed. The result is identical to a full
        evaluation while the number of model queries drops.
        """
        order = list(order)
        if prefix_scores is None or prefix_length <= 0:
            prefix_length = 0
            scores = [self.baseline_]
        else:
            prefix_length = min(prefix_length, len(order))
            scores = list(prefix_scores[: prefix_length + 1])

        state = self.template.copy()
        for position in order[:prefix_length]:
            self.marginalizer.fill(state, self.space.feature_names[position], self.n_rows)

        states = []
        for position in order[prefix_length:]:
            self.marginalizer.fill(state, self.space.feature_names[position], self.n_rows)
            states.append(state.copy())

        scores.extend(self._predict_states(states))
        return scores

    def evaluate(
        self,
        order: Sequence[int],
        incumbent: Optional[RankingEvaluation] = None,
        swap: Optional[Tuple[int, int]] = None,
    ) -> RankingEvaluation:
        """Both raw curves of a ranking, reusing the prefixes left by a swap.

        The LeRF curve follows the reversed ranking, so a swap between the
        positions ``i`` and ``j`` of the ranking occupies the positions
        ``n - 1 - j`` and ``n - 1 - i`` of the reversed one. The two prefixes
        are therefore inherited from opposite ends of the incumbent.
        """
        order = list(order)
        n_features = len(order)
        reversed_order = order[::-1]

        if incumbent is None or swap is None:
            morf = self.curve(order)
            lerf = self.curve(reversed_order)
        else:
            first, second = int(swap[0]), int(swap[1])
            morf = self.curve(order, first, incumbent.morf)
            lerf = self.curve(reversed_order, n_features - 1 - second, incumbent.lerf)

        return RankingEvaluation(
            order=order, morf=morf, lerf=lerf, ds=degradation_score(morf, lerf)
        )

    # ---------------------------------------------------------------- priors
    def single_feature_drops(self) -> np.ndarray:
        """Degradation caused when each feature is marginalized on its own."""
        return self.single_feature_drops_
