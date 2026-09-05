"""Container returned by the explainer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .objective import curve_auc

__all__ = ["SOFIExplanation", "aggregate_explanations", "noise_onset"]

_EPSILON = 1e-12


def noise_onset(curve: Sequence[float], tolerance: float = 0.02) -> Optional[int]:
    """Index from which a perturbation curve grows back systematically.

    Once the informative features are gone, marginalizing what remains may push
    the response towards its unperturbed state, since the values substituted are
    the most typical ones of the training data. A curve that climbs after its
    minimum therefore signals features that carry no evidence.

    Parameters
    ----------
    curve : sequence of float
        Perturbation curve, normally the reported MoRF curve.
    tolerance : float, default=0.02
        Rise required before the region is declared, expressed on the scale of
        the curve. Small oscillations are left unreported.

    Returns
    -------
    int or None
        Index of the minimum of the curve, which equals the number of features
        that precede the region, and ``None`` when no systematic rise occurs.
    """
    values = np.asarray(curve, dtype=float)
    if values.size < 3:
        return None
    boundary = int(np.argmin(values))
    if boundary >= values.size - 1:
        return None
    if float(values[-1] - values[boundary]) < float(tolerance):
        return None
    return boundary


@dataclass
class SOFIExplanation:
    """Result of one SOFI run.

    The curves are kept twice. The raw form holds the response of the model,
    which is what the search optimized, and the reported form places the
    unperturbed response at one and the fully perturbed one at zero, following
    the convention of the perturbation curve literature. Both anchors belong to
    the model and the data rather than to a ranking, so the reported score is
    the optimized score divided by a positive constant.

    Attributes
    ----------
    feature_names : list of str
        Logical features in the order used internally by the explainer.
    order : list of int
        Positions of the features sorted from the most to the least important.
    morf_scores, lerf_scores : list of float
        Reported curves, confined to the unit interval and starting at one.
    ds : float
        Degradation score on the reported scale, namely the integral between
        the two curves.
    ds_raw : float
        The same integral on the raw response of the model, which is the
        quantity the search maximizes.
    task : str
        Either ``classification`` or ``regression``.
    mode : str
        Either ``local`` or ``global``.
    """

    feature_names: List[str]
    order: List[int]
    morf_scores_raw: List[float]
    lerf_scores_raw: List[float]
    ds_raw: float
    task: str
    mode: str
    anchor_high: float = 1.0
    anchor_low: float = 0.0
    raw_baseline: Optional[float] = None
    history: List[Dict] = field(default_factory=list, repr=False)
    instance_index: Optional[object] = None
    n_instances: int = 1
    n_iterations: int = 0
    n_restarts_used: int = 0
    n_evaluations: int = 0
    n_model_calls: int = 0
    noise_tolerance: float = 0.02

    # ------------------------------------------------------------- reporting
    def _report(self, scores: Sequence[float]) -> List[float]:
        span = self.anchor_high - self.anchor_low
        if span <= _EPSILON:
            return [1.0 for _ in scores]
        return [
            float(min(max((float(score) - self.anchor_low) / span, 0.0), 1.0))
            for score in scores
        ]

    @property
    def morf_scores(self) -> List[float]:
        """Reported curve of the ranking itself."""
        return self._report(self.morf_scores_raw)

    @property
    def lerf_scores(self) -> List[float]:
        """Reported curve of the reversed ranking."""
        return self._report(self.lerf_scores_raw)

    @property
    def scores(self) -> List[float]:
        """Alias of the reported MoRF curve."""
        return self.morf_scores

    @property
    def ds(self) -> float:
        """Degradation score on the reported scale."""
        return curve_auc(self.lerf_scores) - curve_auc(self.morf_scores)

    # -------------------------------------------------------------- accessors
    @property
    def ranking(self) -> List[str]:
        """Feature names sorted from the most to the least important."""
        return [self.feature_names[position] for position in self.order]

    @property
    def drops(self) -> np.ndarray:
        """Degradation attributable to each step of the MoRF curve."""
        return -np.diff(np.asarray(self.morf_scores, dtype=float))

    @property
    def auc_morf(self) -> float:
        """Mean height of the reported MoRF curve."""
        return curve_auc(self.morf_scores)

    @property
    def auc_lerf(self) -> float:
        """Mean height of the reported LeRF curve."""
        return curve_auc(self.lerf_scores)

    @property
    def importances(self) -> pd.Series:
        """Rank based importance in the unit interval, one value per feature."""
        n = len(self.feature_names)
        values = np.zeros(n, dtype=float)
        for rank, position in enumerate(self.order):
            values[position] = (n - rank) / n
        return pd.Series(values, index=self.feature_names, name="importance")

    # ----------------------------------------------------------- noise region
    @property
    def noise_onset(self) -> Optional[int]:
        """Number of features that precede the region where fidelity recovers."""
        return noise_onset(self.morf_scores, self.noise_tolerance)

    @property
    def noise_features(self) -> List[str]:
        """Tail of the ranking whose marginalization restores the response."""
        boundary = self.noise_onset
        return [] if boundary is None else self.ranking[boundary:]

    # ------------------------------------------------------------- statistics
    @property
    def statistics(self) -> Dict[str, object]:
        """Every figure of the run gathered in one dictionary."""
        return {
            "task": self.task,
            "mode": self.mode,
            "n_features": len(self.feature_names),
            "n_instances": self.n_instances,
            "degradation_score": self.ds,
            "degradation_score_raw": self.ds_raw,
            "auc_morf": self.auc_morf,
            "auc_lerf": self.auc_lerf,
            "raw_baseline": self.raw_baseline,
            "anchor_high": self.anchor_high,
            "anchor_low": self.anchor_low,
            "noise_onset": self.noise_onset,
            "n_noise_features": len(self.noise_features),
            "n_iterations": self.n_iterations,
            "n_restarts_used": self.n_restarts_used,
            "n_evaluations": self.n_evaluations,
            "n_model_calls": self.n_model_calls,
        }

    def to_frame(self) -> pd.DataFrame:
        """Ranking, per step degradation and rank based importance."""
        importances = self.importances
        return pd.DataFrame(
            {
                "rank": np.arange(1, len(self.order) + 1),
                "feature": self.ranking,
                "morf_score_after": np.asarray(self.morf_scores[1:], dtype=float),
                "drop": self.drops,
                "importance": [importances[name] for name in self.ranking],
            }
        )

    def top(self, k: int = 10) -> List[str]:
        """First ``k`` features of the ranking."""
        return self.ranking[: int(k)]

    def summary(self, top_k: Optional[int] = None) -> str:
        """Report of the run, meant to be printed to the console.

        Parameters
        ----------
        top_k : int, optional
            Number of ranked features listed. Every feature is listed by
            default, and the count of those left out closes the line otherwise.
        """
        listed = self.ranking if top_k is None else self.ranking[: int(top_k)]
        listing = ", ".join(
            f"{position + 1}. {name}" for position, name in enumerate(listed)
        )
        remaining = len(self.order) - len(listed)
        if remaining > 0:
            listing += f", and {remaining} more"

        boundary = self.noise_onset
        if boundary is None:
            noise = "none detected"
        else:
            noise = f"from step {boundary}, {', '.join(self.noise_features)}"

        rows = [
            ("task", self.task),
            ("mode", self.mode),
            ("instances", str(self.n_instances)),
            ("features", str(len(self.feature_names))),
            ("degradation score", f"{self.ds:.4f}"),
            ("area under MoRF", f"{self.auc_morf:.4f}"),
            ("area under LeRF", f"{self.auc_lerf:.4f}"),
        ]
        if self.raw_baseline is not None:
            label = (
                "unperturbed probability"
                if self.task == "classification"
                else "unperturbed estimate"
            )
            rows.append((label, f"{self.raw_baseline:.4f}"))
        rows += [
            ("noise region", noise),
            (
                "search",
                f"{self.n_iterations} iterations, {self.n_restarts_used} restarts, "
                f"{self.n_evaluations} evaluations, {self.n_model_calls} model queries",
            ),
            ("ranking", listing),
        ]
        width = max(len(label) for label, _ in rows)
        lines = ["SOFI explanation"]
        lines += [f"  {label.ljust(width)}   {value}" for label, value in rows]
        return "\n".join(lines)

    def plot(self, **kwargs):
        """Draw both perturbation curves. Requires ``matplotlib``."""
        from .plotting import plot_degradation_curve

        return plot_degradation_curve(self, **kwargs)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        head = ", ".join(self.top(5))
        return (
            f"SOFIExplanation(task={self.task}, mode={self.mode}, "
            f"DS={self.ds:.3f}, top=[{head}])"
        )


def aggregate_explanations(
    explanations: Sequence[SOFIExplanation],
) -> pd.DataFrame:
    """Summarize a collection of local explanations.

    The mean rank orders the features according to their average position,
    while the frequency column reports how often a feature reaches the first
    three positions of an instance level ranking.
    """
    if len(explanations) == 0:
        raise ValueError("At least one explanation is required.")

    names = explanations[0].feature_names
    ranks = np.zeros((len(explanations), len(names)), dtype=float)
    for row, explanation in enumerate(explanations):
        for rank, position in enumerate(explanation.order):
            ranks[row, position] = rank + 1

    summary = pd.DataFrame(
        {
            "feature": names,
            "mean_rank": ranks.mean(axis=0),
            "std_rank": ranks.std(axis=0),
            "top3_frequency": (ranks <= 3).mean(axis=0),
        }
    )
    return summary.sort_values("mean_rank").reset_index(drop=True)
