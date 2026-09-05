"""Hill climbing over the space of feature rankings.

The local search operator swaps two randomly selected positions of the current
ranking. A candidate is accepted when it raises the degradation score, namely
the area between the LeRF and the MoRF curves. The search stops after a budget
of iterations or once the patience expires, and it can restart from a fresh
random ranking instead of stopping.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

__all__ = ["hill_climbing"]

_TOLERANCE = 1e-12


def _progress_bar(total: int, enabled: bool, description: str):
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - progress reporting is optional
        return None
    return tqdm(total=total, desc=description, leave=False)


def hill_climbing(
    n_features: int,
    evaluate: Callable,
    initial_order: Sequence[int],
    max_iterations: int = 200,
    patience: Optional[int] = None,
    n_restarts: int = 0,
    accept_equal: bool = False,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False,
    description: str = "SOFI hill climbing",
) -> Dict:
    """Maximize the degradation score over the space of feature rankings.

    Parameters
    ----------
    n_features : int
        Number of logical features, which equals the length of a ranking.
    evaluate : callable
        Receives a ranking and, optionally, the evaluation of the incumbent
        together with the pair of swapped positions. It returns an object
        exposing the two curves and the attribute ``ds``.
    initial_order : sequence of int
        Ranking used to start the first climb.
    max_iterations : int, default=200
        Upper bound on the number of proposed swaps across all restarts.
    patience : int, optional
        Number of consecutive swaps without improvement that triggers a restart
        or, once the restarts are exhausted, the end of the search.
    n_restarts : int, default=0
        Number of restarts from a random ranking allowed after the patience
        expires.
    accept_equal : bool, default=False
        Whether swaps that leave the score unchanged are accepted, which lets
        the search drift along plateaus.
    rng : Generator, optional
        Source of randomness of the swap operator and of the restarts.
    verbose : bool, default=False
        Whether a progress bar is displayed.

    Returns
    -------
    dict
        Best evaluation found, the search history and a few counters that
        describe the run.
    """
    if n_features < 2:
        evaluation = evaluate(list(initial_order))
        return {
            "evaluation": evaluation,
            "history": [],
            "n_iterations": 0,
            "n_restarts_used": 0,
            "n_evaluations": 1,
        }

    rng = np.random.default_rng() if rng is None else rng
    patience = max_iterations if patience is None else int(patience)

    current = evaluate(list(initial_order))
    cache = {tuple(current.order): current}
    best = current

    history: List[Dict] = []
    iteration = 0
    stagnation = 0
    restarts_used = 0
    evaluations = 1

    bar = _progress_bar(max_iterations, verbose, description)
    try:
        while iteration < max_iterations:
            first, second = sorted(rng.choice(n_features, size=2, replace=False))
            candidate_order = list(current.order)
            candidate_order[first], candidate_order[second] = (
                candidate_order[second],
                candidate_order[first],
            )
            key = tuple(candidate_order)

            if key in cache:
                candidate = cache[key]
            else:
                candidate = evaluate(
                    candidate_order, current, (int(first), int(second))
                )
                cache[key] = candidate
                evaluations += 1

            improved = candidate.ds > current.ds + _TOLERANCE
            accepted = improved or (
                accept_equal and candidate.ds >= current.ds - _TOLERANCE
            )

            if accepted:
                current = candidate
            stagnation = 0 if improved else stagnation + 1

            if candidate.ds > best.ds + _TOLERANCE:
                best = candidate

            iteration += 1
            history.append(
                {
                    "iteration": iteration,
                    "swap": (int(first), int(second)),
                    "candidate_ds": candidate.ds,
                    "current_ds": current.ds,
                    "best_ds": best.ds,
                    "accepted": bool(accepted),
                    "restart": False,
                }
            )
            if bar is not None:
                bar.update(1)
                bar.set_postfix(ds=f"{best.ds:.4f}", refresh=False)

            if stagnation >= patience:
                if restarts_used >= n_restarts:
                    break
                restarts_used += 1
                stagnation = 0
                restart_order = [int(position) for position in rng.permutation(n_features)]
                key = tuple(restart_order)
                if key in cache:
                    current = cache[key]
                else:
                    current = evaluate(restart_order)
                    cache[key] = current
                    evaluations += 1
                if current.ds > best.ds + _TOLERANCE:
                    best = current
                history.append(
                    {
                        "iteration": iteration,
                        "swap": None,
                        "candidate_ds": current.ds,
                        "current_ds": current.ds,
                        "best_ds": best.ds,
                        "accepted": True,
                        "restart": True,
                    }
                )
    finally:
        if bar is not None:
            bar.close()

    return {
        "evaluation": best,
        "history": history,
        "n_iterations": iteration,
        "n_restarts_used": restarts_used,
        "n_evaluations": evaluations,
    }
