"""Sparseness Optimized Feature Importance."""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .encoding import build_feature_space
from .explanation import SOFIExplanation
from .marginalization import Marginalizer
from .model import ModelWrapper
from .objective import Objective
from .search import hill_climbing

__all__ = ["SOFIExplainer"]


def _as_frame(data, columns=None, name: str = "X") -> pd.DataFrame:
    """Return ``data`` as a DataFrame without copying whenever possible."""
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, pd.Series):
        return data.to_frame().T
    array = np.asarray(data)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two dimensional structure.")
    if columns is None:
        columns = [f"f{index}" for index in range(array.shape[1])]
    if len(columns) != array.shape[1]:
        raise ValueError(
            f"{name} has {array.shape[1]} columns while {len(columns)} names are known."
        )
    return pd.DataFrame(array, columns=list(columns))


def _progress(iterable, enabled: bool, description: str):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        return iterable
    return tqdm(iterable, desc=description)


class SOFIExplainer:
    """Post hoc explainer that optimizes the sparseness of feature rankings.

    SOFI searches for a ranking of features whose cumulative marginalization
    degrades the response of the model as fast as possible. A ranking is judged
    through its degradation score, namely the integral between the curve
    obtained when the least relevant features are marginalized first and the
    one obtained when the most relevant features are marginalized first. The
    search works on the raw response of the model, and normalization is applied
    afterwards, for reporting and for drawing alone. Maximizing
    that area rewards sparsity and correctness at once. The search is a hill
    climbing procedure whose operator swaps two randomly selected positions of
    the ranking.

    Classification and regression share the same rationale. A classifier is
    monitored through the probability it assigns to the class predicted before
    any perturbation, while a regressor is monitored through the absolute
    deviation with respect to its unperturbed estimate. Both quantities are
    reported as a prediction fidelity that starts at one and reaches zero once
    every feature has been marginalized, so curves belonging to different
    problems or to different instances are read on the same scale.

    Parameters
    ----------
    model : object
        Fitted estimator. A ``Pipeline`` that carries the encoder and the
        estimator is the recommended input, since feature blocks are then
        handled without any further declaration.
    X_train : DataFrame or ndarray
        Training data expressed in the space where features are perturbed. It
        is the only source of marginalization values, and the data explained
        later never contributes statistics.
    encoder : transformer, optional
        Fitted encoder used when the estimator does not carry one. SOFI applies
        it after every perturbation, so a one-hot block is always rebuilt from
        a complete category.
    feature_groups : dict or "auto", optional
        Mapping from a logical feature onto the columns that represent it. Use
        it when ``X_train`` is already encoded. The value ``auto`` recovers the
        mapping from a fitted ``ColumnTransformer`` given through ``encoder``.
    task : {"auto", "classification", "regression"}, default="auto"
        Learning task, resolved from the estimator by default.
    predict_fn, proba_fn : callable, optional
        Prediction functions for estimators outside the scikit-learn interface.
    marginalization : {"background", "constant"}, default="background"
        Whether a feature is neutralized by averaging the response over a fixed
        sample of training rows or by substituting a single statistic. The
        background variant approximates the expectation the method is defined
        on, and it is the default because a single statistic can place an
        instance in a region the model reads with confidence, which leaves the
        fully perturbed response above the unperturbed one and makes the curves
        unreadable. The constant variant costs one query per step instead of
        one per background draw.
    numeric_strategy : {"mean", "median", "mode"}, default="mean"
        Statistic used for floating point columns.
    integer_strategy : {"mode", "median", "mean"}, default="mode"
        Statistic used for integer columns. The default preserves the dtype.
    categorical_strategy : {"mode"}, default="mode"
        Statistic used for nominal columns.
    n_background : int, default=25
        Size of the background sample of the background variant.
    initialization : {"greedy", "random"} or sequence, default="greedy"
        Starting ranking. The greedy option sorts the features by the
        degradation each of them causes on its own, and an explicit sequence of
        feature names or positions lets prior knowledge enter the search.
    max_iterations : int, default=200
        Budget of proposed swaps.
    patience : int, optional
        Consecutive swaps without improvement tolerated before a restart or the
        end of the search. The default equals the whole budget.
    n_restarts : int, default=0
        Restarts from a random ranking granted once the patience expires.
    accept_equal : bool, default=False
        Whether swaps that leave the degradation score unchanged are accepted.
    chunk_size : int, default=20000
        Upper bound on the number of rows sent to the estimator in one query.
    random_state : int or Generator, optional
        Seed of the background sample and of the search.
    verbose : bool, default=True
        Whether progress bars are displayed.

    Examples
    --------
    >>> explainer = SOFIExplainer(pipeline, X_train, random_state=42)
    >>> explanation = explainer.explain(X_validation, mode="global")
    >>> explanation.ranking[:3]
    """

    def __init__(
        self,
        model,
        X_train,
        encoder=None,
        feature_groups: Optional[Union[Dict, str]] = None,
        task: str = "auto",
        predict_fn=None,
        proba_fn=None,
        marginalization: str = "background",
        numeric_strategy: str = "mean",
        integer_strategy: str = "mode",
        categorical_strategy: str = "mode",
        n_background: int = 25,
        initialization: Union[str, Sequence] = "greedy",
        max_iterations: int = 200,
        patience: Optional[int] = None,
        n_restarts: int = 0,
        accept_equal: bool = False,
        chunk_size: int = 20000,
        random_state=None,
        verbose: bool = True,
    ):
        if encoder is not None and feature_groups is None and _looks_like_pipeline(model):
            warnings.warn(
                "An encoder was supplied together with a pipeline that already "
                "encodes the data. SOFI will apply the encoder before calling the "
                "pipeline, which is rarely the intended behavior.",
                stacklevel=2,
            )

        self.model = model
        self.verbose = bool(verbose)
        self.random_state = random_state
        self.max_iterations = int(max_iterations)
        self.patience = patience
        self.n_restarts = int(n_restarts)
        self.accept_equal = bool(accept_equal)
        self.initialization = initialization
        self.chunk_size = int(chunk_size)

        _check_fitted(model)

        self.X_train_ = _as_frame(X_train, _model_feature_names(model), "X_train")
        if self.X_train_.isna().to_numpy().any():
            warnings.warn(
                "The training data contains missing values. Marginalization "
                "statistics ignore them, yet the estimator must be able to "
                "handle the values that remain in the data.",
                stacklevel=2,
            )

        self.space_ = build_feature_space(
            model, self.X_train_, encoder=encoder, feature_groups=feature_groups
        )
        self.wrapper_ = ModelWrapper(
            model,
            task=task,
            predict_fn=predict_fn,
            proba_fn=proba_fn,
            output_feature_names=self.space_.output_feature_names,
        )
        self.task = self.wrapper_.task
        self._smoke_test()

        self.marginalizer_ = Marginalizer(
            self.space_,
            self.X_train_,
            strategy=marginalization,
            numeric_strategy=numeric_strategy,
            integer_strategy=integer_strategy,
            categorical_strategy=categorical_strategy,
            n_background=n_background,
            random_state=random_state,
        )

    # ---------------------------------------------------------------- setup
    def _smoke_test(self) -> None:
        sample = self.X_train_.head(min(2, len(self.X_train_)))
        try:
            transformed = self.space_.transform(sample)
            if self.task == "classification":
                self.wrapper_.predict_proba(transformed)
            else:
                self.wrapper_.predict(transformed)
        except Exception as error:  # pragma: no cover - depends on user setup
            raise ValueError(
                "The estimator could not consume the training data handled by SOFI. "
                "Supply the model as a pipeline that carries its encoder, or pass the "
                "fitted encoder through the encoder argument, or describe the encoded "
                "columns through feature_groups. Original error, "
                f"{type(error).__name__}, {error}"
            ) from error

    # ------------------------------------------------------------- features
    @property
    def feature_names_(self) -> List[str]:
        return list(self.space_.feature_names)

    def feature_map(self) -> pd.DataFrame:
        """Logical features together with the columns they occupy."""
        return self.space_.describe()

    def marginalization_values(self) -> pd.DataFrame:
        """Values that replace each feature under the constant variant."""
        rows = []
        for feature, values in self.marginalizer_.values_.items():
            for column, value in values.items():
                rows.append({"feature": feature, "column": column, "value": value})
        return pd.DataFrame(rows)

    # -------------------------------------------------------------- explain
    def explain(
        self,
        X,
        mode: str = "global",
        initialization: Optional[Union[str, Sequence]] = None,
        max_iterations: Optional[int] = None,
        patience: Optional[int] = None,
        n_restarts: Optional[int] = None,
        random_state=None,
    ) -> Union[SOFIExplanation, List[SOFIExplanation]]:
        """Explain a dataset or a single instance.

        Parameters
        ----------
        X : DataFrame, Series or ndarray
            Data to be explained, expressed in the same space as the training
            data. Filtering the instances beforehand is the recommended
            practice, since the premise of the method is that performance
            deteriorates as features are marginalized, which only holds for
            instances the model handles well.
        mode : {"global", "local"}, default="global"
            The global mode searches for a single ranking that degrades the
            average response over the supplied instances. The local mode runs
            one search per instance and returns one explanation each.

        Returns
        -------
        SOFIExplanation or list of SOFIExplanation
            A single explanation in the global mode, and one explanation per
            instance in the local mode.
        """
        if mode not in ("global", "local"):
            raise ValueError("mode must be either 'global' or 'local'.")

        data = self._prepare(X)
        seed = self.random_state if random_state is None else random_state

        if mode == "global" or len(data) == 1:
            return self._run(
                data,
                mode=mode,
                initialization=initialization,
                max_iterations=max_iterations,
                patience=patience,
                n_restarts=n_restarts,
                seed=seed,
                verbose=self.verbose,
                instance_index=data.index[0] if len(data) == 1 else None,
            )

        explanations = []
        iterator = _progress(range(len(data)), self.verbose, "SOFI local explanations")
        for position in iterator:
            explanations.append(
                self._run(
                    data.iloc[[position]],
                    mode="local",
                    initialization=initialization,
                    max_iterations=max_iterations,
                    patience=patience,
                    n_restarts=n_restarts,
                    seed=seed,
                    verbose=False,
                    instance_index=data.index[position],
                )
            )
        return explanations

    def _prepare(self, X) -> pd.DataFrame:
        data = _as_frame(X, list(self.X_train_.columns), "X")
        missing = [c for c in self.X_train_.columns if c not in data.columns]
        if missing:
            raise ValueError(
                "The data to be explained lacks columns present in the training "
                f"data. First missing column, {missing[0]}."
            )
        extra = [c for c in data.columns if c not in set(self.X_train_.columns)]
        if extra:
            warnings.warn(
                f"{len(extra)} columns absent from the training data were dropped "
                "before the explanation.",
                stacklevel=3,
            )
        return data.loc[:, list(self.X_train_.columns)]

    def _run(
        self,
        data: pd.DataFrame,
        mode: str,
        initialization,
        max_iterations,
        patience,
        n_restarts,
        seed,
        verbose: bool,
        instance_index=None,
    ) -> SOFIExplanation:
        objective = Objective(
            self.wrapper_,
            self.space_,
            self.marginalizer_,
            data,
            chunk_size=self.chunk_size,
        )
        rng = np.random.default_rng(seed)
        start = self._initial_order(
            self.initialization if initialization is None else initialization,
            objective,
            rng,
        )

        result = hill_climbing(
            n_features=self.space_.n_features,
            evaluate=objective.evaluate,
            initial_order=start,
            max_iterations=self.max_iterations if max_iterations is None else int(max_iterations),
            patience=self.patience if patience is None else patience,
            n_restarts=self.n_restarts if n_restarts is None else int(n_restarts),
            accept_equal=self.accept_equal,
            rng=rng,
            verbose=verbose,
        )
        evaluation = result["evaluation"]

        return SOFIExplanation(
            feature_names=self.feature_names_,
            order=[int(position) for position in evaluation.order],
            morf_scores_raw=[float(score) for score in evaluation.morf],
            lerf_scores_raw=[float(score) for score in evaluation.lerf],
            ds_raw=float(evaluation.ds),
            task=self.task,
            mode=mode,
            anchor_high=float(objective.anchor_high_),
            anchor_low=float(objective.anchor_low_),
            raw_baseline=float(objective.raw_baseline_),
            history=result["history"],
            instance_index=instance_index,
            n_instances=len(data),
            n_iterations=result["n_iterations"],
            n_restarts_used=result["n_restarts_used"],
            n_evaluations=result["n_evaluations"],
            n_model_calls=objective.n_calls_,
        )

    def _initial_order(self, initialization, objective: Objective, rng) -> List[int]:
        n_features = self.space_.n_features
        if isinstance(initialization, str):
            if initialization == "random":
                return [int(position) for position in rng.permutation(n_features)]
            if initialization != "greedy":
                raise ValueError(
                    "initialization must be 'greedy', 'random' or an explicit ranking."
                )
            return list(objective.greedy_order_)

        order = list(initialization)
        if len(order) != n_features:
            raise ValueError(
                f"An explicit initial ranking must list the {n_features} features."
            )
        names = self.feature_names_
        resolved = []
        for item in order:
            if isinstance(item, (int, np.integer)):
                resolved.append(int(item))
            else:
                if item not in names:
                    raise ValueError(f"Unknown feature in the initial ranking, {item}.")
                resolved.append(names.index(item))
        if sorted(resolved) != list(range(n_features)):
            raise ValueError("The initial ranking must be a permutation of the features.")
        return resolved

    # ------------------------------------------------------------ utilities
    def score_ranking(self, ranking: Sequence, X) -> SOFIExplanation:
        """Evaluate a ranking supplied by the user, without any search.

        The method is the natural way of comparing SOFI against a competing
        attribution method under the same marginalization protocol.

        Parameters
        ----------
        ranking : sequence
            Feature names or positions, from the most to the least important.
        X : DataFrame, Series or ndarray
            Data on which the ranking is evaluated. Fidelity is bounded by
            construction, so any two rankings scored on the same set share one
            vertical axis and remain comparable.
        """
        data = self._prepare(X)
        objective = Objective(
            self.wrapper_,
            self.space_,
            self.marginalizer_,
            data,
            chunk_size=self.chunk_size,
        )
        order = self._initial_order(list(ranking), objective, np.random.default_rng(0))
        evaluation = objective.evaluate(order)
        return SOFIExplanation(
            feature_names=self.feature_names_,
            order=order,
            morf_scores_raw=[float(score) for score in evaluation.morf],
            lerf_scores_raw=[float(score) for score in evaluation.lerf],
            ds_raw=float(evaluation.ds),
            task=self.task,
            mode="global" if len(data) > 1 else "local",
            anchor_high=float(objective.anchor_high_),
            anchor_low=float(objective.anchor_low_),
            raw_baseline=float(objective.raw_baseline_),
            n_instances=len(data),
            n_model_calls=objective.n_calls_,
        )


def _looks_like_pipeline(model) -> bool:
    try:
        from sklearn.pipeline import Pipeline
    except ImportError:  # pragma: no cover
        return False
    return isinstance(model, Pipeline)


def _model_feature_names(model):
    names = getattr(model, "feature_names_in_", None)
    return None if names is None else list(names)


def _check_fitted(model) -> None:
    try:
        from sklearn.exceptions import NotFittedError
        from sklearn.utils.validation import check_is_fitted
    except ImportError:  # pragma: no cover
        return
    try:
        check_is_fitted(model)
    except NotFittedError as error:
        raise ValueError(
            "SOFI explains fitted estimators, and the supplied model is not fitted."
        ) from error
    except TypeError:
        return
