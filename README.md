# Sparseness Optimized Feature Importance

Sparseness Optimized Feature Importance (SOFI) is a model agnostic, declarative post hoc explainer. An explanation takes the form of a ranking of features, and its quality is the degradation score obtained after cumulative marginalization. The implementation supports classification and regression, operates at the instance level or over a whole dataset, and marginalizes one-hot encoded features as indivisible blocks.

## Installation

```bash
pip install sofi-tabular
```

The package requires Python 3.9 or later, together with `numpy`, `pandas` and `scikit-learn`,
which is all `pip install sofi-tabular` pulls in. Plotting, progress bars and the demo are
optional extras declared in `pyproject.toml`:

```bash
pip install "sofi-tabular[plot]"      # matplotlib and seaborn, needed by explanation.plot()
pip install "sofi-tabular[progress]"  # tqdm, needed for progress bars
pip install "sofi-tabular[demo]"      # everything above, plus what SOFI_demo.ipynb needs to run
```

Progress bars need `tqdm`, figures need `matplotlib`, and `seaborn` is used only to fix the
font of a session. `requirements.txt` pins all of them together for a quick `pip install -r
requirements.txt` when the distinction does not matter.

To install the development version from GitHub:

```bash
git clone https://github.com/igraugar/sofi.git
cd sofi
pip install -e ".[demo]"
```

## Quick start

```python
from sofi import SOFIExplainer, select_reliable_instances, set_plot_style

set_plot_style(font_scale=1.15)                                  # once per session
validation = select_reliable_instances(model, X_test, y_test)    # instances the model gets right

explainer = SOFIExplainer(model, X_train, random_state=42)
explanation = explainer.explain(validation, mode="global")

print(explanation.summary())
explanation.plot()
```

`SOFI_demo.ipynb` develops four complete examples on datasets read from public GitHub
repositories, each of them with more than 15 features. They cover classification and
regression, features that are mixed, purely nominal or purely numerical, and the four ways of
telling the explainer how features map onto the columns the model consumes.

| Example | Task | Features | Encoding |
| --- | --- | --- | --- |
| 1. Bank marketing | classification | 19, mixed | pipeline carrying the encoder |
| 2. Mushroom | classification | 21, only nominal | encoder passed apart |
| 3. Telecom charges | regression | 18, mixed | encoded data with feature groups |
| 4. College tuition | regression | 16, only numerical | model passed directly |

Each example reports the whole set of metrics twice over, for the global explanation and for
four individual instances chosen so that their explanations disagree as much as possible with
the global one and with each other. The instances alternate the two marginalization variants,
so a single notebook shows both of them at work.

## The degradation score

Two curves are derived from a ranking. The MoRF curve marginalizes the most relevant
features first and should collapse immediately, since a sparse explanation concentrates the
response of the model in a handful of features. The LeRF curve marginalizes the least
relevant features first and should stay flat for as long as possible, since the features
declared irrelevant must indeed be the ones whose removal leaves the response untouched.
Both curves share their first point, the unperturbed response, and their last point, the
response once every feature has been neutralized. The degradation score is the area between
the two curves, and the search maximizes it. A single curve cannot capture both properties:
the MoRF branch alone rewards sparsity while saying nothing about the tail of the ranking, and
the LeRF branch alone rewards correctness at the tail while saying nothing about the head.

The search works on the raw response of the model and on nothing else, as in region
perturbation. Classification tracks the probability that the model assigns to the class it
predicted before any perturbation. Regression tracks the mean absolute deviation with respect
to the unperturbed estimate, taken with a negative sign so that both quantities fall as
evidence is destroyed. Neither task consults the ground truth, which keeps explanations
independent of the error of the model.

Normalization never enters the optimization, since a scale that depends on the ranking being
scored would change what is being maximized. It is applied afterwards, for reporting and for
drawing, and it follows the convention of the perturbation curve literature: the response of
the unperturbed input is placed at one and the response of the fully perturbed input at zero.
Both anchors belong to the model and the data rather than to a ranking, so every ranking
scored on the same data shares them, the reported score is the optimized score divided by one
positive constant, and no ordering can change. Reported curves therefore start at one and stay
inside the unit interval, so figures from different instances, models and problems can be read
on the same axis.

The anchors are read off a reference sweep that marginalizes every feature on its own and then
follows the resulting greedy ordering cumulatively, which also produces the fully perturbed
response. In the ordinary case the two anchors are exactly the two responses the convention
prescribes. Should the response climb substantially above the unperturbed one, the scale
widens to the extremes the sweep observed and a warning names the cause, so that a curve is
never flattened against a false ceiling. When the response never falls at all, a warning says
so too, because the degradation score then carries no information.

## Marginalizing a feature

Marginalization values come from the training data alone, and the variant matters more than
it may appear. The background variant averages the response over a fixed sample of training
rows (`n_background` of them, 25 by default), which approximates the expectation the method is
defined on, and it is the default. The constant variant substitutes one statistic instead,
which costs one query per step rather than one per background draw.

The saving is real and so is the risk. Replacing every feature of an instance with its most
frequent category and its mean value produces the single most typical row in the training
data, and some models classify that row with more confidence than the original instance. When
that happens, the fully perturbed response ends up above the unperturbed one instead of below
it, which is exactly the situation the reference sweep described above is built to detect and
warn about. Averaging over a background sample largely removes the effect, since the response
then approaches the marginal output of the model rather than the output at one atypical point.

Raise `n_background` for a smoother expectation and lower it, or subsample the validation set,
when the search has to run quickly. The whole validation set is replicated once per draw, so
the cost grows linearly with the sample.

## The noise region

Once the informative features are gone, marginalizing what remains often pushes the response
back towards its original state. The MoRF curve then climbs after its minimum, and the tail
of the ranking that follows that minimum carries no evidence. The region is reported by
`noise_onset` and `noise_features`, and it also appears in the printed summary. The figure
leaves it unshaded, so that the only shaded region is the degradation score itself.

## The search

Hill climbing with a local operator that swaps two randomly selected positions of the current
ranking. A candidate is accepted when it raises the degradation score. The starting ranking
is either random or greedy, the latter reusing the ordering already produced by the reference
sweep, at no extra cost. An explicit ranking is also accepted, which is how prior knowledge
enters the procedure. The run ends after a budget of iterations or once the patience expires,
and restarts from a random ranking replace an early stop when `n_restarts` is positive.

## Declaring the structure of the features

Marginalizing a single column of a one-hot block would measure the role of one category
rather than the role of the feature. SOFI therefore needs to know which columns form a
logical feature, and four arrangements are available. The demo devotes one example to each.

**A pipeline that carries the encoder.** Nothing has to be declared. Perturbation happens in
the original feature space and the encoder is applied afterwards, so a category is always
replaced with another complete category. This path is the recommended one.

```python
model = Pipeline([("encoder", column_transformer), ("forest", RandomForestClassifier())])
model.fit(X_train, y_train)
explainer = SOFIExplainer(model, X_train)
```

**A fitted encoder passed apart.** Use it when the estimator was trained on encoded data and
cannot be wrapped. Perturbation still happens in the original space.

```python
explainer = SOFIExplainer(model, X_train_raw, encoder=column_transformer)
```

**Data that is already encoded.** Declare the blocks through `feature_groups`, either
explicitly or recovered from a fitted `ColumnTransformer`. Columns that are not listed become
features of their own.

```python
from sofi import feature_groups_from_encoder

groups = feature_groups_from_encoder(column_transformer)
explainer = SOFIExplainer(model, X_train_encoded, feature_groups=groups)
```

**No encoding at all.** A dataset described only by numerical features needs none of the
above, and the estimator reaches the explainer directly.

```python
explainer = SOFIExplainer(regressor, X_train)
```

One difference between the paths deserves attention. Marginalization statistics follow the
dtypes of the data handed to the explainer, and a `ColumnTransformer` returns a single
floating point matrix regardless of the dtypes it was fed, so an integer column that went
through a passthrough transformer receives the statistic of a continuous variable in the third
path. Restore the original dtypes on the encoded frame when exact agreement matters.

## Containers and estimators

The explainer records the container the estimator was fitted with and adapts every internal
query accordingly, so a model trained on a `DataFrame` keeps receiving named columns in the
original order and a model trained on an array keeps receiving an array. Input given to
`explain` may be a `DataFrame`, a `Series` or an array. Estimators outside the scikit-learn
interface are supported through `predict_fn` and `proba_fn`.

A prediction is attempted at construction time on up to two training rows, and a failure
raises a message that names the three paths above, which turns an encoding mismatch into an
immediate and readable error rather than a silent misinterpretation.

## Local and global modes

`mode="global"` searches for one ranking that degrades the average response over the
supplied instances. `mode="local"` runs one search per instance, returning a single
explanation for one row and a list of explanations for several rows. Instance level results
are summarized with `aggregate_explanations`, which reports the mean rank of every feature
and how often it reaches the first three positions.

Training data is always required, since it is the only source of marginalization values. The
data explained afterwards never contributes statistics.

## Choosing the validation set

The premise of the method is that the response deteriorates as features are marginalized,
which only holds for instances the model handles well. The selection is deliberately left
outside the explainer, and `select_reliable_instances` covers the two usual criteria, namely
correct classification and small absolute error.

```python
validation = select_reliable_instances(model, X_test, y_test)                # classifier
validation = select_reliable_instances(model, X_test, y_test, quantile=0.5)  # regressor
```

## Comparing SOFI against other explainers

`explainer.score_ranking` evaluates a ranking produced elsewhere under the same protocol and
on the same reported scale, which makes the degradation score a common ground for comparison.
Any attribution method that ends in an ordering of features can be measured this way.

```python
scored = explainer.score_ranking(other_ranking, validation)
print(scored.ds, explanation.ds)
```

## Figures

Font sizes are never touched by the plotting routine. One call to `set_plot_style` fixes them
for the whole session, and the title, the axes, the ticks, the legend and the annotation
follow that single setting. Colours work the same way. One call to `set_curve_colors` holds
for every figure afterwards (the session starts with `#03719c` for MoRF and `#1A1A1A` for
LeRF), and a single figure can depart from it through the `morf_color` and `lerf_color`
arguments of `plot`.

```python
from sofi import set_curve_colors, set_plot_style

set_plot_style(font_scale=1.15)
set_curve_colors(morf="#03719c", lerf="#16120e")

explanation.plot()                                    # the colours in force
explanation.plot(lerf_color="#865321")                # this figure alone
```

Both curves are drawn on a fixed axis that spans the unit interval, labeled `Prediction
fidelity` for either task, the legend reads `MoRF` and `LeRF` in a single row underneath the
axes, and the only shaded region is the area between the curves. Several explanations are
drawn side by side with `plot_explanation_grid`, which packs the panels as closely as the
labels allow and takes the same colour arguments.

## Cost and reproducibility

Building the objective for a dataset costs two sweeps of the features: one to marginalize each
feature alone, which produces the greedy ordering, and one to follow that ordering
cumulatively. Every candidate proposed afterwards requires the two curves, so an evaluation
costs twice a single curve. A curve issues one batched query per chunk of `chunk_size` rows,
and a candidate produced by a swap at positions `i` and `j` (with `i < j`) reuses the first `i`
points of the MoRF curve of the incumbent and the first `n - 1 - j` points of its LeRF curve,
since the reversed ranking is affected at mirrored positions. Only the affected tails are
recomputed, and the result is identical to a full evaluation.

Every run is reproducible from `random_state`, which controls the background sample, the swap
operator and the restarts. No internal state is modified during a search, so repeated calls
on the same data return the same explanation.

Global mode on a large validation set is the expensive setting, since every point of both
curves predicts the whole set. Subsampling the validation data is the usual remedy.

## Reference of `SOFIExplainer`

| Parameter | Default | Accepted values | Rationale |
| --- | --- | --- | --- |
| `model` | required | Any fitted estimator exposing `predict`, plus `predict_proba` for classification. A `Pipeline` carrying the encoder is the recommended form. | The explainer is agnostic to the model family and only queries its outputs. |
| `X_train` | required | `DataFrame`, `Series` or two dimensional array. | Source of the marginalization statistics. A frame is preferred, since dtypes drive the choice of statistic. |
| `encoder` | `None` | `None`, or any fitted transformer exposing `transform`. | Declares the second path. The transformer is applied after every perturbation, so blocks are rebuilt from complete categories. `None` means that either the model carries its encoder or no encoding is needed. |
| `feature_groups` | `None` | `None`, a dict mapping a feature name onto a list of columns, or `"auto"`. | Declares the third path, for data that is already encoded. `"auto"` recovers the mapping from the `ColumnTransformer` given in `encoder`. Columns left out become features of their own. |
| `task` | `"auto"` | `"auto"`, `"classification"`, `"regression"`. | `"auto"` reads the task from the estimator and is enough for scikit-learn objects. The explicit values are needed for wrappers whose type cannot be inferred. |
| `predict_fn` | `None` | `None` or a callable receiving the adapted input and returning an array. | Escape hatch for estimators outside the scikit-learn interface. |
| `proba_fn` | `None` | `None` or a callable returning a matrix of class probabilities. | Same escape hatch for classifiers whose probabilities are not exposed through `predict_proba`. |
| `marginalization` | `"background"` | `"background"`, `"constant"`. | `"background"` averages the response over a sample of training rows, which approximates the expectation the method is defined on. `"constant"` substitutes one statistic instead, which costs one query per step rather than one per draw, at the risk of placing an instance in a region the model reads with confidence. |
| `numeric_strategy` | `"mean"` | `"mean"`, `"median"`, `"mode"`. | Statistic of the floating point columns. `"mean"` neutralizes a feature at its center of mass, `"median"` resists skewed distributions and outliers, and `"mode"` keeps a value that the feature actually takes. |
| `integer_strategy` | `"mode"` | `"mode"`, `"median"`, `"mean"`. | Statistic of the integer columns. `"mode"` is the default because it preserves the dtype and produces a value the feature can take, such as a count. The other two are rounded, with a warning, when the column must stay integer. |
| `categorical_strategy` | `"mode"` | `"mode"`. | Statistic of nominal, boolean and categorical columns. Only the most frequent category is admissible, since no average of labels exists. A multi column block declared through `feature_groups` receives the most frequent joint pattern, which is always a valid state of the block. |
| `n_background` | `25` | Integer of at least one, capped at the number of training rows. | Size of the background sample. Larger values reduce the variance of the expectation and raise the cost linearly, since the validation set is replicated once per draw. Ignored under the constant variant. |
| `initialization` | `"greedy"` | `"greedy"`, `"random"`, or a sequence of feature names or positions. | `"greedy"` starts from the ordering produced by the reference sweep, which is already available and usually reaches a good region within few swaps. `"random"` removes that bias and suits repeated runs with restarts. An explicit sequence lets prior knowledge enter the search. |
| `max_iterations` | `200` | Non-negative integer. | Budget of proposed swaps across all restarts. Zero evaluates the initial ranking without any search, which is convenient for scoring a ranking chosen beforehand. |
| `patience` | `None` | `None`, or a positive integer. | Consecutive swaps without improvement tolerated before a restart or the end of the search. `None` sets it to the whole budget, so the search never stops early. |
| `n_restarts` | `0` | Integer of at least zero. | Restarts from a random ranking granted once the patience expires. Zero ends the run instead, and larger values trade coverage of the search space for time. |
| `accept_equal` | `False` | `False`, `True`. | Whether swaps that leave the score unchanged are accepted. `True` lets the search drift along plateaus, which helps when many features are redundant, and it makes the trajectory less stable. |
| `chunk_size` | `20000` | Positive integer. | Upper bound on the rows sent to the estimator in one query. Larger values reduce the call overhead and raise the peak memory, and lower values suit estimators with a costly batch. |
| `random_state` | `None` | `None`, an integer, or a `numpy` `Generator`. | Seed of the background sample, of the swap operator and of the restarts. An integer makes a run reproducible, and `None` draws fresh randomness. |
| `verbose` | `True` | `True`, `False`. | Whether progress bars are displayed, which requires `tqdm`. |

Two further methods report how the explainer resolved the arguments above:

| Method | Returns | Purpose |
| --- | --- | --- |
| `feature_map()` | `DataFrame` with `feature`, `n_columns`, `columns` | One row per logical feature, showing how many columns it occupies and which ones, useful for checking that `feature_groups` or the encoder were understood as intended. |
| `marginalization_values()` | `DataFrame` with `feature`, `column`, `value` | The statistic that replaces each column under the constant variant. Not populated by the background draws, which are resampled internally rather than fixed per feature. |

## Reference of `explain`

| Parameter | Default | Accepted values | Rationale |
| --- | --- | --- | --- |
| `X` | required | `DataFrame`, `Series` or two dimensional array, expressed in the space of the training data. | Instances to be explained. A single row triggers one search and a frame triggers one search per row in local mode. |
| `mode` | `"global"` | `"global"`, `"local"`. | `"global"` seeks one ranking for the averaged response of the whole set, which describes the model. `"local"` seeks one ranking per instance, which describes a decision. |
| `initialization`, `max_iterations`, `patience`, `n_restarts`, `random_state` | `None` | Same values as in the constructor, or `None`. | Per call overrides. `None` keeps the value given at construction time, so the same explainer can be reused under different budgets. |

The method returns a `SOFIExplanation` for a single instance or for the global mode, and a
list of them when several instances are explained locally.

## Reference of `select_reliable_instances`

| Parameter | Default | Accepted values | Rationale |
| --- | --- | --- | --- |
| `model`, `X`, `y` | required | Fitted estimator, a `DataFrame` and an array of targets. | The subset is built outside the explainer, so the criterion stays under the control of the user. |
| `encoder` | `None` | `None`, or a fitted transformer. | Applied before the estimator when the model does not carry its encoder. |
| `task` | `"auto"` | `"auto"`, `"classification"`, `"regression"`. | Chooses between the correctness criterion and the error criterion. |
| `quantile` | `0.5` | Float between zero and one. | Fraction of regression instances retained, ordered by absolute error. Ignored for classifiers and superseded by `tolerance`. |
| `tolerance` | `None` | `None`, or a non-negative float. | Absolute error threshold on the original scale of the target, which is preferable when a domain rule defines what a small error is. |
| `return_mask` | `False` | `False`, `True`. | Whether the boolean mask is returned as well, which is how the matching targets are selected. |

## Reference of `SOFIExplanation`

| Attribute | Meaning |
| --- | --- |
| `ranking`, `order` | Features sorted from the most to the least important, by name and by position |
| `top(k)` | First `k` features of the ranking |
| `morf_scores`, `lerf_scores` | Reported curves of the ranking and of its reverse, inside the unit interval |
| `morf_scores_raw`, `lerf_scores_raw` | The same curves on the raw response of the model |
| `ds_raw` | The integral on the raw scale, which is the quantity the search maximizes |
| `anchor_high`, `anchor_low` | Responses placed at one and at zero when a curve is reported |
| `scores` | Alias of `morf_scores` |
| `ds` | Degradation score, namely the area between the two curves, maximized by the search |
| `auc_morf`, `auc_lerf` | Mean height of each curve |
| `drops` | Degradation attributable to each step of the MoRF curve |
| `importances` | Rank based importance in the unit interval |
| `noise_onset`, `noise_features` | Start of the region where fidelity recovers, and the features it holds |
| `statistics` | Every figure of the run gathered in one dictionary |
| `summary(top_k=None)` | Report meant to be printed, also returned by `print(explanation)` |
| `to_frame()` | Tabular view of the ranking |
| `plot()` | Both curves, the enclosed area and the noise region |
| `history` | One record per iteration of the search |
| `n_iterations`, `n_restarts_used`, `n_evaluations`, `n_model_calls` | Counters describing the run |

## Reference of `plot` and `set_plot_style`

| Parameter | Default | Accepted values | Rationale |
| --- | --- | --- | --- |
| `ax` of `plot` | `None` | `None`, or a matplotlib `Axes`. | `None` creates a figure, and an explicit axes places the curves inside a larger layout, such as a two by two grid of local explanations. |
| `title` of `plot` | `None` | `None`, or a string. | `None` reports the mode of the run. |
| `annotate_ds` of `plot` | `True` | `True`, `False`. | Whether the degradation score is written inside the axes. |
| `legend` of `plot` | `True` | `True`, `False`. | Whether the two curves are labeled. The legend is laid out as a single row underneath the axes, so it never covers the curves. |
| `figsize` of `plot` | `(6.6, 4.2)` | Pair of floats. | Size of the figure created when no axes is supplied. |
| `morf_color`, `lerf_color` of `plot` | `None` | Any colour matplotlib accepts, such as `"#c0392b"` or `"tab:red"`. | Colour of each curve for one figure. The colours in force for the session apply when they are left out. |
| `morf`, `lerf` of `set_curve_colors` | `None` | The same values. | Colour of each curve for the whole session. An argument left out keeps the colour in force. |
| `font_scale` of `set_plot_style` | `1.2` | Positive float. | Multiplier applied to every font of the session. The figure itself never overrides it. |
| `style`, `context` of `set_plot_style` | `"whitegrid"`, `"notebook"` | Any seaborn style and context name. | Passed to `seaborn.set_theme` when seaborn is installed, and ignored otherwise. |

## Advanced building blocks

`SOFIExplainer` is a thin orchestrator over four pieces, all importable from `sofi` for anyone
who wants to reuse part of the machinery outside the explainer, for instance to plug a custom
search into the existing objective:

| Name | What it does |
| --- | --- |
| `FeatureSpace` | Holds the mapping from logical features to columns, together with the transform applied after a perturbation. Returned by `SOFIExplainer.feature_map()` in tabular form. |
| `Marginalizer` | Computes and applies the values that neutralize a feature, under either variant. |
| `Objective` | Turns a ranking into a `RankingEvaluation` by driving the model through cumulative marginalization, and owns the anchors described under "The degradation score". |
| `hill_climbing` | The search itself, decoupled from SOFI's objective: it accepts any `evaluate` callable that returns an object exposing `.order` and `.ds`. |
| `curve_auc`, `degradation_score` | The scoring functions applied to any pair of MoRF/LeRF curves, independent of how they were produced. |
| `RankingEvaluation` | The dataclass (`order`, `morf`, `lerf`, `ds`) passed between the objective and the search. |
| `noise_onset` | The free function behind `SOFIExplanation.noise_onset`, usable directly on any curve. |

Most users never need to import these directly; `SOFIExplainer.explain` and `.score_ranking`
cover the ordinary usage.

## Citation

```bibtex
@inproceedings{grau2024sofi,
  title     = {Sparseness-Optimized Feature Importance},
  author    = {Grau, Isel and N{\'a}poles, Gonzalo},
  booktitle = {Explainable Artificial Intelligence. xAI 2024},
  series    = {Communications in Computer and Information Science},
  volume    = {2154},
  pages     = {393--415},
  publisher = {Springer},
  year      = {2024},
  doi       = {10.1007/978-3-031-63797-1_20}
}

@article{grau2026sofits,
  title   = {Sparseness-Optimized Feature Importance for Time Series Classification},
  author  = {Grau, Isel and N{\'a}poles, Gonzalo and Jastrzebska, Agnieszka and Salgueiro, Yamisleydi},
  journal = {IEEE Access},
  volume  = {14},
  pages   = {29874--29893},
  year    = {2026}
}
```

## License

MIT. See `LICENSE`.
