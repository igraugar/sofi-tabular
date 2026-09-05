"""Visualization of the degradation curves produced by SOFI.

The figure never touches the font settings of the session. Sizes are taken
from the active matplotlib or seaborn theme, so one global call such as
``seaborn.set_theme(font_scale=1.2)`` governs the title, the axes, the ticks,
the legend and the annotation alike.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = [
    "MORF_COLOR",
    "LERF_COLOR",
    "plot_degradation_curve",
    "plot_explanation_grid",
    "set_curve_colors",
    "set_plot_style",
]

def set_plot_style(font_scale: float = 1.2, style: str = "whitegrid", context: str = "notebook"):
    """Fix the font size of the session once, before any figure is drawn.

    The plotting routine of SOFI never overrides font sizes, so the value set
    through this helper governs the title, the axes, the ticks, the legend and
    the annotation alike. ``seaborn`` is used when available and the matplotlib
    defaults are adjusted otherwise.

    Parameters
    ----------
    font_scale : float, default=1.2
        Multiplier applied to every font of the theme.
    style : str, default="whitegrid"
        Name of the seaborn style. Ignored without seaborn.
    context : str, default="notebook"
        Name of the seaborn context. Ignored without seaborn.
    """
    try:
        import seaborn as sns

        sns.set_theme(style=style, context=context, font_scale=font_scale)
    except ImportError:  # pragma: no cover - seaborn is optional
        import matplotlib as mpl

        mpl.rcParams.update({"font.size": 10.0 * font_scale})
    return font_scale


# colours of the two curves, changed for a whole session through set_curve_colors
MORF_COLOR = "#03719c"
LERF_COLOR = "#1A1A1A"


def set_curve_colors(morf=None, lerf=None):
    """Set the colour of the MoRF and the LeRF curves for the whole session.

    Parameters
    ----------
    morf, lerf : str, optional
        Any colour matplotlib accepts, such as ``"#c0392b"`` or ``"tab:red"``.
        An argument left out keeps the colour in force.

    Returns
    -------
    tuple of str
        The two colours after the change.
    """
    global MORF_COLOR, LERF_COLOR
    if morf is not None:
        MORF_COLOR = str(morf)
    if lerf is not None:
        LERF_COLOR = str(lerf)
    return MORF_COLOR, LERF_COLOR
_YLABEL = "Prediction fidelity"


def plot_degradation_curve(
    explanation,
    ax=None,
    title: Optional[str] = None,
    annotate_ds: bool = True,
    legend: bool = True,
    figsize=(6.6, 4.2),
    morf_color: Optional[str] = None,
    lerf_color: Optional[str] = None,
):
    """Draw the MoRF and the LeRF curves together with their enclosed area.

    Reported curves start at one and stay inside the unit interval, so the
    vertical axis is fixed. The shaded region between them is the degradation
    score, which places the quantity maximized by the search in the figure
    itself.

    Parameters
    ----------
    explanation : SOFIExplanation
        Result returned by the explainer.
    ax : matplotlib Axes, optional
        Target axes. A new figure is created when omitted.
    title : str, optional
        Heading of the figure. A default one reports the mode of the run.
    annotate_ds : bool, default=True
        Whether the degradation score is written inside the axes.
    legend : bool, default=True
        Whether the two curves are labeled. The legend is laid out as a single
        row underneath the axes, so it never covers the curves.
    figsize : tuple, default=(6.6, 4.2)
        Size of the figure created when no axes is supplied.
    morf_color, lerf_color : str, optional
        Colour of each curve for this figure alone. The colours in force for
        the session are used when they are left out, and ``set_curve_colors``
        changes those.
    """
    import matplotlib.pyplot as plt

    morf = np.asarray(explanation.morf_scores, dtype=float)
    lerf = np.asarray(explanation.lerf_scores, dtype=float)
    steps = np.arange(len(morf))

    if ax is None:
        _, ax = plt.subplots(figsize=figsize, layout="constrained")

    # reported curves live inside the unit interval, so the axis is fixed, with
    # a margin that leaves room for the annotation
    bottom, top = -0.05, 1.22

    ax.fill_between(steps, morf, lerf, color="grey", alpha=0.22, zorder=2)
    ax.plot(
        steps,
        lerf,
        marker="s",
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=0.8,
        color=lerf_color or LERF_COLOR,
        linewidth=2.6,
        linestyle="-",
        label="LeRF",
        zorder=3,
    )
    ax.plot(
        steps,
        morf,
        marker="s",
        markersize=9,
        markeredgecolor="white",
        markeredgewidth=0.8,
        color=morf_color or MORF_COLOR,
        linewidth=2.6,
        label="MoRF",
        zorder=4,
    )

    ax.set_ylim(bottom, top)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(-0.4, len(morf) - 0.6)
    step = max(1, int(np.ceil(len(morf) / 11)))
    ax.set_xticks(steps[::step])
    ax.set_xlabel("Marginalized features")
    ax.set_ylabel(_YLABEL)
    ax.set_title(
        title if title is not None else f"SOFI, {explanation.mode} mode"
    )

    if legend:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.26),
            ncol=2,
            frameon=False,
            borderaxespad=0.0,
            handlelength=2.0,
            columnspacing=2.0,
        )

    if annotate_ds:
        ax.text(
            0.98,
            0.96,
            f"DS: {explanation.ds:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(
                facecolor="white",
                alpha=0.9,
                edgecolor="#001f3f",
                boxstyle="round,pad=0.35",
            ),
        )
    return ax


def plot_explanation_grid(
    picks,
    title=None,
    ncols: int = 2,
    figsize=None,
    panel_size=(6.3, 3.9),
    **kwargs,
):
    """Draw several explanations on one grid of panels.

    Panels are packed as closely as the labels allow, so the heading sits just
    above the first row and consecutive rows are separated by the height of a
    legend and nothing more.

    Parameters
    ----------
    picks : sequence
        Explanations, pairs holding a heading and an explanation, or objects
        exposing an ``explanation`` and a ``title``.
    title : str, optional
        Heading of the grid.
    ncols : int, default=2
        Number of panels per row.
    figsize : tuple, optional
        Size of the figure. It is derived from ``panel_size`` when omitted.
    panel_size : tuple, default=(6.3, 3.9)
        Size of one panel, used to derive the size of the figure.
    **kwargs
        Further arguments handed to every panel, such as ``morf_color`` and
        ``lerf_color``.
    """
    import matplotlib.pyplot as plt

    items = list(picks)
    if not items:
        raise ValueError("At least one explanation is required.")

    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(items) / ncols))
    if figsize is None:
        figsize = (panel_size[0] * ncols, panel_size[1] * nrows)

    figure, axes = plt.subplots(nrows, ncols, figsize=figsize, layout="constrained")
    figure.get_layout_engine().set(h_pad=0.08, w_pad=0.02, hspace=0.06, wspace=0.03)
    flat = np.atleast_1d(axes).ravel()

    for item, ax in zip(items, flat):
        if isinstance(item, tuple):
            heading, explanation = item
        else:
            explanation = getattr(item, "explanation", item)
            heading = getattr(item, "title", None)
        plot_degradation_curve(explanation, ax=ax, title=heading, **kwargs)
    for ax in flat[len(items):]:
        ax.axis("off")

    if title is not None:
        figure.suptitle(title)
    return figure
