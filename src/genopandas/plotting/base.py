"""Module containing functions that supplant seaborn's plotting functions.

Examples include a ``scatter`` function, which creates a scatterplot whilst
keeping an API similar to the other seaborn plotting functions.
"""

from itertools import cycle

import toolz

from ._constants import CATEGORICAL_COLORS


def scatter_plot(data,
                 x,
                 y,
                 hue=None,
                 hue_order=None,
                 palette=None,
                 color=None,
                 legend=True,
                 legend_kws=None,
                 ax=None,
                 **kwargs):
    """Draws a scatter plot of the given dataset.

    Input data is expected to be in long (tidy) form.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset for plotting. Is expected to be in long (tidy) form.
    x, y, hue : str
        Columns to use for plotting. ``x`` and ``y`` determine what is plotted
        along the x- and y-axes, respectively. If ``hue`` is given, points are
        colored according to the (categorical) values of the respective column.
    hue_order : List[str]
        Order to plot the categorical hue levels in.
    palette : List[str] or Dict[Any, str]
        Colors to use for the different levels of the hue variable. Can either
        be a dictionary mapping values to specific colors, or a list of colors
        to use.
    color : matplotlib color
        Color to use for all elements. Overrides palette if given.
    legend : bool
        Whether to draw a legend for the different hue levels.
        (Only used if hue is given.)
    legend_kws : Dict[str, Any]
        Dictionary of additional keyword arguments to pass to ax.legend
        when drawing the legend.
    ax : AxesSubplot
        Axis to use for drawing.
    kwargs : Dict[str, Any]
        Other keyword arguments are passed through to ``ax.plot`` at draw time.

    Returns
    -------
    AxesSubplot
        Axis on which the scatterplot was drawn.

    """

    from matplotlib import pyplot as plt

    default_plot_kws = {'linestyle': 'None', 'marker': 'o'}
    plot_kws = {**default_plot_kws, **kwargs}

    if ax is None:
        _, ax = plt.subplots()

    if hue is None:
        ax.plot(data[x], data[y], color=color, **plot_kws)
    else:
        if palette is None:
            palette = cycle(CATEGORICAL_COLORS)

        data = data.assign(_color=apply_palette(
            data[hue], palette, color=color, order=hue_order))

        # Group data by label/color.
        groups = {
            label: (color, grp)
            for (label, color), grp in data.groupby([hue, '_color'])
        }

        # Plot groups in hue order.
        hue_order = hue_order or set(groups.keys())

        for label in hue_order:
            color, grp = groups[label]
            ax.plot(grp[x], grp[y], label=label, color=color, **plot_kws)

        if legend:
            # Draw legend.
            default_legend_kws = {
                'frameon': True,
                'title': hue,
                'loc': 'upper right'
            }
            ax.legend(**toolz.merge(default_legend_kws, (legend_kws or {})))

    ax.set_xlabel(x)
    ax.set_ylabel(y)

    return ax


def step_plot(data,
              x,
              y,
              hue=None,
              hue_order=None,
              palette=None,
              color=None,
              legend=True,
              legend_kws=None,
              ax=None,
              **kwargs):

    from matplotlib import pyplot as plt

    default_step_kws = {'where': 'post'}
    step_kws = {**default_step_kws, **kwargs}

    if ax is None:
        _, ax = plt.subplots()

    if hue is None:
        ax.step(data[x], data[y], color=color, **step_kws)
    else:
        if palette is None:
            palette = cycle(CATEGORICAL_COLORS)

        data = data.assign(_color=apply_palette(
            data[hue], palette, color=color, order=hue_order))

        for (label, color), grp in data.groupby([hue, '_color']):
            ax.step(grp[x], grp[y], label=label, color=color, **step_kws)

        if legend:
            default_legend_kws = {
                'frameon': True,
                'title': hue,
                'loc': 'upper right'
            }
            ax.legend(**toolz.merge(default_legend_kws, (legend_kws or {})))

    ax.set_xlabel(x)
    ax.set_ylabel(y)

    return ax


def apply_palette(series, palette, color=None, bg_color='white', order=None):
    """Uses palette to color values in the given categorical series.

    Parameters
    ----------
    series : pd.Series
        Values to color.
    palette : List[str] or Dict[Any, str]
        Colors to use for the different levels of the series. Can either be a
        dictionary mapping values to specific colors or a list of colors.
    color : matplotlib color
        Color to use for all elements. Overrides palette if given.
    bg_color : str
        Background color to use for NA values.
    order : list[str]
        Order to color the categorical hue levels in.

    Returns
    -------
    pd.Series
        Series containing colored values.

    """

    if not isinstance(palette, dict):
        if order is None:
            order = series.unique()

        colors = cycle(palette)
        palette = dict(zip(order, colors))

    if color is not None:
        palette = {k: color for k in palette.keys()}

    return series.map(palette).fillna(bg_color)
