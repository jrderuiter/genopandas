"""Module containing functions for plotting data along a genomic axis."""

from itertools import chain

import numpy as np
import pandas as pd
import toolz

from .base import scatter_plot, step_plot, apply_palette


def genomic_scatter_plot(data,
                         y,
                         hue=None,
                         hue_order=None,
                         palette=None,
                         color=None,
                         chromosomes=None,
                         legend=True,
                         legend_kws=None,
                         ax=None,
                         style_axis=True,
                         **kwargs):
    """Plots genomic data along a chromosomal axis.

    Parameters
    ----------
    data : GenomicDataFrame
        Genomic data to plot.
    y, hue : str
        Columns to use for plotting. ``y`` determines what is drawn on the
        y-axis. If given, ``hue`` points are colored according to the
        (categorical) values of the respective column. If hue == 'chromosome'
        points are colored per chromosome.
    hue_order : List[str]
        Order to plot the categorical hue levels in.
    palette : List[str] or Dict[Any, str]
        Colors to use for the different levels of the hue variable. Can either
        be a dictionary mapping values to specific colors, or a list of colors
        to use.
    color : matplotlib color
        Color to use for all elements. Overrides palette if given.
    chromosomes: List[str]
        List of chromosomes to plot. Can be used to select a subset of
        chromosomes or to specify a specific order.
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
        Axis on which the data was drawn.

    """

    if chromosomes is not None:
        data = data.gloc[chromosomes]

    # Assemble plot data.
    plot_data = pd.DataFrame({
        'chromosome': data.gloc.chromosome.values,
        'position': data.gloc.position_offset.values,
        'y': data[y].values
    })  # yapf: disable

    if hue is not None and hue not in plot_data:
        plot_data[hue] = data[hue]

    # Order hue by data chromosome order if hue == "chromosome"
    # and no specific order is given.
    if hue == 'chromosome' and hue_order is None:
        hue_order = data.gloc.chromosomes

    # Plot using scatter.
    default_plot_kws = {'markersize': 1}
    plot_kws = toolz.merge(default_plot_kws, kwargs)

    ax = scatter_plot(
        data=plot_data,
        x='position',
        y='y',
        hue=hue,
        hue_order=hue_order,
        palette=palette,
        color=color,
        legend=legend,
        legend_kws=legend_kws,
        ax=ax,
        **plot_kws)

    if style_axis:
        # Style axis.
        _draw_dividers(data.gloc.chromosome_offsets, ax=ax)

        ax.set_xlabel('Chromosome')
        ax.set_ylabel(y)

    return ax


def _draw_dividers(chrom_offsets, ax):
    """Draws chromosome dividers at offsets to indicate chromosomal boundaries.

    The chrom_offsets argument is expected to include _END_ marker (which is
    included by default in GenomicDataFrames).

    Parameters
    ----------
    chrom_offsets : Dict[str, int]
        Position offsets at which to draw boundaries for the
        respective chromosomes.
    ax : AxesSubplot
        Axis to use for drawing.
    """

    positions = np.array(list(chrom_offsets.values()))

    # Draw dividers.
    for loc in positions[1:-1]:
        ax.axvline(loc, color='grey', lw=0.5, zorder=5)

    # Draw xtick labels.
    ax.set_xticks((positions[:-1] + positions[1:]) / 2)
    ax.set_xticklabels(chrom_offsets.keys())

    # Set xlim to boundaries.
    ax.set_xlim(0, chrom_offsets['_END_'])


def genomic_step_plot(data,
                      y,
                      hue=None,
                      hue_order=None,
                      palette=None,
                      color=None,
                      chromosomes=None,
                      legend=True,
                      legend_kws=None,
                      ax=None,
                      style_axis=True,
                      **kwargs):

    if chromosomes is not None:
        data = data.gloc[chromosomes]

    # We need to include both start/end positions in the dataframe.
    # To do so, we basically create two copies of the original df
    # (one with start, one with end positions), concat these two frames
    # and then sort the concatenated frame by original index and position.

    # Create initial frame (with start positions).
    plot_data = pd.DataFrame({
        'chromosome': data.gloc.chromosome.values,
        'position': data.gloc.start_offset.values,
        'y': data[y].values,
        'index': np.arange(len(data[y]))
    })

    if hue is not None:
        plot_data[hue] = data[hue]

    # Merge with copy containing end positions.
    plot_data = pd.concat(
        [plot_data,
         plot_data.assign(position=data.gloc.end_offset.values)],
        axis=0)

    # Sort by original row order.
    plot_data = plot_data.sort_values(by=['index', 'position'])
    plot_data = plot_data.drop('index')

    # Order hue by data chromosome order if hue == "chromosome" and
    # no specific order is given.
    if hue == 'chromosome' and hue_order is None:
        hue_order = data.gloc.chromosomes

    # Plot using step.
    default_step_kws = {'where': 'post'}
    step_kws = toolz.merge(default_step_kws, kwargs)

    ax = step_plot(
        data=plot_data,
        x='position',
        y='y',
        hue=hue,
        hue_order=hue_order,
        palette=palette,
        color=color,
        legend=legend,
        legend_kws=legend_kws,
        ax=ax,
        **step_kws)

    if style_axis:
        # Style axis.
        _draw_dividers(data.gloc.chromosome_offsets, ax=ax)

        ax.set_xlabel('Chromosome')
        ax.set_ylabel(y)

    return ax


def genomic_region_plot(data,
                        y=None,
                        hue=None,
                        hue_order=None,
                        palette=None,
                        color=None,
                        chromosomes=None,
                        ax=None,
                        style_axis=True,
                        **kwargs):
    """Plots highlighted regions along a genomic axis.

    Parameters
    ----------
    data : pandas.DataFrame
        Tidy ('long-form'') dataframe where each column is a variable and
        each row is an observation. Should contain specified
        {chrom,start,end}_col columns (which default to 'chromosome', 'start'
        and 'end', respectively).
    hue : str
        Column to color the data points by.
    hue_order:
        Order to plot the categorical levels in, otherwise the levels
        are inferred from the data objects.
    palette:
        Colors to use for the different levels of the hue variable.
    color:
        Color for all of the elements. Only used when hue is not specified.
    chromosomes: List[str]
        List of chromosomes to plot. Can be used to select a subset of
        chromosomes or to specify a specific order.
    ax : matplotlib Axes, optional
        Axes object to draw the plot onto.
    style_axis : bool
        Whether to style axes with dividers, labels etc. If False, leaves the
        axis unchanged. Useful for combining with other functions
        (such as plot_genomic) to avoid double drawing of annotations.
    kwargs : Dict[str, Any]
        Other keyword arguments are passed through to ``PatchCollection``
        (when drawing with y values) or ``ax.axvspan`` (when drawing without
        y values) at draw time.

    Returns
    -------
    matplotlib.Axes
        The Axes object containing the plot.

    """

    from matplotlib import pyplot as plt

    if chromosomes is not None:
        data = data.gloc[chromosomes]

    # Default axes.
    if ax is None:
        _, ax = plt.subplots()

    # Assemble plot data.
    plot_data = pd.DataFrame({
        'chromosome': data.gloc.chromosome.values,
        'start': data.gloc.start_offset.values,
        'end': data.gloc.end_offset.values
    })

    if y is not None:
        plot_data['value'] = data[y].values
        draw_func = _draw_region_patches
    else:
        draw_func = _draw_region_spans

    if hue is None:
        draw_func(plot_data, ax=ax, color=color, **kwargs)
    else:
        plot_data['hue'] = data[hue]

        plot_data = plot_data.assign(_color=apply_palette(
            plot_data[hue], palette, order=hue_order))

        for (_, color), grp in plot_data.groupby(['hue', '_color']):
            draw_func(grp, ax=ax, color=color, **kwargs)

    if style_axis:
        # Style axis.

        _draw_dividers(data.gloc.chromosome_offsets, ax=ax)
        ax.set_xlabel('Chromosome')

        if y is not None:
            ax.set_ylabel(y)

    # TODO: Set ylim?

    return ax


def _draw_region_patches(grp, ax, color=None, **kwargs):
    from matplotlib import patches as mpl_patches
    from matplotlib import collections as mpl_collections

    grp = grp.assign(width=grp['end'] - grp['start'])

    patches = mpl_collections.PatchCollection(
        (mpl_patches.Rectangle(
            xy=(tup.start, 0), width=tup.width, height=tup.value)
         for tup in grp.itertuples()),
        facecolor=color,
        edgecolor=color,
        **kwargs)

    ax.add_collection(patches)


def _draw_region_spans(grp, ax, color=None, **kwargs):
    for tup in grp.itertuples():
        ax.axvspan(tup.start, tup.end, color=color, **kwargs)
