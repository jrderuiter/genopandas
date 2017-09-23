"""Module containing functions for plotting data along a genomic axis."""

import numpy as np
import pandas as pd

from genopandas.util import with_defaults

from .seaborn import scatter


def plot_genomic(data,
                 y,
                 hue=None,
                 hue_order=None,
                 palette=None,
                 plot_kws=None,
                 legend=True,
                 legend_kws=None,
                 ax=None):
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
    plot_kws : Dict[str, Any]
        Dictionary of additional keyword arguments to pass to ax.plot.
    legend : bool
        Whether to draw a legend for the different hue levels.
        (Only used if hue is given.)
    legend_kws : Dict[str, Any]
        Dictionary of additional keyword arguments to pass to ax.legend
        when drawing the legend.
    ax : AxesSubplot
        Axis to use for drawing.

    Returns
    -------
    AxesSubplot
        Axis on which the data was drawn.

    """

    # Assemble plot data.
    plot_data = pd.DataFrame({
        'chromosome': data.gi.chromosome.values,
        'position': (data.gi.start_offset + data.gi.end_offset) // 2,
        'y': data[y].values
    })  # yapf: disable

    if hue is not None and hue not in plot_data:
        plot_data[hue] = data[hue]

    # Order hue by data chromosome order if hue == "chromosome" and
    # no specific order is given.
    if hue == 'chromosome' and hue_order is None:
        hue_order = data.gi.chromosomes

    # Plot using scatter.
    default_plot_kws = {'markersize': 1}
    plot_kws = with_defaults(plot_kws, default_plot_kws)

    ax = scatter(
        data=plot_data,
        x='position',
        y='y',
        hue=hue,
        hue_order=hue_order,
        palette=palette,
        plot_kws=plot_kws,
        legend=legend,
        legend_kws=legend_kws,
        ax=ax)

    # Style axes.
    _draw_dividers(data.gi.chromosome_offsets, ax=ax)

    ax.set_title(y)
    ax.set_xlabel('Chromosome')
    ax.set_ylabel('Value')

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
