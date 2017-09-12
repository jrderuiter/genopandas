import itertools

from cycler import cycler
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

import seaborn as sns


def plot_genomic(df, y, hue=None, palette=None, ax=None, plot_kws=None):
    """Plots data along a genomic (chromosomal) axis."""

    if ax is None:
        _, ax = plt.subplots()

    plot_kws = plot_kws or {}

    plot_data = pd.DataFrame({
        'chromosome':
        df.gi.chromosome.values,
        'position': (df.gi.start_offset + df.gi.end_offset) // 2,
        'y':
        df[y].values
    })

    if hue is not None:
        plot_data['hue'] = df[hue].values
        plot_data['color'] = _apply_palette(plot_data['hue'], palette)

        for (label, color), grp in plot_data.groupby(['hue', 'color']):
            ax.plot(
                grp['position'],
                grp['y'],
                '.',
                label=label,
                color=color,
                **plot_kws)

        ax.legend(frameon=True, title=hue)
    else:
        if palette is not None:
            ax.set_prop_cycle(cycler('color', palette))

        grouped = plot_data.groupby('chromosome')
        for chrom in df.gi.chromosomes:
            grp = grouped.get_group(chrom)
            ax.plot(grp['position'], grp['y'], '.', **plot_kws)

    _draw_dividers(ax=ax, chrom_offsets=df.gi.chromosome_offsets)

    ax.set_title(y)
    ax.set_xlabel('Chromosome')
    ax.set_ylabel('Value')

    return ax


def _apply_palette(series, palette, bg_color='white'):
    if not isinstance(palette, dict):
        colors = itertools.cycle(palette)
        palette = dict(zip(series.unique(), colors))
    return series.map(palette).fillna(bg_color)


def _draw_dividers(ax, chrom_offsets):
    """Plots chromosome dividers.

    Note: chrom_offsets is expected to include _END_ marker.
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

    return ax
