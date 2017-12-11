import itertools
from types import MappingProxyType

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

import numpy as np
import pandas as pd
import seaborn as sns
import toolz

from genopandas.plotting.clustermap import color_annotation

DEFAULT_ALTERATIONS = (
    ('cna', [('amp', '#fd0000'),
             ('homdel', '#0005fe'),
             ('gain', '#feb6c1'),
             ('hetloss', '#90d8d8')]),
    ('mutation', [('trunc', '#000000'),
                  ('missense', '#0b8008')]),
    ('expression', [('increased', '#fe9999'),
                    ('decreased', '#6799cc')])
)  # yapf: disable

DEFAULT_STYLES = MappingProxyType({
    'cna': 'fill',
    'mutation': 'inset',
    'expression': 'border'
})


class OncoPlotter(object):
    """Plotter object that facilitates the drawing of oncoplots."""

    def __init__(self,
                 alterations=DEFAULT_ALTERATIONS,
                 alteration_styles=DEFAULT_STYLES,
                 inset_width=0.6,
                 inset_height=0.6,
                 border_width=2,
                 keep_unaltered=False):

        self._alterations = alterations
        self._alteration_styles = alteration_styles

        self._inset_height = inset_height
        self._inset_width = inset_width

        self._border_lw = border_width

        self._keep_unaltered = keep_unaltered

    def plot(self,
             data,
             sample_order=None,
             gene_order=None,
             sort=True,
             limit=10,
             ax=None,
             linewidths=1):
        """Draws oncoplot."""

        # Determine gene/sample order.
        gene_order, sample_order = self._sort_orders(
            data,
            self._alterations,
            sort=sort,
            gene_order=gene_order,
            sample_order=sample_order,
            limit=limit,
            keep_unaltered=self._keep_unaltered)

        data = data.loc[data['gene'].isin(gene_order)]

        if ax is None:
            _, ax = plt.subplots()

        # Plot background.
        self._plot_background(
            gene_order=gene_order,
            sample_order=sample_order,
            ax=ax,
            linewidths=linewidths)

        # Plot elements.
        self._plot_fill(
            data,
            gene_order=gene_order,
            sample_order=sample_order,
            ax=ax,
            linewidths=linewidths)

        self._plot_inset(
            data, gene_order=gene_order, sample_order=sample_order, ax=ax)

        self._plot_border(
            data,
            gene_order=gene_order,
            sample_order=sample_order,
            ax=ax,
            linewidths=linewidths)

        # Finish styling axes.
        self._style_axis(ax)

        return OncoPlot(
            gene_order=gene_order, sample_order=sample_order, ax=ax)

    @classmethod
    def _sort_orders(cls,
                     data,
                     alterations,
                     sort=True,
                     gene_order=None,
                     sample_order=None,
                     limit=10,
                     keep_unaltered=True):
        """Determines sort orders with given inputs."""

        if sort and (gene_order is None or sample_order is None):
            sorted_genes, sorted_samples = sort_by_alterations(
                data, alterations, gene_order=gene_order, limit=limit)

            gene_order = gene_order or sorted_genes
            sample_order = sample_order or sorted_samples
        else:
            if gene_order is None:
                gene_order = cls._get_series_order(data['gene'])[:limit]

            if sample_order is None:
                sample_order = cls._get_series_order(data['sample'])

        if not keep_unaltered:
            altered = set(data.loc[data['gene'].isin(gene_order), 'sample'])
            sample_order = [
                sample for sample in sample_order if sample in altered
            ]

        return gene_order, sample_order

    @staticmethod
    def _get_series_order(series):
        if hasattr(series, 'cat'):
            return series.cat.categories
        else:
            return list(series.unique())

    def _plot_background(self, gene_order, sample_order, ax, linewidths):
        """Plots oncoplot background."""

        null_data = pd.DataFrame(0, index=gene_order, columns=sample_order)

        sns.heatmap(
            null_data,
            cmap=discrete_cmap(['gainsboro']),
            ax=ax,
            cbar=False,
            linewidths=linewidths)

    def _plot_fill(self, data, gene_order, sample_order, ax, linewidths):
        """Plots oncoplot fill elements."""

        # Fetch alterations, subset data and map to numeric matrix.
        alterations = self._subset_for_style(style='fill')

        alteration_names = {alt_name for alt_name, _ in alterations}
        data = data.loc[data['type'].isin(alteration_names)]

        matrix, colors = self._map_to_order(data, alterations, as_matrix=True)
        matrix = matrix.reindex(index=gene_order, columns=sample_order)

        # Setup matplotlib cmap (order --> colors).
        cmap = discrete_cmap(colors)

        # Plot heatmap.
        sns.heatmap(
            matrix,
            cmap=cmap,
            ax=ax,
            vmin=0,
            vmax=len(colors),
            cbar=False,
            linewidths=linewidths)

    def _subset_for_style(self, style):
        """Subsets alterations for given style."""

        return [(alt, alt_values) for alt, alt_values in self._alterations
                if self._alteration_styles[alt] == style]

    def _map_to_order(self, data, alterations, as_matrix=False):
        """Maps data to numeric order values (keeping only highest
           priority values)."""

        order_mapping, order_colors = self._order_mapping(alterations)

        value_keys = data['type'] + '_' + data['value']
        values = value_keys.map(order_mapping)

        data_num = pd.DataFrame({
            'sample': data['sample'],
            'gene': data['gene'],
            'value': values
        })

        if as_matrix:
            data_num = pd.pivot_table(
                data_num,
                index='gene',
                columns='sample',
                values='value',
                aggfunc='min')
        else:
            data_num = data_num.groupby(
                ['gene', 'sample'])['value'].min().reset_index()

        return data_num, order_colors

    @staticmethod
    def _order_mapping(alterations):
        """Creates mapping of alterations to consecutive numeric values.

        Format: {type}_{value} --> {index}. Used when pivoting matrix, to ensure
        priority is followed when converting from long format to matrix.
        """

        order_mapping, colors = {}, []

        for data_type, val_order in alterations:
            for val, color in val_order:
                key = data_type + '_' + val
                order_mapping[key] = len(order_mapping)

                colors.append(color)

        return order_mapping, colors

    def _plot_inset(self, data, gene_order, sample_order, ax):
        """Plots oncoplot inset elements."""

        margin_x = (1 - self._inset_width) / 2
        margin_y = (1 - self._inset_height) / 2

        def _draw_func(grp, gene_lookup, sample_lookup, color, ax):
            patches = (Rectangle(
                xy=(sample_lookup[row.sample] + margin_x,
                    gene_lookup[row.gene] + margin_y),
                width=self._inset_width,
                height=self._inset_height) for row in grp.itertuples())

            patches = PatchCollection(patches, facecolor=color)
            ax.add_collection(patches)

        self._draw_with_lookup(
            data,
            style='inset',
            gene_order=gene_order,
            sample_order=sample_order,
            draw_func=_draw_func,
            ax=ax)

    def _draw_with_lookup(self, data, style, gene_order, sample_order,
                          draw_func, ax):
        """Helper function that draws elements using row/column lookups."""

        # Fetch alterations, subset data and map to numeric values.
        alterations = self._subset_for_style(style=style)

        alteration_names = {alt_name for alt_name, _ in alterations}
        data = data.loc[data['type'].isin(alteration_names)]

        data_ordered, colors = self._map_to_order(
            data, alterations, as_matrix=False)

        # Draw insets.
        sample_lookup = dict(zip(sample_order, range(len(sample_order))))
        gene_lookup = dict(zip(gene_order, range(len(gene_order))))

        for value, grp in data_ordered.groupby('value'):
            if len(grp) > 0:
                draw_func(
                    grp,
                    sample_lookup=sample_lookup,
                    gene_lookup=gene_lookup,
                    color=colors[int(value)],
                    ax=ax)

    def _plot_border(self, data, gene_order, sample_order, ax, linewidths):
        """Plots 'bordered' elements."""

        offset_x, offset_y = self._border_offsets(
            self._border_lw, linewidths, ax=ax)

        def _draw_func(grp, sample_lookup, gene_lookup, color, ax):
            for row in grp.itertuples():
                patch = Rectangle(
                    xy=(sample_lookup[row.sample] + offset_x,
                        gene_lookup[row.gene] + offset_y),
                    width=1. - (2 * offset_x),
                    height=1. - (2 * offset_y),
                    edgecolor=color,
                    facecolor='none',
                    lw=self._border_lw,
                    joinstyle='miter')

                ax.add_patch(patch)

        self._draw_with_lookup(
            data,
            style='border',
            gene_order=gene_order,
            sample_order=sample_order,
            draw_func=_draw_func,
            ax=ax)

    @staticmethod
    def _border_offsets(border_lw, linewidths, ax):
        # Force fig draw.
        fig = ax.figure
        fig.canvas.draw()

        # Get transform/DPI.
        transform = ax.transData.inverted().transform
        ppd = 72. / fig.dpi

        # Transform line widths to data coords.
        border_offset = (linewidths + 0.5 * border_lw) - 0.5

        transformed = ((transform((border_offset, border_offset)) -
                        transform((0, 0))) * ppd)  # yapf: disable

        offset_x, offset_y = np.abs(transformed)

        return offset_x, offset_y

    @staticmethod
    def _style_axis(ax):
        """Styles axis."""

        ax.set_xlabel('Samples')
        ax.set_ylabel('Genes')

        plt.setp(ax.get_yticklabels(), rotation=0)


class OncoPlot(object):
    """Oncoplot figure object."""

    def __init__(self, gene_order, sample_order, ax):
        self.gene_order = gene_order
        self.sample_order = sample_order
        self.ax = ax

    @property
    def fig(self):
        """Figure on which oncoplot was drawn."""
        return self.ax.figure

    def savefig(self, *args, **kwargs):
        """Saves oncoplot to file using ``figure.savefig``."""
        return self.fig.savefig(*args, **kwargs)


def oncoplot(data,
             alterations=DEFAULT_ALTERATIONS,
             alteration_styles=DEFAULT_STYLES,
             sample_order=None,
             gene_order=None,
             sort=True,
             limit=10,
             ax=None,
             linewidths=1,
             inset_width=0.6,
             inset_height=0.6,
             border_width=2,
             keep_unaltered=False):
    """Plots oncoplot for given data."""

    plotter = OncoPlotter(
        alterations=alterations,
        alteration_styles=alteration_styles,
        inset_width=inset_width,
        inset_height=inset_height,
        border_width=border_width,
        keep_unaltered=keep_unaltered)

    return plotter.plot(
        data=data,
        sample_order=sample_order,
        gene_order=gene_order,
        sort=sort,
        limit=limit,
        ax=ax,
        linewidths=linewidths)


def sort_by_alterations(data, alterations, gene_order=None, limit=10):
    """Returns sorted gene/sample orders for given data."""

    # Convert data to ordered (numeric) format for sorting.
    sort_mapping = _sort_mapping(alterations)
    value_keys = data['type'] + '_' + data['value']

    sort_data = pd.DataFrame({
        'sample': data['sample'],
        'gene': data['gene'],
        'type': data['type'],
        'value': value_keys.map(sort_mapping)
    })

    # Determine gene order.
    if gene_order is None:
        gene_ranking = (sort_data.dropna()
                        .groupby('gene')['sample'].nunique()
                        .sort_values(ascending=False))  # yapf: disable
        gene_order = list(gene_ranking.index)

    gene_order = gene_order[:limit]

    # Take most important effect per type (max).
    sort_data = sort_data.groupby(['sample', 'gene', 'type'])['value'].max()
    sort_data = sort_data.reset_index()

    # Sort, taking gene order into account.
    sort_data = sort_data.loc[sort_data['gene'].isin(gene_order)]
    sort_mat = pd.pivot_table(
        sort_data,
        index='sample',
        columns='gene',
        values='value',
        aggfunc='sum')

    sort_mat = sort_mat.sort_values(by=gene_order, ascending=False)
    sample_order = list(sort_mat.index)

    return gene_order, sample_order


def _sort_mapping(alterations):
    """Maps data types and their values to numeric values used for sorting."""

    n_levels = len(alterations)

    mapping = {}
    for i, (data_type, val_order) in enumerate(alterations):
        multiplier = 10**(n_levels - i)

        for j, (val, _) in enumerate(val_order):
            key = data_type + '_' + val
            mapping[key] = multiplier * (len(val_order) - j)

    return mapping


def discrete_cmap(colors):
    """Create an N-bin discrete colormap from specified colors."""

    if not colors:
        return None
    else:
        if len(colors) == 1:
            # Repeat single colors as a work-around
            # for single color issue.
            colors = colors * 2

        n_colors = len(colors)
        base = plt.cm.get_cmap()
        cmap_name = base.name + str(n_colors)
        return LinearSegmentedColormap.from_list(cmap_name, colors, n_colors)


class OncoprintPlotter(object):
    """Plotter object that facilitates the drawing of oncoprints."""

    def __init__(self,
                 alterations=DEFAULT_ALTERATIONS,
                 alteration_styles=DEFAULT_STYLES,
                 inset_width=0.6,
                 inset_height=0.6,
                 border_width=2,
                 keep_unaltered=False,
                 bar_color='dimgray'):

        self._alterations = alterations
        self._bar_color = bar_color

        self._oncoplotter = OncoPlotter(
            alterations=alterations,
            alteration_styles=alteration_styles,
            inset_width=inset_width,
            inset_height=inset_height,
            border_width=border_width,
            keep_unaltered=keep_unaltered)

    def plot(self,
             data,
             sample_order=None,
             gene_order=None,
             annotation=None,
             annotation_colors=None,
             sort=True,
             limit=10,
             linewidths=1,
             figsize=None,
             gridspec_kws=None):
        """Draws oncoprint on figure."""

        # Setup figure.
        fig, axes = self._setup_figure(
            figsize=figsize,
            gridspec_kws=gridspec_kws,
            with_annotation=annotation is not None)

        # Draw oncoplot.
        oncoplot = self._oncoplotter.plot(
            data=data,
            sample_order=sample_order,
            gene_order=gene_order,
            sort=sort,
            limit=limit,
            ax=axes[-1, 0],
            linewidths=linewidths)

        # Draw frequencies + annotation (if given).
        self._plot_frequencies(
            data,
            gene_order=oncoplot.gene_order,
            ax=axes[-1, 1],
            color=self._bar_color)

        if annotation is not None:
            annotation = annotation.reindex(index=oncoplot.sample_order)

            self._plot_annotation(
                annotation,
                annotation_colors,
                ax=axes[0, 0],
                linewidths=linewidths)

        # Style axes.
        axes[0, 1].axis('off')

        return Oncoprint(fig=fig, axes=axes, oncoplot=oncoplot)

    @staticmethod
    def _setup_figure(figsize, gridspec_kws=None, with_annotation=False):
        """Sets up figure + axes for plotting."""

        gridspec_defaults = {
            'width_ratios': [0.8, 0.2],
            'wspace': 0.02,
            'hspace': 0.02
        }

        if with_annotation:
            nrows = 2
            gridspec_defaults['height_ratios'] = [0.2, 0.8]
        else:
            nrows = 1

        gridspec_kws = toolz.merge(gridspec_defaults, (gridspec_kws or {}))

        fig, axes = plt.subplots(
            figsize=figsize,
            ncols=2,
            nrows=nrows,
            gridspec_kw=gridspec_kws,
            squeeze=False)

        return fig, axes

    def _plot_frequencies(self, data, gene_order, ax=None, **kwargs):
        """Plots bargraph with alteration frequencies."""

        # Subset data for alterations.
        alt_values = set(alt + '_' + value
                         for alt, alt_values in self._alterations
                         for value, _ in alt_values)

        mask = (data['type'] + '_' + data['value']).isin(alt_values)
        data = data.loc[mask]

        # Calculate frequencies/fractions.
        freqs = (data.groupby('gene')['sample'].nunique()
                 .sort_values(ascending=False))  # yapf: disable

        fractions = freqs / data['sample'].nunique()

        # Use gene_order to select shown genes.
        fractions = fractions.loc[gene_order]

        # Plot barplot.
        ax = sns.barplot(
            data=fractions.to_frame('fraction').reset_index(),
            y='gene',
            x='fraction',
            ax=ax,
            order=fractions.index,
            **kwargs)

        # Annotate with values.
        for i, value in enumerate(fractions):
            ax.text(
                value + 0.02,
                i,
                '{0:.0f}%'.format(value * 100),
                ha='left',
                va='center')

        # Style axes.
        ax.get_xaxis().set_visible(False)

        ax.set_yticks([])
        ax.set_ylabel('')

        sns.despine(ax=ax, left=True, bottom=True)

        return ax

    def _plot_annotation(self, annotation, annotation_colors, ax, linewidths):
        """Plots annotation along top, in similar style as sns.clustermap."""

        colored, _ = color_annotation(annotation, annotation_colors)
        colored_numeric, cmap = self._color_matrix_to_numeric(colored)

        sns.heatmap(
            colored_numeric.T,
            cmap=cmap,
            ax=ax,
            cbar=False,
            linewidths=linewidths,
            yticklabels=True)

        ax.set_xticks([])
        ax.set_xlabel('')

        plt.setp(ax.get_yticklabels(), rotation=0)

    @staticmethod
    def _color_matrix_to_numeric(matrix):
        """Converts matrix containing color values to a numeric matrix + cmap,
           which can be used to draw the colors using sns.heatmap."""

        # Get list of all colors in matrix.
        col_colors = [col_values.unique()
                      for _, col_values in matrix.items()]  # yapf: disable
        all_colors = list(set(itertools.chain.from_iterable(col_colors)))

        # Map colors to integer values.
        value_map = dict(zip(all_colors, range(len(all_colors))))
        num_matrix = matrix.applymap(lambda v: value_map.get(v))

        # Create cmap.
        cmap = discrete_cmap(all_colors)

        return num_matrix, cmap


def oncoprint(data,
              alterations=DEFAULT_ALTERATIONS,
              alteration_styles=DEFAULT_STYLES,
              sample_order=None,
              gene_order=None,
              annotation=None,
              annotation_colors=None,
              sort=True,
              limit=10,
              linewidths=1,
              inset_width=0.6,
              inset_height=0.6,
              border_width=2,
              keep_unaltered=False,
              bar_color='dimgray',
              figsize=None,
              gridspec_kws=None):
    """Plots oncoprint (with freq. and annotation) for given data."""

    plotter = OncoprintPlotter(
        alterations=alterations,
        alteration_styles=alteration_styles,
        inset_width=inset_width,
        inset_height=inset_height,
        border_width=border_width,
        bar_color=bar_color,
        keep_unaltered=keep_unaltered)

    return plotter.plot(
        data=data,
        annotation=annotation,
        annotation_colors=annotation_colors,
        sample_order=sample_order,
        gene_order=gene_order,
        sort=sort,
        limit=limit,
        linewidths=linewidths,
        figsize=figsize,
        gridspec_kws=gridspec_kws)


class Oncoprint(object):
    """Oncoprint figure object."""

    def __init__(self, fig, axes, oncoplot):
        self.fig = fig
        self.axes = axes
        self.oncoplot = oncoplot

    @property
    def ax_oncoplot(self):
        """Axis on which oncoplot was drawn."""
        return self.oncoplot.ax

    @property
    def ax_frequency(self):
        """Axis on which alteration frequencies were drawn."""
        return self.axes[1, 1]

    def savefig(self, *args, **kwargs):
        """Saves figure using ``fig.savefig``."""
        return self.fig.savefig(*args, **kwargs)
