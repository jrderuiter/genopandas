from itertools import chain, cycle
from operator import attrgetter
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tqdm import tqdm
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from sklearn.decomposition import PCA
from pandas.api.types import is_numeric_dtype

from ngstk2.plotting.genomic import plot_genomic
from ngstk2.plotting.clustermap import color_annotation, draw_legends

from genopandas import GenomicDataFrame


class AnnotatedMatrix(object):
    """Base AnnotatedMatrix class."""

    def __init__(self, values, design=None):
        assert design is None or (values.shape[1] == design.shape[0]
                                  and all(values.columns == design.index))

        # Check if all columns are numeric.
        for col_name, col_values in values.items():
            if not is_numeric_dtype(col_values):
                raise ValueError('Column {} is not numeric'.format(col_name))

        if design is None:
            design = pd.DataFrame({}, index=values.columns)

        self._values = values
        self._design = design

    @property
    def values(self):
        """Matrix values."""
        return self._values

    @property
    def design(self):
        """Matrix design (sample annotation)."""
        return self._design

    @property
    def samples(self):
        """List of samples in matrix."""
        return list(self._values.columns)

    @property
    def features(self):
        """List of features in matrix."""
        return list(self._values.index)

    @property
    def loc(self):
        """Label based indexer (similar to pandas .loc)."""

        return LocIndexer(
            df=self._values,
            class_=self.__class__,
            extra_args={'design': self._design})

    def __getitem__(self, item):
        values = self._values[item]
        design = self._design.loc[item]
        return self.__class__(values, design=design)

    def rename(self, name_map, drop=False):
        """Renames samples using given mapping."""

        renamed = self._rename(self._values, name_map, drop=drop)

        design = self._design.rename(index=name_map)
        design = design.loc[renamed.columns]

        return self.__class__(renamed, design=design)

    @staticmethod
    def _rename(df, name_map, drop=False):
        renamed = df.rename(columns=name_map)

        if drop:
            extra = set(renamed.columns) - set(name_map.values())
            renamed = renamed.drop(extra, axis=1)

        return renamed

    def query(self, expr):
        design = self._design.query(expr)
        values = self._values[design.index]
        return self.__class__(values, design=design)

    def dropna(self, subset=None, how='any', thresh=None):
        design = self._design.dropna(subset=subset, how=how, thresh=thresh)
        values = self._values[design.index]
        return self.__class__(values, design=design)

    @classmethod
    def concat(cls, matrices, axis='sample'):
        if axis == 'sample':
            value_axis, design_axis = 1, 0
        elif axis == 'feature':
            value_axis, design_axis = 0, 1
        else:
            raise ValueError('Unexpected value for axis ({}). Valid values '
                             'are \'sample\' or \'feature\''.format(axis))

        values = pd.concat([mat.values for mat in matrices], axis=value_axis)
        design = pd.concat([mat.design for mat in matrices], axis=design_axis)

        return cls(values, design=design)

    @classmethod
    def from_csv(cls,
                 file_path,
                 design=None,
                 name_map=None,
                 drop_cols=None,
                 **kwargs):

        values = cls._read_csv(
            file_path,
            design=design,
            name_map=name_map,
            drop_cols=drop_cols,
            **kwargs)

        return cls(values, design=design)

    @classmethod
    def _read_csv(cls,
                  file_path,
                  design=None,
                  name_map=None,
                  drop_cols=None,
                  **kwargs):
        values = pd.read_csv(file_path, **kwargs)

        if drop_cols is not None:
            values = values.drop(drop_cols, axis=1)

        values = cls._rename_and_match(
            values, design=design, name_map=name_map)

        return values

    @classmethod
    def _rename_and_match(cls, values, design=None, name_map=None):
        """Optionally renames values and matches to design."""

        if name_map is not None:
            values = cls._rename(values, name_map, drop=True)

        if design is not None:
            values = values[design.index]

        return values

    def plot_heatmap(self,
                     cmap='RdBu_r',
                     annotation=None,
                     annotation_colors=None,
                     metric='euclidean',
                     method='complete',
                     transpose=False,
                     legend_kws=None,
                     **kwargs):
        """Plots (clustered) heatmap of values."""

        if annotation is not None:
            annot_colors, annot_cmap = color_annotation(
                self._design[annotation], colors=annotation_colors)
        else:
            annot_colors, annot_cmap = None, None

        clustermap_kws = dict(kwargs)

        if transpose:
            values = self._values.T
            clustermap_kws['row_colors'] = annot_colors
            xlabel, ylabel = 'Features', 'Samples'
        else:
            values = self._values
            clustermap_kws['col_colors'] = annot_colors
            xlabel, ylabel = 'Samples', 'Features'

        cm = sns.clustermap(
            values, cmap=cmap, metric=metric, method=method, **clustermap_kws)

        plt.setp(cm.ax_heatmap.get_yticklabels(), rotation=0)

        cm.ax_heatmap.set_xlabel(xlabel)
        cm.ax_heatmap.set_ylabel(ylabel)

        if annot_cmap is not None:
            draw_legends(cm, annot_cmap, **(legend_kws or {}))

        return cm


class LocIndexer(object):
    def __init__(self, df, class_, extra_args=None):
        self._df = df
        self._class = class_
        self._extra_args = extra_args or {}

    def __getitem__(self, item):
        values = self._df.loc[item]

        if len(values.shape) == 2:
            return self._class(values, **self._extra_args)
        else:
            return values


class FeatureMatrix(AnnotatedMatrix):
    def __init__(self, values, design=None):
        super().__init__(values, design=design)

    def map_ids(self, mapper, **kwargs):
        """Maps gene ids to a different identifier type using genemap."""

        import genemap
        mapped = genemap.map_dataframe(self._values, mapper=mapper, **kwargs)

        return self.__class__(mapped, design=self._design)

    def plot_gene(self, gene, group=None, kind='box', ax=None, **kwargs):
        """Plots distribution of expression for given gene."""

        if group is not None and self._design is None:
            raise ValueError('Grouping is not possible if no design is given')

        # Determine plot type.
        if kind == 'box':
            plot_func = sns.boxplot
        elif kind == 'swarm':
            plot_func = sns.swarmplot
        elif kind == 'violin':
            plot_func = sns.violinplot
        else:
            raise ValueError('Unknown plot type: {}'.format(kind))

        # Assemble plot data (design + expression values).
        values = self._values.loc[gene].to_frame(name='value')
        plot_data = pd.concat([values, self.design], axis=1)

        # Plot expression.
        ax = plot_func(data=plot_data, x=group, y='value', ax=ax, **kwargs)

        ax.set_title(gene)
        ax.set_ylabel('Value')

        return ax

    def plot_pca(self, components=(1, 2), **kwargs):
        """Plots PCA of samples."""

        # Fit PCA and transform expression.
        n_components = max(components)

        pca = PCA(n_components=max(components))
        transform = pca.fit_transform(self.values.values.T)

        transform = pd.DataFrame(
            transform,
            columns=['pca_{}'.format(i + 1) for i in range(n_components)],
            index=self.values.columns)

        # Assemble plot data.
        plot_data = pd.concat([transform, self.design], axis=1)

        # Draw using lmplot.
        pca_x, pca_y = ['pca_{}'.format(c) for c in components]
        grid = sns.lmplot(
            data=plot_data, x=pca_x, y=pca_y, fit_reg=False, **kwargs)

        grid.ax.set_xlabel('Component ' + str(components[0]))
        grid.ax.set_ylabel('Component ' + str(components[1]))

        return grid

    def plot_pca_variance(self, n_components, ax=None):
        """Plots variance explained by PCA components."""

        pca = PCA(n_components=n_components)
        pca.fit(self.values.values.T)

        if ax is None:
            _, ax = plt.subplots()

        ax.plot(
            np.arange(pca.n_components_) + 1, pca.explained_variance_ratio_)

        ax.set_xlabel('Component')
        ax.set_ylabel('Explained variance')
        sns.despine(ax=ax)

        return ax


class RegionMatrix(AnnotatedMatrix):
    def __init__(self, values, design=None):
        values = GenomicDataFrame(values, use_index=True)
        super().__init__(values, design=design)

    @classmethod
    def from_csv(cls,
                 file_path,
                 design=None,
                 name_map=None,
                 drop_cols=None,
                 expand_index=False,
                 **kwargs):
        default_kws = {'index_col': 0 if expand_index else [0, 1, 2]}
        kwargs = {**default_kws, **kwargs}

        values = cls._read_csv(
            file_path,
            design=design,
            name_map=name_map,
            drop_cols=drop_cols,
            **kwargs)

        if expand_index:
            values.index = _expand_index(
                values, names=['chromosome', 'start', 'end'])

        return cls(values, design=design)

    def resample(self, bin_size, agg='mean'):
        """Resamples chromosomal interval at given interval by binning."""

        # Perform resampling per chromosome.
        resampled = pd.concat(
            (self._resample_chrom(grp, bin_size=bin_size, agg=agg)
             for _, grp in self._values.groupby(level=0)),
            axis=0)  # yapf: disable

        # Restore original index order.
        resampled = resampled.reindex(self._values.gi.chromosomes, level=0)

        return self.__class__(resampled)

    @staticmethod
    def _resample_chrom(values, bin_size, agg):
        # Bin rows by their centre positions.
        positions = ((values.gi.start + values.gi.end) // 2).values

        range_start = values.gi.start.min()
        range_end = values.gi.end.max() + bin_size

        bins = np.arange(range_start, range_end, bin_size)
        binned = pd.cut(positions, bins=bins)

        # Resample.
        resampled = values.groupby(binned).agg(agg)

        # Set index.
        resampled['chromosome'] = values.index[0][0]
        resampled['start'] = bins[:-1]
        resampled['end'] = bins[1:] - 1
        resampled = resampled.set_index(['chromosome', 'start', 'end'])

        return resampled

    def expand(self, fill_value=None):
        """Expands value matrix to include all bins within range."""

        expanded = self._expand(self._values, fill_value=fill_value)
        return self.__class__(expanded)

    @staticmethod
    def _expand(values, fill_value=None):
        """Internal function for expanding values.

        Used by expand and impute methods.
        """

        def _chrom_index(grp, bin_size):
            chrom = grp.index[0][0]
            start = grp.index.get_level_values(1).min()
            end = grp.index.get_level_values(2).max()

            bins = np.arange(start - 1, end + 1, step=bin_size)

            starts = bins[:-1] + 1
            ends = bins[1:]

            return zip(cycle([chrom]), starts, ends)

        bin_size = (values.gi.end[0] - values.gi.start[0]) + 1

        indices_per_chrom = (_chrom_index(grp, bin_size)
                             for _, grp in values.groupby(level=0))
        indices = list(chain.from_iterable(indices_per_chrom))

        expanded = values.loc[indices]

        if fill_value is not None:
            expanded = expanded.fillna(fill_value)

        return expanded

    def impute(self, window=11, min_probes=5):
        """Imputes nan values from neighboring bins."""

        values = self._expand(self._values)

        # Calculate median value within window (allowing for
        # window - min_probes number of NAs within the window).
        rolling = values.rolling(
            window=window, min_periods=min_probes, center=True)
        avg_values = rolling.median()

        # Copy over values for null rows for the imputation.
        imputed = values.copy()

        mask = imputed.isnull().all(axis=1)
        imputed.loc[mask] = avg_values.loc[mask]

        return self.__class__(imputed)

    def subset(self, chromosomes):
        """Subsets regions to given chromosomes."""

        values = self._values.gi.subset(chromosomes)
        return self.__class__(values)

    def annotate(self, genes, id_='gene_id', verbose=False):
        """Annotates region values for gene."""

        # Determine chrom/start/end columns.
        get_chrom = attrgetter(genes.gi.chromosome_col)
        get_start = attrgetter(genes.gi.start_col)
        get_end = attrgetter(genes.gi.end_col)
        get_id = attrgetter(id_)

        # Wrap gene tuples if needed.
        gene_rows = genes.itertuples()

        if verbose:
            gene_rows = tqdm(gene_rows, total=genes.shape[0])

        # Determine calls.
        gene_calls = {}
        for gene in gene_rows:
            try:
                overlap = self._values.gi.search(
                    get_chrom(gene), get_start(gene), get_end(gene))
                gene_calls[get_id(gene)] = overlap.median()
            except KeyError:
                pass

        # Assemble into dataframe.
        annotated = pd.DataFrame.from_records(gene_calls).T
        annotated.index.name = 'gene'

        return FeatureMatrix(annotated)

    def plot_sample(self, sample, ax=None, **kwargs):
        """Plots values for given sample along genomic axis."""
        ax = plot_genomic(self.values, y=sample, ax=ax, **kwargs)
        return ax

    def plot_heatmap(self,
                     cmap='RdBu_r',
                     annotation=None,
                     metric='euclidean',
                     method='complete',
                     transpose=True,
                     cluster=True,
                     **kwargs):
        """Plots heatmap of gene expression over samples."""

        if 'row_cluster' in kwargs or 'col_cluster' in kwargs:
            raise ValueError(
                'RegionMatrix only supports clustering by region. '
                'Use the \'cluster\' argument to specify whether '
                'clustering should be performed.')

        if cluster:
            # Do clustering on matrix with only finite values.
            values_clust = self._values.replace([np.inf, -np.inf], np.nan)
            values_clust = values_clust.dropna()

            dist = pdist(values_clust.T, metric=metric)
            sample_linkage = linkage(dist, method=method)
        else:
            sample_linkage = None

        # Draw heatmap.
        heatmap_kws = dict(kwargs)
        if transpose:
            heatmap_kws.update({
                'row_cluster': sample_linkage is not None,
                'row_linkage': sample_linkage,
                'col_cluster': False
            })
        else:
            heatmap_kws.update({
                'col_cluster': sample_linkage is not None,
                'col_linkage': sample_linkage,
                'row_cluster': False
            })

        cm = super().plot_heatmap(
            cmap=cmap,
            annotation=annotation,
            metric=metric,
            method=method,
            transpose=transpose,
            **heatmap_kws)

        self._style_axis(cm, transpose=transpose)

    def _style_axis(self, cm, transpose):
        chrom_breaks = self._values.groupby(level=0).size().cumsum()

        chrom_labels = self._values.gi.chromosomes
        chrom_label_pos = np.concatenate([[0], chrom_breaks])
        chrom_label_pos = (chrom_label_pos[:-1] + chrom_label_pos[1:]) / 2

        if transpose:
            cm.ax_heatmap.set_xticks([])

            for loc in chrom_breaks[:-1]:
                cm.ax_heatmap.axvline(loc, color='grey', lw=1)

            cm.ax_heatmap.set_xticks(chrom_label_pos)
            cm.ax_heatmap.set_xticklabels(chrom_labels, rotation=0)

            cm.ax_heatmap.set_xlabel('Genomic position')
            cm.ax_heatmap.set_ylabel('Samples')
        else:
            cm.ax_heatmap.set_yticks([])

            for loc in chrom_breaks[:-1]:
                cm.ax_heatmap.axhline(loc, color='grey', lw=1)

            cm.ax_heatmap.set_yticks(chrom_label_pos)
            cm.ax_heatmap.set_yticklabels(chrom_labels, rotation=0)

            cm.ax_heatmap.set_xlabel('Samples')
            cm.ax_heatmap.set_ylabel('Genomic position')

        return cm


def _expand_index(df, regex=None, names=('chromosome', 'start', 'end')):
    if regex is None:
        regex = re.compile(r'(?P<chromosome>\w+):(?P<start>\d+)-(?P<end>\d+)')

    regions = [_parse_region_str(i, regex=regex) for i in df.index]
    return pd.MultiIndex.from_tuples(regions, names=names)


def _parse_region_str(region_str, regex):
    match = regex.search(region_str)

    if match is None:
        raise ValueError('Unable to parse region {!r}'.format(region_str))

    groups = match.groupdict()

    return (groups['chromosome'], int(groups['start']), int(groups['end']))
