from itertools import chain, cycle
from operator import attrgetter
import re

import numpy as np
import pandas as pd

from pandas.api.types import is_numeric_dtype

from genopandas.plotting.genomic import plot_genomic
from genopandas.plotting.clustermap import color_annotation, draw_legends
from genopandas.plotting.seaborn import scatter
from genopandas.util import with_defaults, lookup

from .frame import GenomicDataFrame


class AnnotatedMatrix(object):
    """Base AnnotatedMatrix class.

    Annotated matrix classes respresent 2D numeric feature-by-sample matrices
    (with 'features' along the rows and samples along the columns), which can
    be annotated with an optional 'design' frame that describes the respective
    samples. The type of feature varies between different sub-classes, examples
    being genes (for gene expression matrices) and region-based bins (for
    copy-number data).

    This base class mainly contains a variety of methods for querying,
    subsetting and combining different annotation matrices. General plotting
    methods are also provided (``plot_heatmap``).

    Note that the class follows the feature-by-sample convention that is
    typically followed in biological packages, rather than the sample-by-feature
    orientation. This has the additional advantage of allowing more complex
    indices (such as a region-based MultiIndex) for the features, which are
    more difficult to use for DataFrame columns than for rows.

    Attributes
    ----------
    values : pd.DataFrame
        Matrix values.
    design : pd.DataFrame
        Sample design.
    samples : list[str]
        List of samples in the matrix (along the columns).
    features : list[Any]
        List of features in the matrix (along the rows).
    loc : LocIndexer
        Location based indexer for the matrix. Uses similar conventions
        as .loc on pandas DataFrames.

    Examples
    --------
    Subsetting a matrix:


    """

    def __init__(self, values, design=None):
        """Constructs an AnnotatedMatrix instance.

        Parameters
        ----------
        values : pd.DataFrame
            Numeric dataframe (matrix) with features along the rows
            and samples along the columns.
        design : pd.DataFrame
            Sample design, with sample names in the index. Sample names
            should correspond with the columns of the matrix value frame.
        """
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

        return LocIndexer(self)

    def _loc(self, item):
        values = self._values.loc[item]

        if len(values.shape) != 2:
            return values

        return self.__class__(values.copy(), design=self._design.copy())

    def __getitem__(self, item):
        values = self._values[item]

        if len(values.shape) != 2:
            return values

        design = self._design.reindex(index=item)
        return self.__class__(values.copy(), design=design)

    def rename(self, mapping, drop=False):
        """Renames samples using given mapping.

        Method for renaming samples using a dictionary name mapping. Optionally
        drops samples not in mapping (if drop is True).

        Parameters
        ----------
        mapping : Dict[str, str]
            Dictionary mapping old sample names to their new values.
        drop : bool
            Whether samples that are not in the mapping should be dropped.

        Returns
        -------
        AnnotatedMatrix
            Returns a new AnnotatedMatrix instance with the renamed samples.
        """

        renamed = self._rename(self._values, mapping, drop=drop)

        design = self._design.rename(index=mapping)
        design = design.reindex(index=renamed.columns)

        return self.__class__(renamed, design=design)

    @staticmethod
    def _rename(df, name_map, drop=False):
        renamed = df.rename(columns=name_map)

        if drop:
            extra = set(renamed.columns) - set(name_map.values())
            renamed = renamed.drop(extra, axis=1)

        return renamed

    def query(self, expr):
        """Subsets samples in matrix by querying design with given expression.

        Similar to the pandas ``query`` method, this method queries the design
        frame of the matrix with the given boolean expression. Any samples for
        which the expression evaluates to True are returned in the resulting
        AnnotatedMatrix.

        Parameters
        ----------
        expr : str
            The query string to evaluate. You can refer to variables in the
            environment by prefixing them with an ‘@’ character like @a + b.

        Returns
        -------
        AnnotatedMatrix
            Subsetted matrix, containing only the samples for which ``expr``
            evaluates to True.

        """
        design = self._design.query(expr)
        values = self._values.reindex(columns=design.index)
        return self.__class__(values, design=design)

    def dropna(self, subset=None, how='any', thresh=None):
        design = self._design.dropna(subset=subset, how=how, thresh=thresh)
        values = self._values.reindex(columns=design.index)
        return self.__class__(values, design=design)

    def dropna_values(self, how='any', thresh=None):
        values = self._values.dropna(how=how, thresh=thresh)
        return self.__class__(values, design=self._design.copy())

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
                 index_col=None,
                 **kwargs):

        if index_col is None:
            index_col = 0

        values = cls._read_csv(
            file_path,
            index_col,
            design=design,
            name_map=name_map,
            drop_cols=drop_cols,
            **kwargs)

        return cls(values, design=design)

    @classmethod
    def _read_csv(cls,
                  file_path,
                  index_col,
                  design=None,
                  name_map=None,
                  drop_cols=None,
                  **kwargs):
        values = pd.read_csv(file_path, index_col=index_col, **kwargs)

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

        import matplotlib.pyplot as plt
        import seaborn as sns

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
    def __init__(self, obj):
        self._obj = obj

    def __getitem__(self, item):
        return self._obj._loc(item)


class FeatureMatrix(AnnotatedMatrix):
    """Feature matrix class.

    FeatureMatrices are 2D feature-by-sample matrices, in which the rows
    contain measurements for specific (labeled) features across samples.
    Examples are gene expression matrices, in which each row contains the
    expression of a specific gene across different samples.
    """

    def __init__(self, values, design=None):
        super().__init__(values, design=design)

    def map_ids(self, mapper, **kwargs):
        """Maps feature ids to a different identifier type using genemap."""

        import genemap
        mapped = genemap.map_dataframe(self._values, mapper=mapper, **kwargs)

        return self.__class__(mapped, design=self._design)

    def plot_feature(self, feature, group=None, kind='box', ax=None, **kwargs):
        """Plots distribution of expression for given feature."""

        import seaborn as sns

        if group is not None and self._design is None:
            raise ValueError('Grouping is not possible if no design is given')

        # Determine plot type.
        plot_funcs = {
            'box': sns.boxplot,
            'swarm': sns.swarmplot,
            'violin': sns.violinplot
        }
        plot_func = lookup(plot_funcs, key=kind, label='plot')

        # Assemble plot data (design + expression values).
        values = self._values.loc[feature].to_frame(name='value')
        plot_data = pd.concat([values, self.design], axis=1)

        # Plot expression.
        ax = plot_func(data=plot_data, x=group, y='value', ax=ax, **kwargs)

        ax.set_title(feature)
        ax.set_ylabel('Value')

        return ax

    def plot_pca(self, components=(1, 2), ax=None, **kwargs):
        """Plots PCA of samples."""

        try:
            from sklearn.decomposition import PCA
        except ImportError:
            raise ImportError('Scikit-learn must be installed to '
                              'perform PCA analyses')

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
        ax = scatter(data=plot_data, x=pca_x, y=pca_y, ax=ax, **kwargs)

        ax.set_xlabel('Component ' + str(components[0]))
        ax.set_ylabel('Component ' + str(components[1]))

        return ax

    def plot_pca_variance(self, n_components, ax=None):
        """Plots variance explained by PCA components."""

        import matplotlib.pyplot as plt
        import seaborn as sns

        try:
            from sklearn.decomposition import PCA
        except ImportError:
            raise ImportError('Scikit-learn must be installed to '
                              'perform PCA analyses')

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

    def plot_correlation(self,
                         x,
                         y,
                         test=None,
                         kind='scatter',
                         annotate_kws=None,
                         **kwargs):
        """Plots correlation between the values of two features."""

        plot_data = self._values.T[[x, y]]
        plot_data = pd.concat([plot_data, self._design], axis=1)

        return self._plot_correlation(
            plot_data,
            x,
            y,
            test=test,
            kind=kind,
            annotate_kws=annotate_kws,
            **kwargs)

    @staticmethod
    def _plot_correlation(data,
                          x,
                          y,
                          test=None,
                          annotate_kws=None,
                          ax=None,
                          kind='scatter',
                          **kwargs):

        # Determine plot type.
        import seaborn as sns

        plot_funcs = {
            'box': sns.boxplot,
            'swarm': sns.swarmplot,
            'violin': sns.violinplot,
            'scatter': scatter
        }
        plot_func = lookup(plot_funcs, key=kind, label='plot')

        # Create plot.
        ax = plot_func(data=data, x=x, y=y, ax=ax, **kwargs)

        if test is not None:
            from scipy import stats

            # Perform stastical test.
            test_funcs = {
                'spearman': stats.spearmanr,
                'pearson': stats.pearsonr
            }
            test_func = lookup(test_funcs, key=test, label='test')

            corr, pvalue = test_func(data[x], data[y])

            # Draw annotation.
            default_annotate_kws = {
                'xy': (0.05, 0.88),
                'xycoords': 'axes fraction',
                'ha': 'left'
            }

            ax.annotate(
                s='corr = {:3.2f}\np = {:3.2e}'.format(corr, pvalue),
                **with_defaults(annotate_kws, default_annotate_kws))

        return ax

    def plot_correlation_other(self,
                               other,
                               x,
                               y,
                               test=None,
                               kind='scatter',
                               annotate_kws=None,
                               suffixes=('_self', '_other'),
                               **kwargs):

        values_x = self.loc[x]
        values_y = other.loc[y]

        if x == y:
            x = values_x.name + suffixes[0]
            values_x.name = x

            y = values_y.name + suffixes[1]
            values_y.name = y

        plot_data = pd.concat([values_x, values_y, self._design], axis=1)
        plot_data = plot_data.dropna(subset=[x, y])

        if plot_data.shape[0] == 0:
            raise ValueError('Datasets have no samples in common')

        return self._plot_correlation(
            plot_data,
            x,
            y,
            test=test,
            kind=kind,
            annotate_kws=annotate_kws,
            **kwargs)


class RegionMatrix(AnnotatedMatrix):
    """AnnotatedMatrix that contains values indexed by genomic regions.

    RegionMatrices are 2D region-by-sample matrices, in which the rows
    contain measurements for specific genomic regions (indicated by chromosome
    and start/end positions). Examples are copy-number expression matrices,
    in which each row contains the copy-number values for a specific bin in
    the genome.
    """

    def __init__(self, values, design=None):
        values = GenomicDataFrame(values, use_index=True)
        super().__init__(values, design=design)

    @classmethod
    def from_csv(cls,
                 file_path,
                 design=None,
                 name_map=None,
                 drop_cols=None,
                 index_col=None,
                 expand_index=False,
                 **kwargs):

        if index_col is None:
            index_col = 0 if expand_index else [0, 1, 2]

        values = cls._read_csv(
            file_path,
            index_col=index_col,
            design=design,
            name_map=name_map,
            drop_cols=drop_cols,
            **kwargs)

        if expand_index:
            values.index = _expand_index(
                values, names=['chromosome', 'start', 'end'])

        return cls(values, design=design)

    @classmethod
    def from_position(cls, values, design=None, width=1):
        # Build index.
        positions = values.index.get_level_values(1)
        start = positions - (width // 2)
        end = positions + ((width // 2) + 1)

        chromosome = values.index.get_level_values(0)
        tuples = zip(chromosome, start, end)

        index = pd.MultiIndex.from_tuples(
            list(tuples), names=['chromosome', 'start', 'end'])

        # Create and sort frame.
        values = values.copy()
        values.index = index

        values = values.sort_index()

        return cls(values, design=design)

    def to_position(self):
        positions = (self._values.gi.start + self._values.gi.end) // 2
        tuples = zip(self._values.gi.chromosome, positions)

        index = pd.MultiIndex.from_tuples(
            list(tuples), names=['chromosome', 'position'])

        values = self._values.copy()
        values.index = index

        return values

    def to_igv(self, file_path, feature_fmt='BIN_{}', data_type=None):
        """Writes dataframe to numerical format for viewing in IGV.

        Parameters
        ----------
        file_path : str
            Desination path for output file.
        data_type : str
            Datatype to include in the header, indicates to IGV what kind
            of data the file contains. Valid values are: COPY_NUMBER,
            GENE_EXPRESSION, CHIP, DNA_METHYLATION,
            ALLELE_SPECIFIC_COPY_NUMBER, LOH, RNAI
        """

        # Set index names correctly.
        igv_data = self.values.copy()
        igv_data.index.names = 'chromosome', 'start', 'end'

        # Add feature column to index.
        igv_data['feature'] = [
            feature_fmt.format(i + 1) for i in range(igv_data.shape[0])
        ]
        igv_data.set_index('feature', append=True)

        with open(file_path, 'w') as file_:
            if data_type is not None:
                print('#type=' + data_type, file=file_)
                igv_data.to_csv(file_, sep='\t', index=True)

    def as_segments(self):
        """Converts matrix to segments by collapsing stretches with same values.

        Returns
        -------
        GenomicDataFrame
            Dataframe containing the same-value segments per sample. Contains
            'chromosome', 'start', 'end', 'value' and 'sample' columns,
            indicating the range, value and sample of each corresponding
            segment. The sample column indicates to which sample a segment
            belongs.

        """

        sample_segments = (
            self._segments_for_sample(values).assign(sample=sample)
            for sample, values in self._values.items())  # yapf: disable
        segments = pd.concat(sample_segments, axis=0, ignore_index=True)

        # Order segments on chromosome and start position.
        segments['chromosome'] = pd.Categorical(
            segments['chromosome'], categories=self._values.gi.chromosomes)

        segments = segments.dropna(subset=['chromosome'])
        segments = segments.sort_values(['sample', 'chromosome', 'start'])

        # Drop existing index after sort.
        segments = segments.reset_index(drop=True)

        # Convert chromosome to str (no categorial).
        segments['chromosome'] = segments['chromosome'].astype(str)

        return GenomicDataFrame(segments)

    @staticmethod
    def _segments_for_sample(segment_value_col):
        # TODO: Sort by index?

        # Aggregate by segment groups.
        diff = _padded_diff(segment_value_col, pad_value=0)
        segment_ids = np.cumsum(diff != 0)

        grouped = (segment_value_col.reset_index().rename(
            columns={segment_value_col.name: 'value'})
                   .groupby(['chromosome', segment_ids]))

        segments = grouped.agg({
            'chromosome': 'first',
            'start': 'min',
            'end': 'max',
            'value': ['first', 'size']
        })

        # Rename columns to flat index.
        segments.columns = ['_'.join(s) for s in segments.columns]
        segments = segments.rename(columns={
            'chromosome_first': 'chromosome',
            'start_min': 'start',
            'end_max': 'end',
            'value_first': 'value',
            'value_size': 'size'
        })

        # Clean-up result.
        segments = segments.reindex(
            columns=['chromosome', 'start', 'end', 'value', 'size'])

        segments['sample'] = segment_value_col.name
        segments = segments.reset_index(drop=True)

        return segments

    def as_markers(self, name_fmt='Marker_{}'):
        """Converts matrix to a list of probe markers."""

        # Build marker frame.
        markers = pd.DataFrame({
            'chromosome': self._values.gi.chromosome,
            'position': (self._values.gi.start + self._values.gi.end) // 2
        })  # yapf: disable

        # Sort by chromosome and position.
        markers['chromosome'] = pd.Categorical(
            markers['chromosome'], categories=self._values.gi.chromosomes)

        markers = markers.sort_values(['chromosome', 'position'])
        markers['chromosome'] = markers['chromosome'].astype(str)

        # Add marker names.
        markers['name'] = [name_fmt.format(i + 1)
                           for i in range(self._values.shape[0])]# yapf: disable
        markers = markers.reindex(columns=['name', 'chromosome', 'position'])

        markers = markers.reset_index(drop=True)

        return markers

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
        return self.__class__(expanded, design=self._design)

    @staticmethod
    def _expand(values, fill_value=None):
        """Internal function for expanding values.

        Used by ``expand`` and ``impute`` methods. See ``expand``
        documentation for more details .
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

        return self.__class__(imputed, design=self._design)

    def subset(self, chromosomes):
        """Subsets regions to given chromosomes."""

        values = self._values.gi.subset(chromosomes)
        return self.__class__(values, design=self._design)

    def annotate(self, features, id_='gene_id'):
        """Annotates region values for given features."""

        # Setup getters for chrom/start/end columns.
        get_chrom = attrgetter(features.gi.chromosome_col)
        get_start = attrgetter(features.gi.start_col)
        get_end = attrgetter(features.gi.end_col)
        get_id = attrgetter(id_)

        # Calculate calls.
        annotated_calls = {}
        for feature in features.itertuples():
            try:
                overlap = self._values.gi.search(
                    get_chrom(feature), get_start(feature), get_end(feature))
                annotated_calls[get_id(feature)] = overlap.median()
            except KeyError:
                pass

        # Assemble into dataframe.
        annotated = pd.DataFrame.from_records(annotated_calls).T
        annotated.index.name = 'gene'

        return FeatureMatrix(annotated, design=self._design)

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
            from scipy.spatial.distance import pdist
            from scipy.cluster.hierarchy import linkage

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

        return cm

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


def _padded_diff(values, pad_value=0):
    """Same as np.diff, with leading 0 to keep same length as input."""
    diff = np.diff(values)
    return np.pad(
        diff, pad_width=(1, 0), mode='constant', constant_values=pad_value)
