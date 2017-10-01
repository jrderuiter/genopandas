import functools
from itertools import chain, cycle
from operator import attrgetter
import re

import numpy as np
import pandas as pd

from pandas.api.types import is_numeric_dtype

from genopandas.plotting.genomic import plot_genomic
from genopandas.plotting.clustermap import color_annotation, draw_legends
from genopandas.plotting.seaborn import scatter
from genopandas.util import with_defaults, lookup, expand_index

from .frame import GenomicDataFrame, GenomicSlice


class AnnotatedMatrix(object):
    """Base AnnotatedMatrix class.

    Annotated matrix classes respresent 2D numeric feature-by-sample matrices
    (with 'features' along the rows and samples along the columns), which can
    be annotated with an optional sample_data frame that describes the
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
    sample_data : pd.DataFrame
        Sample data.
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

    def __init__(self, values, sample_data=None):
        """Constructs an AnnotatedMatrix instance.

        Parameters
        ----------
        values : pd.DataFrame
            Numeric dataframe (matrix) with features along the rows
            and samples along the columns.
        sample_data : pd.DataFrame
            Sample data, with sample names in the index. Sample names
            should correspond with the columns of the matrix value frame.
        """
        assert sample_data is None or (
            values.shape[1] == sample_data.shape[0]
            and all(values.columns == sample_data.index))

        # Check if all columns are numeric.
        for col_name, col_values in values.items():
            if not is_numeric_dtype(col_values):
                raise ValueError('Column {} is not numeric'.format(col_name))

        if sample_data is None:
            sample_data = pd.DataFrame({}, index=values.columns)

        self._values = values
        self._sample_data = sample_data

    def __eq__(self, other):
        if not isinstance(other, GenomicMatrix):
            return False
        return all(self.values == other.values) and \
            all(self.sample_data == other.sample_data)

    @property
    def values(self):
        """Matrix values."""
        return self._values

    @property
    def shape(self):
        """Matrix shape."""
        return self._values.shape

    @property
    def sample_data(self):
        """Sample data."""
        return self._sample_data

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
        """Label-based indexer (similar to pandas .loc)."""
        return LocWrapper(self._values.loc, constructor=self._loc_constructor)

    def _loc_constructor(self, values):
        """Constructor used by LocWrapper."""

        if len(values.shape) != 2:
            return values

        return self.__class__(
            values.copy(), sample_data=self._sample_data.copy())

    @property
    def iloc(self):
        """Index-based indexer (similar to pandas .iloc)."""
        return LocWrapper(self._values.iloc, constructor=self._loc_constructor)

    def __getitem__(self, item):
        values = self._values[item]

        if len(values.shape) != 2:
            return values

        sample_data = self._sample_data.reindex(index=item)
        return self.__class__(values.copy(), sample_data=sample_data)

    @classmethod
    def from_csv(cls,
                 file_path,
                 sample_data=None,
                 drop_cols=None,
                 index_col=None,
                 **kwargs):

        if index_col is None:
            index_col = 0

        values = pd.read_csv(file_path, index_col=index_col, **kwargs)

        values = cls._preprocess_values(
            values, sample_data=sample_data, drop_cols=drop_cols)

        return cls(values, sample_data=sample_data)

    @staticmethod
    def _preprocess_values(values, sample_data=None, drop_cols=None):
        """Preprocesses values to match to sample_data and drop any extra
           (non-numeric) columns.
        """

        if drop_cols is not None:
            values = values.drop(drop_cols, axis=1)

        if sample_data is not None:
            values = values[sample_data.index]

        return values

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

        sample_data = self._sample_data.rename(index=mapping)
        sample_data = sample_data.reindex(index=renamed.columns)

        return self.__class__(renamed, sample_data=sample_data)

    @staticmethod
    def _rename(df, name_map, drop=False):
        renamed = df.rename(columns=name_map)

        if drop:
            extra = set(renamed.columns) - set(name_map.values())
            renamed = renamed.drop(extra, axis=1)

        return renamed

    def query(self, expr):
        """Subsets samples in matrix by querying sample_data with expression.

        Similar to the pandas ``query`` method, this method queries the sample
        data of the matrix with the given boolean expression. Any samples for
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
        sample_data = self._sample_data.query(expr)
        values = self._values.reindex(columns=sample_data.index)
        return self.__class__(values, sample_data=sample_data)

    def dropna(self, subset=None, how='any', thresh=None):
        """Drops samples with NAs in sample_data."""

        sample_data = self._sample_data.dropna(
            subset=subset, how=how, thresh=thresh)
        values = self._values.reindex(columns=sample_data.index)

        return self.__class__(values, sample_data=sample_data)

    def dropna_values(self, axis=0, how='any', thresh=None):
        """Drops rows in the matrix that contain NAs."""

        values = self._values.dropna(axis=axis, how=how, thresh=thresh)
        sample_data = self._sample_data.reindex(index=values.columns)
        return self.__class__(values, sample_data=sample_data)

    @classmethod
    def concat(cls, matrices, axis=0):
        """Concatenates two AnntotatedMatrices."""

        # TODO: Check for overlapping samples?

        assert 0 <= axis <= 1

        value_axis, sample_axis = axis, 1 - axis

        values = pd.concat([mat.values for mat in matrices], axis=value_axis)
        sample_data = pd.concat(
            [mat.sample_data for mat in matrices], axis=sample_axis)

        if any(values.columns.duplicated()):
            raise ValueError('Matrices contain duplicate samples')

        if any(values.index.duplicated()):
            raise ValueError('Matrices contain duplicate features')

        return cls(values, sample_data=sample_data)

    def plot_heatmap(self,
                     cmap='RdBu_r',
                     annotation=None,
                     annotation_colors=None,
                     metric='euclidean',
                     method='complete',
                     transpose=False,
                     legend_kws=None,
                     **kwargs):
        """Plots clustered heatmap of matrix values."""

        import matplotlib.pyplot as plt
        import seaborn as sns

        if annotation is not None:
            annot_colors, annot_cmap = color_annotation(
                self._sample_data[annotation], colors=annotation_colors)
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
        plot_data = pd.concat([transform, self._sample_data], axis=1)

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


class LocWrapper(object):
    """Wrapper class that wraps an objects loc/iloc accessor."""

    def __init__(self, loc, constructor=None):
        if constructor is None:
            constructor = lambda x: x

        self._loc = loc
        self._constructor = constructor

    def __getitem__(self, item):
        result = self._loc[item]
        return self._constructor(result)


class FeatureMatrix(AnnotatedMatrix):
    """Feature matrix class.

    FeatureMatrices are 2D feature-by-sample matrices, in which the rows
    contain measurements for specific (labeled) features across samples.
    Examples are gene expression matrices, in which each row contains the
    expression of a specific gene across different samples.
    """

    def __init__(self, values, sample_data=None):
        super().__init__(values, sample_data=sample_data)

    def map_ids(self, mapper, **kwargs):
        """Maps feature ids to a different identifier type using genemap."""

        import genemap
        mapped = genemap.map_dataframe(self._values, mapper=mapper, **kwargs)

        return self.__class__(mapped, sample_data=self._sample_data)

    def plot_feature(self, feature, group=None, kind='box', ax=None, **kwargs):
        """Plots distribution of expression for given feature."""

        import seaborn as sns

        if group is not None and self._sample_data is None:
            raise ValueError('Grouping not possible if no sample_data given')

        # Determine plot type.
        plot_funcs = {
            'box': sns.boxplot,
            'swarm': sns.swarmplot,
            'violin': sns.violinplot
        }
        plot_func = lookup(plot_funcs, key=kind, label='plot')

        # Assemble plot data (sample_data + expression values).
        values = self._values.loc[feature].to_frame(name='value')
        plot_data = pd.concat([values, self._sample_data], axis=1)

        # Plot expression.
        ax = plot_func(data=plot_data, x=group, y='value', ax=ax, **kwargs)

        ax.set_title(feature)
        ax.set_ylabel('Value')

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
        plot_data = pd.concat([plot_data, self._sample_data], axis=1)

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

        plot_data = pd.concat([values_x, values_y, self._sample_data], axis=1)
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


class GenomicMatrix(AnnotatedMatrix):
    """Base class for matrices indexed by genomic range/position.

    Should not be used directly. Use either ``PositionMatrix`` for
    positioned data or ``RegionMatrix`` for ranged data.
    """

    def __init__(self, values, sample_data=None):
        if not isinstance(values, GenomicDataFrame):
            values = GenomicDataFrame(values)
        super().__init__(values, sample_data=sample_data)

    @property
    def gloc(self):
        """Genomic-position indexer.

        Used to select rows from the matrix by their genomic position.
        Interface is the same as for the GenomicDataFrame gloc property
        (which this method delegates to).
        """

        return GLocWrapper(self._values.gloc, self._gloc_constructor)

    def _gloc_constructor(self, values):
        if isinstance(values, GenomicDataFrame):
            values = self.__class__(
                values.copy(), sample_data=self._sample_data.copy())
        return values

    @classmethod
    def from_csv_condensed(cls,
                           file_path,
                           expression,
                           sample_data=None,
                           drop_cols=None,
                           index_col=0,
                           **kwargs):
        """Reads matrix from CSV with condensed index, in which genomic
           positions are stored in a condensed format (i.e. chrom:start-end)
           in the dataframe.
        """

        # Read values and expand index.
        values = pd.read_csv(file_path, index_col=index_col, **kwargs)
        values.index = expand_index(values.index, expression)

        values = cls._preprocess_values(
            values, sample_data=sample_data, drop_cols=drop_cols)

        return cls(values, sample_data=sample_data)

    def as_segments(self):
        raise NotImplementedError()

    @classmethod
    def _segment_ranged(cls, values):
        values = values.sort_index()

        # Get segments per sample.
        segments = pd.concat(
            (cls._segments_for_sample(sample_values)
             for _, sample_values in values.items()),
            axis=0, ignore_index=True)  # yapf: disable

        segments = segments.set_index(
            ['chromosome', 'start', 'end'], drop=False)

        return GenomicDataFrame(segments)

    @staticmethod
    def _segments_for_sample(sample_values):
        # Calculate segment ids (distinguished by diff values).
        segment_ids = np.cumsum(_padded_diff(sample_values) != 0)

        # Get sample and position columns.
        sample = sample_values.name
        chrom_col, start_col, end_col = sample_values.index.names

        # Group and determine positions + values.
        grouped = sample_values.reset_index().groupby(
            by=[chrom_col, segment_ids])

        segments = grouped.agg({
            chrom_col: 'first',
            start_col: 'min',
            end_col: 'max',
            sample: ['first', 'size']
        })

        # Flatten column levels and rename.
        segments.columns = ['_'.join(s) for s in segments.columns]
        segments = segments.rename(columns={
            chrom_col + '_first': 'chromosome',
            start_col + '_min': 'start',
            end_col + '_max': 'end',
            sample + '_first': 'value',
            sample + '_size': 'size'
        })

        # Add sample name and reorder columns.
        segments = segments.reindex(
            columns=['chromosome', 'start', 'end', 'value', 'size'])
        segments['sample'] = sample

        return segments.reset_index(drop=True)

    def expand(self):
        """Expands matrix to include values from missing bins.

        Assumes rows are regularly spaced with a fixed bin size.
        """

        raise NotImplementedError()

    @staticmethod
    def _expand_ranged(values):
        def _bin_indices(grp, bin_size):
            chrom = grp.index[0][0]

            start = grp.index.get_level_values(1).min()
            end = grp.index.get_level_values(2).max()

            bins = np.arange(start, end + 1, step=bin_size)

            return zip(cycle([chrom]), bins[:-1], bins[1:])

        bin_size = values.index[0][2] - values.index[0][1]

        indices = list(
            chain.from_iterable(
                _bin_indices(grp, bin_size=bin_size)
                for _, grp in values.groupby(level=0)))

        return values.reindex(index=indices)

    def impute(self, window=11, min_probes=5, expand=True):
        """Imputes nan values from neighboring bins."""

        if expand:
            values = self.expand()._values
        else:
            values = self._values

        # Calculate median value within window (allowing for
        # window - min_probes number of NAs within the window).
        rolling = values.rolling(
            window=window, min_periods=min_probes, center=True)
        avg_values = rolling.median()

        # Copy over values for null rows for the imputation.
        imputed = values.copy()

        mask = imputed.isnull().all(axis=1)
        imputed.loc[mask] = avg_values.loc[mask]

        return self.__class__(imputed, sample_data=self._sample_data)

    def resample(self, bin_size, start=0, agg='mean'):
        """Resamples values at given interval by binning."""
        raise NotImplementedError()

    @classmethod
    def _resample_ranged(cls, values, bin_size, start=0, agg='mean'):
        # Perform resampling per chromosome.
        resampled = pd.concat(
            (cls._resample_chromosome(
                grp, bin_size=bin_size, agg=agg, start=start)
             for _, grp in values.groupby(level=0)),
            axis=0)  # yapf: disable

        # Restore original index order.
        resampled = resampled.reindex(values.gloc.chromosomes, level=0)

        return GenomicDataFrame(resampled)

    @staticmethod
    def _resample_chromosome(values, bin_size, start=0, agg='mean'):
        # Bin rows by their centre positions.
        starts = values.index.get_level_values(1)
        ends = values.index.get_level_values(2)

        positions = (starts + ends) // 2

        range_start = start
        range_end = ends.max() + bin_size

        bins = np.arange(range_start, range_end, bin_size)

        if len(bins) < 2:
            raise ValueError('No bins in range ({}, {}) with bin_size {}'.
                             format(range_start, ends.max(), bin_size))

        binned = pd.cut(positions, bins=bins)

        # Resample.
        resampled = values.groupby(binned).agg(agg)
        resampled.index = pd.MultiIndex.from_arrays(
            [[values.index[0][0]] * (len(bins) - 1), bins[:-1], bins[1:]],
            names=values.index.names)

        return resampled

    def annotate(self, features, id_='gene_id'):
        """Annotates values for given features."""

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

        return FeatureMatrix(annotated, sample_data=self._sample_data)

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


class GLocWrapper(object):
    """Wrapper class that wraps gloc indexer from given object."""

    def __init__(self, gloc, constructor):
        self._gloc = gloc
        self._constructor = constructor

    def __getattr__(self, name):
        attr = getattr(self._gloc, name)

        if callable(attr):
            return self._wrap(attr)

        return attr

    def __getitem__(self, item):
        result = self._gloc[item]

        if isinstance(result, GenomicSlice):
            result = GLocSliceWrapper(
                self._gloc, chromosome=item, constructor=self._constructor)
        else:
            result = self._constructor(result)

        return result

    def _wrap(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """Wrapper that calls _constructor on returned result."""
            result = func(*args, **kwargs)
            return self._constructor(result)

        return wrapper


class GLocSliceWrapper(object):
    """Wrapper class that wraps slice from gloc indexer on given object."""

    def __init__(self, gloc, chromosome, constructor):
        self._gloc = gloc
        self._chromosome = chromosome
        self._constructor = constructor

    def __getitem__(self, item):
        result = self._gloc[self._chromosome][item]
        return self._constructor(result)


POSITION_REGEX = r'(?P<chromosome>\w+):(?P<position>\d+)'


class PositionMatrix(GenomicMatrix):
    """AnnotatedMatrix that contains values indexed by genomic positions."""

    def __init__(self, values, sample_data=None):
        if not values.index.nlevels == 2:
            raise ValueError('Values are not positioned')
        super().__init__(values, sample_data=sample_data)

    @classmethod
    def from_csv(cls,
                 file_path,
                 sample_data=None,
                 drop_cols=None,
                 index_col=None,
                 **kwargs):
        return super().from_csv(
            file_path,
            sample_data=sample_data,
            drop_cols=drop_cols,
            index_col=[0, 1] if index_col is None else index_col,
            **kwargs)

    @classmethod
    def from_csv_condensed(cls,
                           file_path,
                           expression=POSITION_REGEX,
                           sample_data=None,
                           drop_cols=None,
                           index_col=0,
                           **kwargs):
        return super().from_csv_condensed(
            file_path,
            expression=expression,
            sample_data=sample_data,
            drop_cols=drop_cols,
            index_col=index_col,
            **kwargs)

    def as_regions(self, width=1):
        """Returns matrix with ranged index."""

        values = self._values.as_ranged(width=width)
        return RegionMatrix(values=values, sample_data=self.sample_data)

    def as_segments(self):
        values = self._values.as_ranged()
        return self._segment_ranged(values)

    def expand(self):
        positions = self._values.index.get_level_values(1)
        bin_size = np.diff(positions).min()

        values = self._values.as_ranged(width=bin_size)
        expanded = self._expand_ranged(values).as_positioned()

        return self.__class__(expanded, sample_data=self._sample_data.copy())

    def resample(self, bin_size, start=0, agg='mean'):
        values = self._values.as_ranged()
        resampled = self._resample_ranged(
            values, bin_size, start=start, agg=agg)
        resampled = resampled.as_positioned()

        return self.__class__(resampled, sample_data=self._sample_data.copy())


REGION_REGEX = r'(?P<chromosome>\w+):(?P<start>\d+)-(?P<end>\d+)'


class RegionMatrix(GenomicMatrix):
    """AnnotatedMatrix that contains values indexed by genomic regions.

    RegionMatrices are 2D region-by-sample matrices, in which the rows
    contain measurements for specific genomic regions (indicated by chromosome
    and start/end positions). Examples are copy-number expression matrices,
    in which each row contains the copy-number values for a specific bin in
    the genome.
    """

    def __init__(self, values, sample_data=None):
        if not values.index.nlevels == 3:
            raise ValueError('Values are not ranged')
        super().__init__(values, sample_data=sample_data)

    @classmethod
    def from_csv(cls,
                 file_path,
                 sample_data=None,
                 drop_cols=None,
                 index_col=None,
                 **kwargs):
        return super().from_csv(
            file_path,
            sample_data=sample_data,
            drop_cols=drop_cols,
            index_col=[0, 1, 2] if index_col is None else index_col,
            **kwargs)

    @classmethod
    def from_csv_condensed(cls,
                           file_path,
                           expression=REGION_REGEX,
                           sample_data=None,
                           drop_cols=None,
                           index_col=0,
                           **kwargs):
        return super().from_csv_condensed(
            file_path,
            expression=expression,
            sample_data=sample_data,
            drop_cols=drop_cols,
            index_col=index_col,
            **kwargs)

    def as_positions(self):
        """Returns matrix with positioned index."""

        values = self._values.as_positioned()
        return PositionMatrix(values=values, sample_data=self.sample_data)

    def as_segments(self):
        return self._segment_ranged(self._values)

    def expand(self):
        expanded = self._expand_ranged(self._values)
        return self.__class__(expanded, sample_data=self._sample_data.copy())

    def resample(self, bin_size, start=0, agg='mean'):
        resampled = self._resample_ranged(
            self._values, bin_size, start=start, agg=agg)
        return self.__class__(resampled, sample_data=self._sample_data.copy())

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
