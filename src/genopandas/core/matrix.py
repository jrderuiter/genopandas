import functools
import itertools
import operator
import re

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import toolz

from genopandas import plotting as gplot
from genopandas.util.pandas_ import DfWrapper

from .frame import GenomicDataFrame, GenomicSlice

RANGED_REGEX = r'(?P<chromosome>\w+):(?P<start>\d+)-(?P<end>\d+)'
POSITIONED_REGEX = r'(?P<chromosome>\w+):(?P<position>\d+)'


class AnnotatedMatrix(DfWrapper):
    """AnnotatedMatrix class.

    Annotated matrix classes respresent 2D numeric feature-by-sample matrices
    (with 'features' along the rows and samples along the columns), which can
    be annotated with optional sample_data and feature_data frames that
    describe the samples. The type of feature varies between different
    sub-classes, examples being genes (for gene expression matrices) and
    region-based bins (for copy-number data).

    This (base) class mainly contains a variety of methods for querying,
    subsetting and combining different annotation matrices. General plotting
    methods are also provided (``plot_heatmap``).

    Note that the class follows the feature-by-sample convention that is
    typically followed in biological packages, rather than the sample-by-feature
    orientation. This has the additional advantage of allowing more complex
    indices (such as a region-based MultiIndex) for the features, which are
    more difficult to use for DataFrame columns than for rows.

    Attributes
    ----------
    values : pd.DataFrame or AnnotatedMatrix
        Matrix values.
    sample_data : pd.DataFrame
        DataFrame containing sample annotations, whose index corresponds
        with the columns of the matrix.
    feature_data : pd.DataFrame
        DataFrame containing feature annotations, whose index corresponds
        with the rows of the matrix.

    """

    def __init__(self, values, sample_data=None, feature_data=None):
        if isinstance(values, AnnotatedMatrix):
            # Copy values from existing matrix (only copies sample/feature
            # data if these are not given explictly).
            sample_data = sample_data or values.sample_data
            feature_data = feature_data or values.feature_data
            values = values.values.copy()
        else:
            # Create empty annotations if none given.
            if sample_data is None:
                sample_data = pd.DataFrame({}, index=values.columns)

            if feature_data is None:
                feature_data = pd.DataFrame({}, index=values.index)

            # Check {sample,feature}_data.
            # assert (values.shape[1] == sample_data.shape[0]
            #         and all(values.columns == sample_data.index))

            # assert (values.shape[0] == feature_data.shape[0]
            #         and all(values.index == feature_data.index))

            # Check if all matrix columns are numeric.
            for col_name, col_values in values.items():
                if not is_numeric_dtype(col_values):
                    raise ValueError(
                        'Column {} is not numeric'.format(col_name))

        super().__init__(values)

        self._sample_data = sample_data.reindex(index=values.columns)
        self._feature_data = feature_data.reindex(index=values.index)

    def _constructor(self, values):
        """Constructor that attempts to build new instance
           from given values."""

        if isinstance(values, pd.DataFrame):
            return self.__class__(
                values.copy(),
                sample_data=self._sample_data,
                feature_data=self._feature_data)

        return values

    @property
    def feature_data(self):
        return self._feature_data

    @feature_data.setter
    def feature_data(self, value):
        value = value.reindex(index=self._values.index)
        self._feature_data = value

    @property
    def sample_data(self):
        return self._sample_data

    @sample_data.setter
    def sample_data(self, value):
        value = value.reindex(index=self._values.columns)
        self._sample_data = value

    @classmethod
    def from_csv(cls,
                 file_path,
                 sample_data=None,
                 feature_data=None,
                 sample_mapping=None,
                 feature_mapping=None,
                 drop_cols=None,
                 read_data_kws=None,
                 **kwargs):

        default_kwargs = {'index_col': 0}
        kwargs = toolz.merge(default_kwargs, kwargs)

        values = pd.read_csv(str(file_path), **kwargs)

        # If sample/feature_data are not dataframes, assume they are
        # file paths or objects and try to read from them.
        read_data_kws_default = {
            'sep': kwargs.pop('sep', None),
            'index_col': 0
        }
        read_data_kws = toolz.merge(read_data_kws_default, read_data_kws or {})

        if not (sample_data is None or isinstance(sample_data, pd.DataFrame)):
            sample_data = pd.read_csv(sample_data, **read_data_kws)

        if not (feature_data is None
                or isinstance(feature_data, pd.DataFrame)):
            feature_data = pd.read_csv(feature_data, **read_data_kws)

        values = cls._preprocess_values(
            values,
            sample_data=sample_data,
            feature_data=feature_data,
            sample_mapping=sample_mapping,
            feature_mapping=feature_mapping,
            drop_cols=drop_cols)

        return cls(values, sample_data=sample_data, feature_data=feature_data)

    @classmethod
    def _preprocess_values(cls,
                           values,
                           sample_data=None,
                           feature_data=None,
                           sample_mapping=None,
                           feature_mapping=None,
                           drop_cols=None):
        """Preprocesses matrix to match given sample/feature data."""

        # Drop extra columns (if needed).
        if drop_cols is not None:
            values = values.drop(drop_cols, axis=1)

        # Rename samples/features using mappings (if given).
        if sample_mapping is not None or feature_mapping is not None:
            values = values.rename(
                columns=sample_mapping, index=feature_mapping)

        # Reorder values to match annotations.
        sample_order = None if sample_data is None else sample_data.index
        feat_order = None if feature_data is None else feature_data.index

        values = values.reindex(
            columns=sample_order, index=feat_order, copy=False)

        return values

    def to_csv(self,
               file_path,
               sample_data_path=None,
               feature_data_path=None,
               **kwargs):
        """Writes matrix values to a csv file, using pandas' to_csv method."""

        # Write matrix values.
        self._values.to_csv(file_path, **kwargs)

        # Write sample/feature data if paths given.
        if sample_data_path is not None:
            self._sample_data.to_csv(
                sample_data_path, sep=kwargs.pop('sep', None), index=True)

        if feature_data_path is not None:
            self._feature_data.to_csv(
                feature_data_path, sep=kwargs.pop('sep', None), index=True)

    def rename(self, index=None, columns=None):
        """Rename samples/features in the matrix."""

        renamed = self._values.rename(index=index, columns=columns)

        if index is not None:
            feature_data = self._feature_data.rename(index=index)
        else:
            feature_data = self._feature_data

        if columns is not None:
            sample_data = self._sample_data.rename(index=columns)
        else:
            sample_data = self._sample_data

        return self.__class__(
            renamed, feature_data=feature_data, sample_data=sample_data)

    def melt(self,
             with_sample_data=False,
             with_feature_data=False,
             value_name='value'):
        """Melts values into 'tidy' format, optionally including annotation."""

        feat_col = self._feature_data.index.name or 'feature'
        sample_col = self._sample_data.index.name or 'sample'

        values_long = pd.melt(
            self._values.rename_axis(feat_col).reset_index(),
            id_vars=feat_col,
            var_name=sample_col,
            value_name=value_name)

        if with_sample_data and self._sample_data.shape[1] > 0:
            sample_data = (self._sample_data.rename_axis(sample_col)
                           .reset_index())

            values_long = pd.merge(
                values_long, sample_data, how='left', on=sample_col)

        if with_feature_data and self._feature_data.shape[1] > 0:
            feature_data = (self._feature_data.rename_axis(feat_col)
                            .reset_index())

            # Merge with annotation.
            values_long = pd.merge(
                values_long, feature_data, how='left', on=feat_col)

        return values_long

    def query_samples(self, expr):
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

        return self.__class__(
            values, sample_data=sample_data, feature_data=self._feature_data)

    def dropna_samples(self, subset=None, how='any', thresh=None):
        """Drops samples with NAs in sample_data."""

        sample_data = self._sample_data.dropna(
            subset=subset, how=how, thresh=thresh)
        values = self._values.reindex(columns=sample_data.index)

        return self.__class__(
            values, sample_data=sample_data, feature_data=self._feature_data)

    def __eq__(self, other):
        if not isinstance(other, AnnotatedMatrix):
            return False
        return all(self.values == other.values) and \
            all(self.sample_data == other.sample_data) and \
            all(self.feature_data == other.feature_data)

    def plot_heatmap(
            self,
            cmap='RdBu_r',
            sample_cols=None,
            sample_colors=None,
            feature_cols=None,
            feature_colors=None,
            metric='euclidean',
            method='complete',
            transpose=False,
            # legend_kws=None,
            **kwargs):
        """Plots clustered heatmap of matrix values."""

        import matplotlib.pyplot as plt
        import seaborn as sns

        if sample_cols is not None:
            sample_annot, _ = gplot.color_annotation(
                self._sample_data[sample_cols], colors=sample_colors)
        else:
            sample_annot, _ = None, None

        if feature_cols is not None:
            feature_annot, _ = gplot.color_annotation(
                self._feature_data[feature_cols], colors=feature_colors)
        else:
            feature_annot, _ = None, None

        clustermap_kws = dict(kwargs)

        if transpose:
            values = self._values.T
            clustermap_kws['row_colors'] = sample_annot
            clustermap_kws['col_colors'] = feature_annot
            xlabel, ylabel = 'Features', 'Samples'
        else:
            values = self._values
            clustermap_kws['col_colors'] = sample_annot
            clustermap_kws['row_colors'] = feature_annot
            xlabel, ylabel = 'Samples', 'Features'

        cm = sns.clustermap(
            values, cmap=cmap, metric=metric, method=method, **clustermap_kws)

        plt.setp(cm.ax_heatmap.get_yticklabels(), rotation=0)

        cm.ax_heatmap.set_xlabel(xlabel)
        cm.ax_heatmap.set_ylabel(ylabel)

        # TODO: handle legend drawing.
        #if annot_cmap is not None:
        #    draw_legends(cm, annot_cmap, **(legend_kws or {}))

        return cm

    def pca(self,
            n_components=None,
            axis='columns',
            transform=False,
            with_annotation=False):
        """Performs PCA on matrix."""

        try:
            from sklearn.decomposition import PCA
        except ImportError:
            raise ImportError('Scikit-learn must be installed to '
                              'perform PCA analyses')

        # Fit PCA and transform expression.
        pca = PCA(n_components=n_components)

        if axis in {1, 'columns', 'samples'}:
            values = self._values.T
            annotation = self._sample_data
        elif axis in {0, 'index', 'features'}:
            values = self._values
            annotation = self._feature_data
        else:
            raise ValueError('Unknown value for axis')

        pca.fit(values.values)

        if transform:
            transformed = pca.transform(values.values)

            n_components = transformed.shape[1]

            transformed = pd.DataFrame(
                transformed,
                columns=['pca_{}'.format(i + 1) for i in range(n_components)],
                index=values.index)

            if with_annotation:
                transformed = pd.concat([transformed, annotation], axis=1)

            return pca, transformed
        else:
            return pca

    def plot_pca(self, components=(1, 2), axis='columns', ax=None, **kwargs):
        """Plots PCA of samples."""

        pca, transformed = self.pca(
            n_components=max(components),
            axis=axis,
            transform=True,
            with_annotation=True)

        # Draw using lmplot.
        pca_x, pca_y = ['pca_{}'.format(c) for c in components]
        ax = gplot.scatter_plot(
            data=transformed, x=pca_x, y=pca_y, ax=ax, **kwargs)

        var = pca.explained_variance_ratio_[components[0] - 1] * 100
        ax.set_xlabel('Component {} ({:3.1f}%)'.format(components[0], var))

        var = pca.explained_variance_ratio_[components[1] - 1] * 100
        ax.set_ylabel('Component {} ({:3.1f}%)'.format(components[1], var))

        return ax

    def plot_pca_variance(self, n_components=None, axis='columns', ax=None):
        """Plots variance explained by PCA components."""

        import matplotlib.pyplot as plt
        import seaborn as sns

        pca = self.pca(n_components=n_components, axis=axis, transform=False)

        if ax is None:
            _, ax = plt.subplots()

        x = np.arange(pca.n_components_) + 1
        y = pca.explained_variance_ratio_
        ax.plot(x[:len(y)], y)

        ax.set_xlabel('Component')
        ax.set_ylabel('Explained variance')
        sns.despine(ax=ax)

        return ax

    def plot_feature(self, feature, group=None, kind='box', ax=None, **kwargs):
        """Plots distribution of expression for given feature."""

        import seaborn as sns

        if group is not None and self._sample_data.shape[1] == 0:
            raise ValueError('Grouping not possible without sample data')

        # Determine plot type.
        plot_funcs = {
            'box': sns.boxplot,
            'swarm': sns.swarmplot,
            'violin': sns.violinplot
        }

        try:
            plot_func = plot_funcs[kind]
        except KeyError:
            raise ValueError('Unknown plot type {!r}'.format(kind))

        # Assemble plot data (sample_data + expression values).
        values = self._values.loc[feature].to_frame(name='value')
        plot_data = pd.concat([values, self._sample_data], axis=1)

        # Plot expression.
        ax = plot_func(data=plot_data, x=group, y='value', ax=ax, **kwargs)

        ax.set_title(feature)
        ax.set_ylabel('Value')

        return ax

    @classmethod
    def concat(cls, matrices, axis):
        """Concatenates matrices along given axis."""

        # Collect value/sample/feature data.
        tuples = ((mat.values, mat.sample_data, mat.feature_data)
                  for mat in matrices)
        value_list, sample_list, feat_list = zip(*tuples)

        # Merge values.
        values = pd.concat(value_list, axis=axis)

        # Merge sample/feature data.
        if axis == 'index' or axis == 0:
            sample_data = pd.concat(sample_list, axis='columns')
            feature_data = pd.concat(feat_list, axis='index')
        elif axis == 'columns' or axis == 1:
            sample_data = pd.concat(sample_list, axis='index')
            feature_data = pd.concat(feat_list, axis='columns')
        else:
            raise ValueError('Unknown value for axis')

        return cls(values, sample_data=sample_data, feature_data=feature_data)

    def drop_duplicate_indices(self, axis='index', keep='first'):
        """Drops duplicate indices along given axis."""

        if axis == 'index':
            mask = ~self._values.index.duplicated(keep=keep)
            values = self._values.loc[mask]

            sample_data = self._sample_data
            feature_data = self._feature_data.loc[mask]
        elif axis == 'columns':
            mask = ~self._values.columns.duplicated(keep=keep)
            values = self._values.loc[:, mask]

            sample_data = self._sample_data.loc[mask]
            feature_data = self._feature_data
        else:
            raise ValueError('Unknown value for axis')

        return self.__class__(
            values.copy(), sample_data=sample_data, feature_data=feature_data)


class GenomicMatrix(AnnotatedMatrix):
    """Class respresenting matrices indexed by genomic positions."""

    def __init__(self, values, sample_data=None, feature_data=None):
        if not isinstance(values, GenomicDataFrame):
            values = GenomicDataFrame(values)

        super().__init__(
            values, sample_data=sample_data, feature_data=feature_data)

    @classmethod
    def from_df(cls, values, chrom_lengths=None, **kwargs):
        """Constructs a genomic matrix from the given DataFrame."""

        if not isinstance(values, GenomicDataFrame):
            values = GenomicDataFrame.from_df(
                values, chrom_lengths=chrom_lengths)

        return cls(values, **kwargs)

    @classmethod
    def from_csv(cls,
                 file_path,
                 index_col,
                 sample_data=None,
                 feature_data=None,
                 sample_mapping=None,
                 feature_mapping=None,
                 drop_cols=None,
                 chrom_lengths=None,
                 read_data_kws=None,
                 **kwargs):
        """Reads values from a csv file."""

        if not 2 <= len(index_col) <= 3:
            raise ValueError('index_col should contain 2 entries'
                             ' (for positioned data or 3 entries'
                             ' (for ranged data)')

        default_dtype = {index_col[0]: str}
        dtype = toolz.merge(default_dtype, kwargs.pop('dtype', {}))

        values = pd.read_csv(file_path, dtype=dtype, **kwargs)
        values = values.set_index(index_col)

        # If sample/feature_data are not dataframes, assume they are
        # file paths or objects and try to read from them.
        read_data_kws_default = {
            'sep': kwargs.pop('sep', None),
            'index_col': 0
        }
        read_data_kws = toolz.merge(read_data_kws_default, read_data_kws or {})

        if not (sample_data is None or isinstance(sample_data, pd.DataFrame)):
            sample_data = pd.read_csv(sample_data, **read_data_kws)

        if not (feature_data is None
                or isinstance(feature_data, pd.DataFrame)):
            feature_data = pd.read_csv(feature_data, **read_data_kws)

        values = cls._preprocess_values(
            values,
            sample_data=sample_data,
            feature_data=feature_data,
            sample_mapping=sample_mapping,
            feature_mapping=feature_mapping,
            drop_cols=drop_cols)

        return cls.from_df(
            values,
            sample_data=sample_data,
            feature_data=feature_data,
            chrom_lengths=chrom_lengths)

    @classmethod
    def from_csv_condensed(cls,
                           file_path,
                           index_col=0,
                           sample_data=None,
                           feature_data=None,
                           sample_mapping=None,
                           feature_mapping=None,
                           drop_cols=None,
                           chrom_lengths=None,
                           index_regex=RANGED_REGEX,
                           is_one_based=False,
                           is_inclusive=False,
                           read_data_kws=None,
                           **kwargs):
        """Reads values from a csv file with a condensed index."""

        values = pd.read_csv(file_path, index_col=index_col, **kwargs)

        values.index = cls._expand_condensed_index(
            values.index,
            index_regex,
            is_one_based=is_one_based,
            is_inclusive=is_inclusive)

        # If sample/feature_data are not dataframes, assume they are
        # file paths or objects and try to read from them.
        read_data_kws_default = {
            'sep': kwargs.pop('sep', None),
            'index_col': 0
        }
        read_data_kws = toolz.merge(read_data_kws_default, read_data_kws or {})

        if not (sample_data is None or isinstance(sample_data, pd.DataFrame)):
            sample_data = pd.read_csv(sample_data, **read_data_kws)

        if not (feature_data is None
                or isinstance(feature_data, pd.DataFrame)):
            feature_data = pd.read_csv(feature_data, **read_data_kws)

        values = cls._preprocess_values(
            values,
            sample_data=sample_data,
            feature_data=feature_data,
            sample_mapping=sample_mapping,
            feature_mapping=feature_mapping,
            drop_cols=drop_cols)

        return cls.from_df(
            values,
            sample_data=sample_data,
            feature_data=feature_data,
            chrom_lengths=chrom_lengths)

    @classmethod
    def _expand_condensed_index(cls,
                                index,
                                regex_expr,
                                is_one_based=False,
                                is_inclusive=False):
        """Expands condensed index into a MultiIndex."""

        # Parse entries.
        regex = re.compile(regex_expr)
        group_dicts = (regex.match(el).groupdict() for el in index)

        # Extract chromosome, start, end positions.
        if regex.groups == 3:
            tups = ((grp['chromosome'], int(grp['start']), int(grp['end']))
                    for grp in group_dicts)
            chrom, starts, ends = zip(*tups)
        elif regex.groups == 2:
            tups = ((grp['chromosome'], int(grp['position']))
                    for grp in group_dicts)
            chrom, starts = zip(*tups)
            ends = None
        else:
            raise ValueError('Regex should have two or three groups '
                             '(for positioned/ranged data, respectively)')

        # Correct for one-base and inclusive-ness to match Python conventions.
        starts = np.array(starts)

        if is_one_based:
            starts -= 1

        if ends is not None and is_inclusive:
            ends = np.array(ends)
            ends += 1

        # Build index.
        if ends is None:
            index = pd.MultiIndex.from_arrays(
                [chrom, starts], names=['chromosome', 'position'])
        else:
            index = pd.MultiIndex.from_arrays(
                [chrom, starts, ends], names=['chromosome', 'start', 'end'])

        return index

    @property
    def gloc(self):
        """Genomic-position indexer.

        Used to select rows from the matrix by their genomic position.
        Interface is the same as for the GenomicDataFrame gloc property
        (which this method delegates to).
        """

        return GLocWrapper(self._values.gloc, self._gloc_constructor)

    def _gloc_constructor(self, values):
        """Constructor that attempts to build new instance
           from given values."""

        if isinstance(values, GenomicDataFrame):
            sample_data = self._sample_data.reindex(index=values.columns)
            feature_data = self._feature_data.reindex(index=values.index)

            return self.__class__(
                values.copy(),
                sample_data=sample_data,
                feature_data=feature_data)

        return values

    def expand(self):
        """Expands matrix to include values from missing bins.

        Assumes rows are regularly spaced with a fixed bin size.
        """

        expanded = self._expand(self._values)
        feature_data = self._feature_data.reindex(index=expanded.index)

        return self.__class__(
            expanded, sample_data=self._sample_data, feature_data=feature_data)

    @staticmethod
    def _expand(values):
        def _bin_indices(grp, bin_size):
            chrom = grp.index[0][0]

            start = grp.index.get_level_values(1).min()
            end = grp.index.get_level_values(2).max()

            bins = np.arange(start, end + 1, step=bin_size)

            return zip(itertools.cycle([chrom]), bins[:-1], bins[1:])

        bin_size = values.index[0][2] - values.index[0][1]

        # TODO: Warn if bin_size is 1? (Probably positioned data).

        # Check inferred bin size.
        starts = values.index.get_level_values(1)
        ends = values.index.get_level_values(2)
        diffs = ends - starts

        if not all(diffs == bin_size):
            raise ValueError('Bins do not match inferred bin size')

        # Check if following bins match inferred bin size.
        if not all(np.mod(np.diff(starts), bin_size) == 0):
            raise ValueError('Following bins do not match inferred bin size')

        indices = list(
            itertools.chain.from_iterable(
                _bin_indices(grp, bin_size=bin_size)
                for _, grp in values.groupby(level=0)))

        return values.reindex(index=indices)

    def impute(self, window=11, min_probes=5, expand=True):
        """Imputes nan values from neighboring bins."""

        if expand:
            values = self._expand(self._values)
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

        # Match feature data to new values.
        feature_data = self._feature_data.reindex(index=imputed.index)

        return self.__class__(
            imputed, sample_data=self._sample_data, feature_data=feature_data)

    def resample(self, bin_size, start=None, agg='mean'):
        """Resamples values at given interval by binning."""

        # Perform resampling per chromosome.
        resampled = pd.concat(
            (self._resample_chromosome(
                grp, bin_size=bin_size, agg=agg, start=start)
             for _, grp in self._values.groupby(level=0)),
            axis=0)  # yapf: disable

        # Restore original index order.
        resampled = resampled.reindex(self._values.gloc.chromosomes, level=0)

        return self.__class__(
            GenomicDataFrame(
                resampled, chrom_lengths=self._values.chromosome_lengths),
            sample_data=self._sample_data)

    @staticmethod
    def _resample_chromosome(values, bin_size, start=None, agg='mean'):
        # Bin rows by their centre positions.
        starts = values.index.get_level_values(1)
        ends = values.index.get_level_values(2)

        positions = (starts + ends) // 2

        range_start = starts.min() if start is None else start
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

    def rename_chromosomes(self, mapping):
        """Returns copy of matrix with renamed chromosomes."""

        return self.__class__(
            values=self._values.rename_chromosomes(mapping),
            sample_data=self.sample_data,
            feature_data=self.feature_data)

    def annotate(self, features, feature_id='gene_id'):
        """Annotates values for given features."""

        # Calculate calls.
        get_id = operator.attrgetter(feature_id)

        annotated_calls = {}
        for feature in features.itertuples():
            try:
                chrom, start, end = feature.Index
                overlap = self._values.gloc.search(chrom, start, end)

                annotated_calls[get_id(feature)] = overlap.median()
            except KeyError:
                pass

        # Assemble into dataframe.
        annotated = pd.DataFrame.from_records(annotated_calls).T
        annotated.index.name = feature_id

        return AnnotatedMatrix(annotated, sample_data=self._sample_data)

    def plot_sample(self, sample, ax=None, **kwargs):
        """Plots values for given sample along genomic axis."""
        ax = gplot.genomic_scatter_plot(
            self._values, y=sample, ax=ax, **kwargs)
        return ax

    def plot_heatmap(self,
                     cmap='RdBu_r',
                     sample_cols=None,
                     sample_colors=None,
                     metric='euclidean',
                     method='complete',
                     transpose=True,
                     cluster=True,
                     **kwargs):
        """Plots heatmap of gene expression over samples."""

        if 'row_cluster' in kwargs or 'col_cluster' in kwargs:
            raise ValueError(
                'GenomicMatrices only supports clustering by samples. '
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
            sample_cols=sample_cols,
            sample_colors=sample_colors,
            cmap=cmap,
            metric=metric,
            method=method,
            transpose=transpose,
            **heatmap_kws)

        self._style_heatmap(cm, transpose=transpose)

        return cm

    def _style_heatmap(self, cm, transpose):
        chrom_breaks = self._values.groupby(level=0).size().cumsum()

        chrom_labels = self._values.gloc.chromosomes
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
            return self._wrap_function(attr)

        return attr

    def __getitem__(self, item):
        result = self._gloc[item]

        if isinstance(result, GenomicSlice):
            result = GLocSliceWrapper(
                self._gloc, chromosome=item, constructor=self._constructor)
        else:
            result = self._constructor(result)

        return result

    def _wrap_function(self, func):
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
