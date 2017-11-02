import functools
import re

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import toolz

from genopandas.core.frame import GenomicDataFrame, GenomicSlice
from genopandas.plotting.base import scatter_plot
from genopandas.plotting.clustermap import color_annotation, draw_legends
from genopandas.util.pandas_ import DfWrapper

RANGED_REGEX = r'(?P<chromosome>\w+):(?P<start>\d+)-(?P<end>\d+)'
POSITIONED_REGEX = r'(?P<chromosome>\w+):(?P<position>\d+)'


class AnnotatedMatrix(DfWrapper):
    def __init__(self, values, sample_data=None, feature_data=None):

        # Create empty annotations if none given.
        if sample_data is None:
            sample_data = pd.DataFrame({}, index=values.columns)

        if feature_data is None:
            feature_data = pd.DataFrame({}, index=values.index)

        # Check {sample,feature}_data.
        assert (values.shape[1] == sample_data.shape[0]
                and all(values.columns == sample_data.index))

        assert (values.shape[0] == feature_data.shape[0]
                and all(values.index == feature_data.index))

        # Check if all matrix columns are numeric.
        for col_name, col_values in values.items():
            if not is_numeric_dtype(col_values):
                raise ValueError('Column {} is not numeric'.format(col_name))

        super().__init__(values)

        self._sample_data = sample_data
        self._feature_data = feature_data

    def _constructor(self, values):
        """Constructor that attempts to build new instance
           from given values."""

        if isinstance(values, pd.DataFrame):
            sample_data = self._sample_data.reindex(index=values.columns)
            feature_data = self._feature_data.reindex(index=values.index)

            return self.__class__(
                values.copy(),
                sample_data=sample_data,
                feature_data=feature_data)

        return values

    @property
    def feature_data(self):
        return self._feature_data

    @property
    def sample_data(self):
        return self._sample_data

    @classmethod
    def from_csv(cls,
                 file_path,
                 sample_data=None,
                 feature_data=None,
                 sample_mapping=None,
                 feature_mapping=None,
                 drop_cols=None,
                 **kwargs):

        default_kwargs = {'index_col': 0}
        kwargs = toolz.merge(default_kwargs, kwargs)

        values = pd.read_csv(str(file_path), **kwargs)

        values = cls._preprocess_values(
            values,
            sample_data=sample_data,
            feature_data=feature_data,
            sample_mapping=sample_mapping,
            feature_mapping=feature_mapping,
            drop_cols=drop_cols)

        return cls(values, sample_data=sample_data, feature_data=feature_data)

    @staticmethod
    def _preprocess_values(values,
                           sample_data=None,
                           feature_data=None,
                           sample_mapping=None,
                           feature_mapping=None,
                           drop_cols=None):

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

    def rename(self, index=None, columns=None):
        """Rename samples/features in the matrix."""

        renamed = self._values.rename(index=index, columns=columns)

        if index is not None:
            feature_data = self._feature_data.rename(index=index)
        else:
            feature_data = self._feature_data.copy()

        if columns is not None:
            sample_data = self._sample_data.rename(index=columns)
        else:
            sample_data = self._sample_data.copy()

        return self.__class__(
            renamed, feature_data=feature_data, sample_data=sample_data)

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
            values,
            sample_data=sample_data,
            feature_data=self._feature_data.copy())

    def dropna_samples(self, subset, how='any', thresh=None):
        """Drops samples with NAs in sample_data."""

        sample_data = self._sample_data.dropna(
            subset=subset, how=how, thresh=thresh)
        values = self._values.reindex(columns=sample_data.index)

        return self.__class__(
            values,
            sample_data=sample_data,
            feature_data=self._feature_data.copy())

    def __eq__(self, other):
        if not isinstance(other, AnnotatedMatrix):
            return False
        return all(self.values == other.values) and \
            all(self.sample_data == other.sample_data) and \
            all(self.feature_data == other.feature_data)

    def plot_heatmap(self,
                     cmap='RdBu_r',
                     sample_cols=None,
                     sample_colors=None,
                     feature_cols=None,
                     feature_colors=None,
                     metric='euclidean',
                     method='complete',
                     transpose=False,
                     legend_kws=None,
                     **kwargs):
        """Plots clustered heatmap of matrix values."""

        import matplotlib.pyplot as plt
        import seaborn as sns

        if sample_cols is not None:
            sample_annot, sample_cmap = color_annotation(
                self._sample_data[sample_cols], colors=sample_colors)
        else:
            sample_annot, sample_cmap = None, None

        if feature_cols is not None:
            feature_annot, feature_cmap = color_annotation(
                self._feature_data[feature_cols], colors=feature_colors)
        else:
            feature_annot, feature_cmap = None, None

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

        #if annot_cmap is not None:
        #    draw_legends(cm, annot_cmap, **(legend_kws or {}))

        return cm

    def plot_pca(self, components=(1, 2), ax=None, by_features=False,
                 **kwargs):
        """Plots PCA of samples."""

        try:
            from sklearn.decomposition import PCA
        except ImportError:
            raise ImportError('Scikit-learn must be installed to '
                              'perform PCA analyses')

        # Fit PCA and transform expression.
        n_components = max(components)

        pca = PCA(n_components=max(components))

        if by_features:
            # Do PCA on features.
            transform = pca.fit_transform(self._values.values)

            transform = pd.DataFrame(
                transform,
                columns=['pca_{}'.format(i + 1) for i in range(n_components)],
                index=self.values.index)

            # Assemble plot data.
            plot_data = pd.concat([transform, self._feature_data], axis=1)
        else:
            # Do PCA on samples.
            transform = pca.fit_transform(self._values.values.T)

            transform = pd.DataFrame(
                transform,
                columns=['pca_{}'.format(i + 1) for i in range(n_components)],
                index=self.values.columns)

            # Assemble plot data.
            plot_data = pd.concat([transform, self._sample_data], axis=1)

        # Draw using lmplot.
        pca_x, pca_y = ['pca_{}'.format(c) for c in components]
        ax = scatter_plot(data=plot_data, x=pca_x, y=pca_y, ax=ax, **kwargs)

        ax.set_xlabel('Component ' + str(components[0]))
        ax.set_ylabel('Component ' + str(components[1]))

        return ax

    def plot_pca_variance(self, n_components=None, ax=None, by_features=False):
        """Plots variance explained by PCA components."""

        import matplotlib.pyplot as plt
        import seaborn as sns

        try:
            from sklearn.decomposition import PCA
        except ImportError:
            raise ImportError('Scikit-learn must be installed to '
                              'perform PCA analyses')

        pca = PCA(n_components=n_components)
        pca.fit(self._values.values.T if by_features else self._values.values)

        if ax is None:
            _, ax = plt.subplots()

        x = np.arange(pca.n_components_) + 1
        y = pca.explained_variance_ratio_
        ax.plot(x[:len(y)], y)

        ax.set_xlabel('Component')
        ax.set_ylabel('Explained variance')
        sns.despine(ax=ax)

        return ax


class GenomicMatrix(AnnotatedMatrix):
    """Base class for matrices indexed by genomic range/position.

    Should not be used directly. Use either ``PositionedGenomicMatrix`` for
    positioned data or ``RangedGenomicMatrix`` for ranged data.
    """

    def __init__(self, values, sample_data=None, feature_data=None):
        if not isinstance(values, GenomicDataFrame):
            raise ValueError('Values should be a GenomicDataFrame instance')

        super().__init__(
            values, sample_data=sample_data, feature_data=feature_data)

    @classmethod
    def from_df(cls, values, chrom_lengths=None, **kwargs):
        if not isinstance(values, GenomicDataFrame):
            values = GenomicDataFrame.from_df(
                values, chrom_lengths=chrom_lengths)

        if values.index.nlevels == 3:
            return RangedGenomicMatrix(values, **kwargs)
        else:
            raise NotImplementedError()

    @classmethod
    def from_csv(cls,
                 file_path,
                 index_col,
                 sample_data=None,
                 feature_data=None,
                 sample_mapping=None,
                 feature_mapping=None,
                 chrom_lengths=None,
                 drop_cols=None,
                 **kwargs):

        if not 2 <= len(index_col) <= 3:
            raise ValueError('index_col should contain 2 entries'
                             ' (for positioned data or 3 entries'
                             ' (for ranged data)')

        values = pd.read_csv(file_path, index_col=index_col, **kwargs)

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
                           index_regex,
                           index_col=0,
                           sample_data=None,
                           feature_data=None,
                           sample_mapping=None,
                           feature_mapping=None,
                           chrom_lengths=None,
                           drop_cols=None,
                           **kwargs):

        # Read csv and expand index.
        values = pd.read_csv(file_path, index_col=index_col, **kwargs)
        values.index = cls._expand_condensed_index(values.index, index_regex)

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
                                one_based=False,
                                inclusive=False):
        raise NotImplementedError()

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


class RangedGenomicMatrix(GenomicMatrix):
    @classmethod
    def _expand_condensed_index(cls,
                                index,
                                regex_expr,
                                one_based=False,
                                inclusive=False):

        # TODO: Check regex.

        regex = re.compile(regex_expr)

        # Extract chromosome, start, end positions.
        group_dicts = (regex.match(el).groupdict() for el in index)

        tups = ((grp['chromosome'], int(grp['start']), int(grp['end']))
                for grp in group_dicts)

        chrom, starts, ends = zip(*tups)

        # Correct for one-base and/or inclusive-ness to
        # match Python conventions.
        starts = np.array(starts)
        ends = np.array(ends)

        if one_based:
            starts -= 1

        if inclusive:
            ends += 1

        # Build index.
        index = pd.MultiIndex.from_arrays(
            [chrom, starts, ends], names=['chromosome', 'start', 'end'])

        return index
