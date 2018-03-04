"""Dataframe-related functions/classes."""

from collections import OrderedDict
import math
import re

from natsort import natsorted
import numpy as np
import pandas as pd

from .tree import GenomicIntervalTree


class GenomicDataFrame(pd.DataFrame):
    """DataFrame with fast indexing by genomic position.

    Requires columns 'chromosome', 'start' and 'end' to be present in the
    DataFrame, as these columns are used for indexing.

    Examples
    --------

    Constructing from scratch:

    >>> df = pd.DataFrame.from_records(
    ...    [('1', 10, 20, 'a'), ('2', 10, 20, 'b'), ('2', 30, 40, 'c')],
    ...    columns=['chromosome', 'start', 'end'])
    >>> df.set_index(['chromosome', 'start', 'end'])
    >>> GenomicDataFrame(df)

    Reading from a GTF file:

    >>> GenomicDataFrame.from_gtf('/path/to/reference.gtf')

    Querying by a genomic range:

    >>> genomic_df.gloc['2'][30:50]

    """

    _internal_names = pd.DataFrame._internal_names + ['_gloc']
    _internal_names_set = set(_internal_names)

    _metadata = ['_chrom_lengths']

    def __init__(self, *args, chrom_lengths=None, **kwargs):
        super().__init__(*args, **kwargs)

        self._gloc = None
        self._chrom_lengths = chrom_lengths

    @property
    def _constructor(self):
        return GenomicDataFrame

    @classmethod
    def from_df(cls, df, **kwargs):
        """Constructs instance from dataframe containing
           ranged/positioned data."""

        if df.index.nlevels == 3:
            return cls(df, **kwargs)
        elif df.index.nlevels == 2:
            return cls.from_position_df(df, **kwargs)
        else:
            raise ValueError('DataFrame should have either two index levels '
                             '(for positioned data) or three index levels '
                             '(for ranged data)')

    @classmethod
    def from_position_df(cls, df, width=1, **kwargs):
        """Constructs instance from positioned dataframe."""

        positions = df.index.get_level_values(1)

        starts = positions - (width // 2)
        ends = positions + math.ceil(width / 2)

        new_df = df.copy()
        new_df.index = pd.MultiIndex.from_arrays(
            [df.index.get_level_values(0), starts, ends],
            names=['chromosome', 'start', 'end'])

        return cls(new_df, **kwargs)

    @classmethod
    def from_records(cls,
                     records,
                     index_col,
                     columns=None,
                     drop_index_col=True,
                     chrom_lengths=None,
                     **kwargs):
        """Creates a GenomicDataFrame from a structured or record ndarray."""

        if not 2 <= len(index_col) <= 3:
            raise ValueError('index_col should contain 2 entries'
                             ' (for positioned data or 3 entries'
                             ' (for ranged data)')

        df = super().from_records(records, columns=columns, **kwargs)

        # Convert chromosome to str.
        df[index_col[0]] = df[index_col[0]].astype(str)

        df = df.set_index(index_col, drop=drop_index_col)

        return cls.from_df(df, chrom_lengths=chrom_lengths)

    @classmethod
    def from_csv(cls,
                 file_path,
                 index_col,
                 drop_index_col=True,
                 chrom_lengths=None,
                 **kwargs):
        """Creates a GenomicDataFrame from a csv file using ``pandas.read_csv``.

        Parameters
        ----------
        file_path : str
            Path to file.
        index_col : List[str]
            Columns to use for index. Columns should be indicated by their name.
            Should contain two entries for positioned data, three
            entries for ranged data. If not given, the first three columns of
            the DataFrame are used by default.
        drop_index_col : bool
            Whether to drop the index columns in the DataFrame (True, default)
            or to drop them from the dataframe (False).
        chrom_lengths : Dict[str, int]
            Chromosome lengths.
        **kwargs
            Any extra kwargs are passed to ``pandas.read_csv``.

        Returns
        -------
        GenomicDataFrame
            DataFrame containing the file contents.

        """

        if not 2 <= len(index_col) <= 3:
            raise ValueError('index_col should contain 2 entries'
                             ' (for positioned data or 3 entries'
                             ' (for ranged data)')

        df = pd.read_csv(
            file_path, index_col=None, dtype={index_col[0]: str}, **kwargs)
        df = df.set_index(index_col, drop=drop_index_col)

        return cls.from_df(df, chrom_lengths=chrom_lengths)

    @classmethod
    def from_gtf(cls, gtf_path, filter_=None):
        """Creates a GenomicDataFrame from a GTF file."""

        if (gtf_path.suffixes[-1] == '.gz' and
            gtf_path.with_suffix('.gz.tbi').exists()):
            gdf = cls._from_gtf_pysam(gtf_path, filter_=filter_)
        else:
            gdf = cls._from_gtf_pandas(gtf_path, filter_=filter_)

        return gdf

    @classmethod
    def _from_gtf_pandas(cls, gtf_path, filter_=None):
        """Reads GTF directly using pandas."""

        # Read gtf data.
        data = pd.read_csv(
            gtf_path, sep='\t', header=None, index_col=None,
            names=('contig', 'source', 'feature', 'start', 'end', 'score',
                   'strand', 'frame', 'attributes'),
            dtype={'contig': str})

        # Expand attributes.
        regex = re.compile(r'(?P<key>\w+) "(?P<value>\w+)"')

        rows = (dict(regex.findall(el)) for el in data['attributes'])
        attr_data = pd.DataFrame.from_records(rows)

        data = pd.concat([data.drop('attributes', axis=1), attr_data], axis=1)
        data = data.set_index(['contig', 'start', 'end'], drop=False)

        # Filter rows if filter_ is given.
        if filter_ is not None:
            data = data.query(filter_)

        return cls(data)

    @classmethod
    def _from_gtf_pysam(cls, gtf_path, filter_=None):
        """Reads GTF more efficiently using pysam."""

        try:
            import pysam
        except ImportError:
            raise ImportError('Pysam needs to be installed for '
                              'reading GTF files')

        # Parse records into rows.
        gtf_file = pysam.TabixFile(str(gtf_path), parser=pysam.asGTF())
        records = (rec for rec in gtf_file.fetch())

        # Filter records if needed.
        if filter_ is not None:
            records = (rec for rec in records if filter_(rec))

        # Build dataframe.
        def _record_to_row(record):
            row = {
                'contig': record.contig,
                'source': record.source,
                'feature': record.feature,
                'start': record.start,
                'end': record.end,
                'score': record.score,
                'strand': record.strand,
                'frame': record.frame
            }
            row.update(dict(record))
            return row

        gdf = cls.from_records(
            (_record_to_row(rec) for rec in records),
            index_col=['contig', 'start', 'end'],
            drop_index_col=False)

        # Reorder columns to correspond with GTF format.
        columns = ('contig', 'source', 'feature', 'start', 'end', 'score',
                   'strand', 'frame')
        gdf = _reorder_columns(gdf, order=columns)

        return gdf

    @property
    def gloc(self):
        """Genomic indexer for querying the dataframe."""

        if self._gloc is None:
            self._gloc = GenomicIndexer(self)

        return self._gloc

    @property
    def chromosome_lengths(self):
        """Chromosome lengths."""

        if self._chrom_lengths is None:
            chrom_lengths = self._calculate_chrom_lengths()
            self._chrom_lengths = self._order_chrom_lengths(chrom_lengths)

        return self._chrom_lengths

    @chromosome_lengths.setter
    def chromosome_lengths(self, value):
        if not isinstance(value, OrderedDict):
            value = self._order_chrom_lengths(value)
        self._chrom_lengths = value

    def _calculate_chrom_lengths(self):
        chromosomes = self.index.get_level_values(0)
        ends = self.index.get_level_values(2) - 1

        lengths = pd.Series(ends).groupby(chromosomes).max()
        return dict(zip(lengths.index, lengths.values))

    @staticmethod
    def _order_chrom_lengths(chrom_lengths):
        if not isinstance(chrom_lengths, OrderedDict):
            order = natsorted(chrom_lengths.keys())
            values = (chrom_lengths[k] for k in order)
            chrom_lengths = OrderedDict(zip(order, values))
        return chrom_lengths

    @property
    def chromosome_offsets(self):
        """Chromosome offsets (used when plotting chromosomes linearly)."""

        # Record offsets in ordered dict.
        sorted_lengths = list(self.chromosome_lengths.values())

        cumsums = np.concatenate([[0], np.cumsum(sorted_lengths)])
        offsets = OrderedDict(zip(self.chromosome_lengths.keys(),
                                  cumsums[:-1]))  # yapf: disable

        # Add special marker for end.
        offsets['_END_'] = cumsums[-1]

        return offsets

    def reset_index(self, level=None, drop=False, col_level=0, col_fill=''):
        """Mirrors pd.DataFrame.reset_index, but returns a standard DataFrame
        instead of a GenomicDataFrame, as the (genomic) index is being dropped.
        """

        reset_values = super().reset_index(
            level=level, drop=drop, col_level=col_level, col_fill=col_fill)

        return pd.DataFrame(reset_values)

    def rename_chromosomes(self, mapping):
        """Returns copy with renamed chromosomes."""

        if callable(mapping):
            map_func = mapping
        else:
            map_func = lambda s: mapping.get(s, s)

        # Map chromosomes.
        chrom_col, start_col, end_col = self.index.names

        new_values = self.reset_index()
        new_values[chrom_col] = new_values[chrom_col].map(map_func)

        new_values.set_index([chrom_col, start_col, end_col], inplace=True)

        return self.__class__(new_values)


class GenomicIndexer(object):
    """Base GenomicIndexer class used to index GenomicDataFrames."""

    def __init__(self, gdf):
        self._gdf = gdf
        self._trees = None

    def __getitem__(self, item):
        """Accessor used to query the dataframe by position.

        If a list of chromosomes is given, the dataframe is subset to the
        given chromosomes. Note that chromosomes are also re-ordered to
        adhere to the given order. If a single chromosome is given, a
        GenomicSlice is returned. This slice object can be sliced to query
        a specific genomic range.
        """

        if isinstance(item, list):
            # Add extra index level to enforce uniqueness.
            augmented = self._gdf.set_index(
                self._gdf.groupby(level=0).cumcount(), append=True)

            # Reindex and drop added level.
            subset = augmented.reindex(index=item, level=0)
            subset.index = subset.index.droplevel(subset.index.nlevels - 1)

            # Remove any duplicates.
            items = list(OrderedDict.fromkeys(item))

            # Subset lengths.
            prev_lengths = subset.chromosome_lengths
            subset.chromosome_lengths = OrderedDict(
                (k, prev_lengths[k]) for k in items)  # yapf: disable

            return subset

        return GenomicSlice(self, chromosome=item)

    @property
    def gdf(self):
        """The indexed DataFrame."""
        return self._gdf

    @property
    def chromosome(self):
        """Chromosome values."""
        return self._gdf.index.get_level_values(0)

    @property
    def chromosomes(self):
        """Available chromosomes."""
        return list(self.chromosome_lengths.keys())

    @property
    def chromosome_lengths(self):
        """Chromosome lengths."""
        return self._gdf.chromosome_lengths

    @property
    def chromosome_offsets(self):
        """Chromosome offsets."""
        return self._gdf.chromosome_offsets

    @property
    def start(self):
        """Start positions."""
        return self._gdf.index.get_level_values(1)

    @property
    def start_offset(self):
        """Start positions, offset by chromosome lengths."""
        return self._offset_positions(self.start)

    @property
    def end(self):
        """End positions."""
        return self._gdf.index.get_level_values(2)

    @property
    def end_offset(self):
        """End positions, offset by chromosome lengths."""
        return self._offset_positions(self.end)

    @property
    def position(self):
        """Mid positions (between start/end).

        Should corrrespond with original positions for
        expanded positioned data.
        """
        return (self.start + self.end) // 2

    @property
    def position_offset(self):
        """Mid positions (see position), offset by chromosome lengths."""
        return self._offset_positions(self.position)

    def _offset_positions(self, positions):
        offsets = pd.Series(self.chromosome_offsets)
        return positions + offsets.loc[self.chromosome].values

    @property
    def trees(self):
        """Trees used for indexing the DataFrame."""

        if self._trees is None:
            self._trees = self._build_trees()

        return self._trees

    def rebuild(self):
        """Rebuilds the genomic interval trees."""
        self._trees = self._build_trees()

    def _build_trees(self):
        tuples = zip(self.chromosome, self.start, self.end,
                     range(self._gdf.shape[0]))
        return GenomicIntervalTree.from_tuples(tuples)

    def search(self,
               chromosome,
               start,
               end,
               strict_left=False,
               strict_right=False):
        """Searches the DataFrame for rows within given range."""

        overlap = self.trees.search(
            chromosome,
            start,
            end,
            strict_left=strict_left,
            strict_right=strict_right)

        indices = [interval[2] for interval in overlap]

        return self._gdf.iloc[indices].sort_index()


class GenomicSlice(object):
    """Supporting class used by the GenomicIndexer for slicing chromosomes."""

    def __init__(self, indexer, chromosome):
        self._indexer = indexer
        self._chromosome = chromosome

    def __getitem__(self, item):
        if isinstance(item, slice):
            subset = self._indexer.search(
                self._chromosome, start=item.start, end=item.stop)

            # Subset lengths.
            subset.chromosome_lengths = OrderedDict(
                [(self._chromosome,
                  subset.chromosome_lengths[self._chromosome])])

            return subset

        return self._indexer.search(self._chromosome, start=item)


def _reorder_columns(df, order):
    """Reorders dataframe columns, sorting any extra columns alphabetically."""
    extra_cols = set(df.columns) - set(order)
    return df[list(order) + sorted(extra_cols)]
