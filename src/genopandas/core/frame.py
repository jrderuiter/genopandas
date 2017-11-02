"""Dataframe-related functions/classes."""

from collections import OrderedDict
import math

from natsort import natsorted
import numpy as np
import pandas as pd

from genopandas.util.misc import reorder_columns
from .tree import GenomicIntervalTree


class GenomicDataFrame(pd.DataFrame):
    """DataFrame with fast indexing by genomic position.

    Requires columns 'chromosome', 'start' and 'end' to be present in the
    DataFrame, as these columns are used for indexing.

    Examples
    --------

    Constructing from scratch:

    >>> df = pd.DataFrame.from_records(
    ...    [('1', 10, 20), ('2', 10, 20), ('2', 30, 40)],
    ...    columns=['chromosome', 'start', 'end'])
    >>> GenomicDataFrame(df)

    Constructing with non-default columns:

    >>> df = pd.DataFrame.from_records(
    ...    [('1', 10, 20), ('2', 10, 20), ('2', 30, 40)],
    ...    columns=['chrom', 'chromStart', 'chromEnd'])
    >>> GenomicDataFrame(
    ...    df,
    ...    chromosome_col='chrom',
    ...    start_col='start',
    ...    end_col='end')

    Reading from a GTF file:

    >>> GenomicDataFrame.from_gtf('/path/to/reference.gtf.gz')

    Querying by genomic position:

    >>> genomic_df.gi.search('2', 30, 50)

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
        raise NotImplementedError()

    @property
    def _genomic_indexer(self):
        raise NotImplementedError()

    @classmethod
    def from_df(cls, df, **kwargs):
        """Constructs appropriate GenomicDataFrame subclass for given frame."""

        if df.index.nlevels == 3:
            return RangedGenomicDataFrame(df, **kwargs)
        elif df.index.nlevels == 2:
            return PositionedGenomicDataFrame(df, **kwargs)
        else:
            raise ValueError('DataFrame should have either two index levels '
                             '(for positioned data) or three index levels '
                             '(for ranged data)')

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

    @property
    def gloc(self):
        """Genomic indexer for querying the dataframe."""

        if self._gloc is None:
            self._gloc = self._genomic_indexer(self)

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
        raise NotImplementedError()

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
            subset = self._gdf.reindex(index=[item], level=0)

            # Subset lengths.
            prev_lengths = subset.chromosome_lengths
            subset.chromosome_lengths = OrderedDict(
                (k, prev_lengths[k]) for k in item)  # yapf: disable

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
        raise NotImplementedError()

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


class RangedGenomicDataFrame(GenomicDataFrame):
    @property
    def _constructor(self):
        return RangedGenomicDataFrame

    @property
    def _genomic_indexer(self):
        return RangedGenomicIndexer

    def _calculate_chrom_lengths(self):
        chromosomes = self.index.get_level_values(0)
        ends = self.index.get_level_values(2)

        lengths = pd.Series(ends).groupby(chromosomes).max()
        return dict(zip(lengths.index, lengths.values))

    @classmethod
    def from_gtf(cls, gtf_path, filter_=None):
        """Creates a GenomicDataFrame from a GTF file."""

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
        gdf = reorder_columns(gdf, order=columns)

        return gdf

    @classmethod
    def from_positioned_gdf(cls, positioned_gdf, width=1):
        """Builds ranged GDF from given positioned GDF."""

        chromosomes = positioned_gdf.index.get_level_values(0)
        positions = positioned_gdf.index.get_level_values(1)

        starts = positions - (width // 2)
        ends = positions + math.ceil(width / 2)

        names = [positioned_gdf.index.names[0], 'start', 'end']
        new_index = pd.MultiIndex.from_arrays(
            [chromosomes, starts, ends], names=names)

        new_df = positioned_gdf.copy()
        new_df.index = new_index

        # TODO: Adjust chromosome lengths for given width.
        chrom_lengths = positioned_gdf.chromosome_lengths

        return cls.from_df(new_df, chrom_lengths=chrom_lengths)


class RangedGenomicIndexer(GenomicIndexer):
    """GenomicIndexer class for querying ranged (start/end) data."""

    def __init__(self, gdf):

        if not gdf.index.nlevels == 3:
            raise ValueError('Dataframe must have three levels')

        super().__init__(gdf)

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

    def _build_trees(self):
        tuples = zip(self.chromosome, self.start, self.end,
                     range(self._gdf.shape[0]))
        return GenomicIntervalTree.from_tuples(tuples)


class PositionedGenomicDataFrame(GenomicDataFrame):
    @property
    def _constructor(self):
        return PositionedGenomicDataFrame

    @property
    def _genomic_indexer(self):
        return PositionedGenomicIndexer

    def _calculate_chrom_lengths(self):
        chromosomes = self.index.get_level_values(0)
        positions = self.index.get_level_values(1)

        lengths = pd.Series(positions).groupby(chromosomes).max()
        return dict(zip(lengths.index, lengths.values))

    @classmethod
    def from_ranged_gdf(cls, ranged_gdf):
        """Builds positioned GDF from given ranged GDF."""

        chromosomes = ranged_gdf.gloc.chromosome.values
        starts = ranged_gdf.gloc.start.values
        ends = ranged_gdf.gloc.end.values

        positions = (starts + ends) // 2

        names = [ranged_gdf.index.names[0], 'position']
        new_index = pd.MultiIndex.from_arrays(
            [chromosomes, positions], names=names)

        df = ranged_gdf.copy()
        df.index = new_index

        chrom_lengths = ranged_gdf.chromosome_lengths

        return cls.from_df(df, chrom_lengths=chrom_lengths)


class PositionedGenomicIndexer(GenomicIndexer):
    """GenomicIndexer class for querying positioned (single position) data."""

    def __init__(self, gdf):

        if not gdf.index.nlevels == 2:
            raise ValueError('Dataframe must have two levels')

        super().__init__(gdf)

    @staticmethod
    def _calculate_lengths(df):
        chromosomes = df.index.get_level_values(0)
        positions = df.index.get_level_values(1)

        lengths = pd.Series(positions).groupby(chromosomes).max()
        return dict(zip(lengths.index, lengths.values))

    @property
    def position(self):
        """Positions."""
        return self._gdf.index.get_level_values(1)

    @property
    def position_offset(self):
        """Positions, offset by chromosome lengths."""
        return self._offset_positions(self.position)

    def _build_trees(self):
        positions = self.position
        tuples = zip(self.chromosome, positions, positions + 1,
                     range(self._gdf.shape[0]))
        return GenomicIntervalTree.from_tuples(tuples)
