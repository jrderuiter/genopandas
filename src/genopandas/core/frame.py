"""Dataframe-related functions/classes."""

from collections import OrderedDict

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

    _internal_names = pd.DataFrame._internal_names + ['_gi']
    _internal_names_set = set(_internal_names)

    _metadata = ['_gi_metadata']

    def __init__(self,
                 *args,
                 use_index=False,
                 chromosome_col='chromosome',
                 start_col='start',
                 end_col='end',
                 chrom_lengths=None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self._gi = None
        self._gi_metadata = {
            'use_index': use_index,
            'chromosome_col': chromosome_col,
            'start_col': start_col,
            'end_col': end_col,
            'lengths': chrom_lengths
        }

    @property
    def gi(self):
        """Genomic indexer for querying the dataframe."""
        if self._gi is None:
            self._gi = GenomicIndexer(self, **self._gi_metadata)
        return self._gi

    @property
    def _constructor(self):
        return GenomicDataFrame

    @classmethod
    def from_csv(cls,
                 file_path,
                 use_index=False,
                 chromosome_col='chromosome',
                 start_col='start',
                 end_col='end',
                 chrom_lengths=None,
                 **kwargs):
        data = pd.DataFrame.from_csv(file_path, **kwargs)
        return cls(
            data,
            use_index=use_index,
            chromosome_col=chromosome_col,
            start_col=start_col,
            end_col=end_col,
            chrom_lengths=chrom_lengths)

    @classmethod
    def from_gtf(cls, gtf_path, filter_=None):
        """Build a GenomicDataFrame from a GTF file."""

        try:
            import pysam
        except ImportError:
            raise ImportError('Pysam needs to be installed for '
                              'reading GTF files')

        def _record_to_row(record):
            row = {
                'contig': record.contig,
                'source': record.source,
                'feature': record.feature,
                'start': int(record.start),
                'end': int(record.end),
                'score': record.score,
                'strand': record.strand,
                'frame': record.frame
            }
            row.update(dict(record))
            return row

        # Parse records into rows.
        gtf_file = pysam.TabixFile(str(gtf_path), parser=pysam.asGTF())
        records = (rec for rec in gtf_file.fetch())

        # Filter records if needed.
        if filter_ is not None:
            records = (rec for rec in records if filter_(rec))

        # Build dataframe.
        rows = (_record_to_row(rec) for rec in records)
        data = cls(pd.DataFrame.from_records(rows), chromosome_col='contig')

        # Reorder columns to correspond with GTF format.
        columns = ('contig', 'source', 'feature', 'start', 'end', 'score',
                   'strand', 'frame')
        data = _reorder_columns(data, order=columns)

        return data

    @classmethod
    def from_position_df(cls, df, position_col='position', width=0, **kwargs):
        """Builds a GenomicDataFrame from a dataframe with positions."""

        # Note: end is exclusive.
        start_col = kwargs.get('start_col', 'start')
        end_col = kwargs.get('end_col', 'end')

        half_width = width // 2
        df = df.assign(**{
            start_col: df[position_col] - half_width,
            end_col: (df[position_col] - half_width) + 1
        })

        df = df.drop(position_col, axis=1)

        return cls(df, **kwargs)


class GenomicIndexer(object):
    """Indexer class used to index GenomicDataFrames."""

    def __init__(self,
                 df,
                 use_index=False,
                 chromosome_col='chromosome',
                 start_col='start',
                 end_col='end',
                 lengths=None):

        if use_index:
            if not 2 <= df.index.nlevels <= 3:
                raise ValueError('Dataframe index does not have the required '
                                 'number of levels')
        else:
            if end_col is None:
                req_columns = [chromosome_col, start_col]
            else:
                req_columns = [chromosome_col, start_col, end_col]

            for col in req_columns:
                if col not in df.columns:
                    raise ValueError(
                        'Column {!r} not in dataframe'.format(col))

        if use_index:
            chromosomes = df.index.get_level_values(0).values
            starts = df.index.get_level_values(1).values

            if df.index.nlevels == 3:
                ends = df.index.get_level_values(2).values
            else:
                ends = starts + 1
        else:
            chromosomes = df[chromosome_col].values
            starts = df[start_col].values
            ends = df[end_col].values if end_col else starts + 1

        self._df = df
        self._lengths = lengths

        self._chromosome = chromosomes
        self._start = starts
        self._end = ends

        self._trees = None

    def __getitem__(self, item):
        mask = self._chromosome == item
        return self._df.loc[mask]

    @property
    def df(self):
        """The indexed dataframe."""
        return self._df

    @property
    def chromosome(self):
        """Chromosome values."""
        return self._chromosome

    @property
    def chromosomes(self):
        """Available chromosomes."""
        return list(np.unique(self._chromosome))

    @property
    def chromosome_lengths(self):
        """Chromosome lengths."""
        if self._lengths is None:
            lengths = pd.Series(self.end).groupby(self.chromosome).max()
            self._lengths = dict(zip(lengths.index, lengths.values))
        return {
            k: v
            for k, v in self._lengths.items() if k in set(self.chromosomes)
        }

    @property
    def chromosome_offsets(self):
        """Chromosome offsets (used when plotting chromosomes linearly)."""

        # Sort lengths by chromosome.
        chromosomes = self.chromosomes
        lengths = self.chromosome_lengths

        # Record offsets in ordered dict.
        sorted_lengths = [lengths[chrom] for chrom in chromosomes]

        cumsums = np.concatenate([[0], np.cumsum(sorted_lengths)])
        offsets = OrderedDict(zip(chromosomes, cumsums[:-1]))

        # Add special marker for end.
        offsets['_END_'] = cumsums[-1]

        return offsets

    @property
    def start(self):
        """Start positions."""
        return self._start

    @property
    def start_offset(self):
        """Start positions, offset by chromosome lengths."""
        return self._offset_positions(self.start)

    def _offset_positions(self, positions):
        offsets = pd.Series(self.chromosome_offsets)
        return positions + offsets.loc[self.chromosome].values

    @property
    def end(self):
        """End positions."""
        return self._end

    @property
    def end_offset(self):
        """End positions, offset by chromosome lengths."""
        return self._offset_positions(self.end)

    @property
    def trees(self):
        """Trees used for indexing the DataFrame."""

        if self._trees is None:
            tuples = zip(self.chromosome, self.start, self.end,
                         range(self._df.shape[0]))
            self._trees = GenomicIntervalTree.from_tuples(tuples)

        return self._trees

    def search(self,
               chromosome,
               begin,
               end,
               strict_left=False,
               strict_right=False):
        """Subsets the DataFrame for rows within given range."""

        overlap = self.trees.search(
            chromosome,
            begin,
            end,
            strict_left=strict_left,
            strict_right=strict_right)

        indices = [interval[2] for interval in overlap]

        return self._df.iloc[indices].sort_index()


def _reorder_columns(df, order):
    """Reorders dataframe columns, sorting any extra columns alphabetically."""

    extra_cols = set(df.columns) - set(order)
    return df[list(order) + sorted(extra_cols)]
