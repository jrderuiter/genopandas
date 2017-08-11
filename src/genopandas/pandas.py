"""Dataframe-related functions/classes."""

import itertools
import operator

from natsort import natsorted
import numpy as np
import pandas as pd
import pysam

from .tree import GenomicIntervalTree


class GenomicDataFrame(pd.DataFrame):
    """DataFrame with fast indexing by genomic position.

    Requires columns 'chromosome', 'start' and 'end' to be present in the
    DataFrame, as these columns are used for indexing.
    """

    _internal_names = pd.DataFrame._internal_names + ['_gi']
    _internal_names_set = set(_internal_names)

    _metadata = ['_gi_metadata']

    def __init__(self,
                 *args,
                 chrom_col='chromosome',
                 start_col='start',
                 end_col='end',
                 chrom_lengths=None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self._gi = None
        self._gi_metadata = {
            'chrom_col': chrom_col,
            'start_col': start_col,
            'end_col': end_col,
            'lengths': chrom_lengths
        }

    @property
    def gi(self):
        """Genomic index for querying the dataframe."""
        if self._gi is None:
            self._gi = GenomicIndex(self, **self._gi_metadata)
        return self._gi

    @property
    def _constructor(self):
        return GenomicDataFrame

    @classmethod
    def from_gtf(cls, gtf_path, filter_=None):
        """Build a GenomicDataFrame from a GTF file."""

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
        data = cls(pd.DataFrame.from_records(rows), chrom_col='contig')

        # Reorder columns to correspond with GTF format.
        columns = ('contig', 'source', 'feature', 'start', 'end', 'score',
                   'strand', 'frame')
        data = reorder_columns(data, order=columns)

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


class GenomicIndex(object):
    """Index class used to index GenomicDataFrames."""

    def __init__(self,
                 df,
                 chrom_col='chromosome',
                 start_col='start',
                 end_col='end',
                 lengths=None):
        self._df = df

        self._chrom_col = chrom_col
        self._start_col = start_col
        self._end_col = end_col
        self._lengths = lengths

        self._trees = None

    @property
    def df(self):
        """The indexed dataframe."""
        return self._df

    @property
    def chromosome(self):
        """Chromosome values."""
        return self._df[self._chrom_col]

    @property
    def chromosomes(self):
        """Available chromosomes."""
        return natsorted(self.chromosome.unique())

    @property
    def chromosome_lengths(self):
        """Lengths of available chromosomes."""
        if self._lengths is None:
            lengths = self.end.groupby(self.chromosome).max()
            self._lengths = dict(zip(lengths.index, lengths.values))
        return {
            k: v
            for k, v in self._lengths.items() if k in set(self.chromosomes)
        }

    @property
    def start(self):
        """Start values."""
        return self._df[self._start_col]

    @property
    def end(self):
        """End values."""
        return self._df[self._end_col]

    @property
    def chromosome_col(self):
        """Chromosome column name."""
        return self._chrom_col

    @property
    def start_col(self):
        """Start column name."""
        return self._start_col

    @property
    def end_col(self):
        """End column name."""
        return self._end_col

    @property
    def trees(self):
        """Trees used for indexing the DataFrame."""

        if self._trees is None:
            self._trees = self._build_trees()

        return self._trees

    def _build_trees(self):
        # Subset frame to positions and rename columns to defaults.
        position_df = self._df[[
            self._chrom_col, self._start_col, self._end_col
        ]]
        position_df.columns = ['chromosome', 'start', 'end']

        # Add index and sort by chromosome (for grouping).
        position_df = position_df.assign(index=np.arange(len(self._df)))
        position_df = position_df.sort_values(by='chromosome')

        # Convert to tuples.
        tuples = ((tup.chromosome, tup.start, tup.end, tup.index)
                  for tup in position_df.itertuples())

        return GenomicIntervalTree.from_tuples(tuples)

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


def reorder_columns(df, order):
    """Reorders dataframe columns, sorting any extra columns alphabetically."""

    extra_cols = set(df.columns) - set(order)
    return df[list(order) + sorted(extra_cols)]
