"""Tests for pandas-related classes/functions."""

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import pytest

from genopandas.core.frame import GenomicDataFrame

# pylint: disable=redefined-outer-name,no-self-use


@pytest.fixture()
def ranged_df():
    """Simple example of a ranged dataframe."""

    index = pd.MultiIndex.from_tuples(
        [('1', 20, 30), ('1', 30, 40), ('2', 10, 25), ('2', 50, 60)],
        names=['chromosome', 'start', 'end'])

    df = pd.DataFrame(
        np.random.randn(4, 4),
        columns=['s{}'.format(i + 1) for i in range(4)],
        index=index)

    return df


@pytest.fixture()
def positioned_df():
    """Simple example of a ranged dataframe."""

    index = pd.MultiIndex.from_tuples(
        [('1', 20), ('1', 30), ('2', 10), ('2', 50)],
        names=['chromosome', 'position'])

    df = pd.DataFrame(
        np.random.randn(4, 4),
        columns=['s{}'.format(i + 1) for i in range(4)],
        index=index)

    return df


@pytest.fixture()
def ranged_gdf(ranged_df):
    """Simple example genomic dataframe."""
    return GenomicDataFrame(ranged_df)


class TestGenomicDataFrame(object):
    """Tests for GenomicDataFrame class."""

    def test_init(self, ranged_df):
        """Tests init with example data."""

        gdf = GenomicDataFrame(ranged_df)

        # Check shape of frame.
        assert len(gdf) == 4
        assert list(gdf.columns) == ['s1', 's2', 's3', 's4']

        # Check index access via named properties.
        assert list(gdf.gloc.chromosome) == ['1', '1', '2', '2']
        assert list(gdf.gloc.start) == [20, 30, 10, 50]
        assert list(gdf.gloc.end) == [30, 40, 25, 60]
        assert list(gdf.gloc.position) == [25, 35, 17, 55]

        # Check available chromosomes.
        assert gdf.gloc.chromosomes == ['1', '2']

    def test_gloc_subset(self, ranged_gdf):
        """Tests subsetting chromosomes using gloc."""

        subset = ranged_gdf.gloc[['2']]

        assert len(subset) == 2
        assert list(subset.gloc.chromosome) == ['2', '2']
        assert subset.gloc.chromosomes == ['2']

    def test_gloc_reorder(self, ranged_gdf):
        """Tests reordering chromosomes using gloc."""

        subset = ranged_gdf.gloc[['2', '1']]

        assert len(subset) == 4
        assert list(subset.gloc.chromosome) == ['2', '2', '1', '1']
        assert subset.gloc.chromosomes == ['2', '1']

    def test_gloc_slice(self, ranged_gdf):
        """Tests slicing of dataframe using gloc."""

        subset = ranged_gdf.gloc['1'][10:30]
        assert len(subset) == 1

    def test_gloc_search(self, ranged_gdf):
        """Test searching of dataframe using gloc."""

        # Test same example as slice.
        subset = ranged_gdf.gloc.search('1', start=10, end=30)
        assert len(subset) == 1

        # Test strict search with example within bounds...
        subset = ranged_gdf.gloc.search(
            '1', start=10, end=30, strict_right=True)
        assert len(subset) == 1

        # ...and extending beyond bounds.
        subset = ranged_gdf.gloc.search(
            '1', start=10, end=29, strict_right=True)
        assert len(subset) == 0

    def test_gloc_lengths(self, ranged_gdf):
        """Tests computation of chromosome lengths."""

        expected = {'1': 39, '2': 59}
        assert ranged_gdf.gloc.chromosome_lengths == expected

    def test_gloc_lengths_predefined(self, ranged_df):
        """Tests use of predefined lengths."""

        # Test example.
        ranged_gdf = GenomicDataFrame(
            ranged_df, chrom_lengths={
                '1': 120,
                '2': 90,
            })

        expected = {'1': 120, '2': 90}
        assert ranged_gdf.gloc.chromosome_lengths == expected

        # Test example with extra (unused) chromosome.
        # Should keep the chromosome in lengths.
        ranged_gdf = GenomicDataFrame(
            ranged_df, chrom_lengths={'1': 120,
                                      '2': 90,
                                      '3': 80})

        expected = {'1': 120, '2': 90, '3': 80}
        assert ranged_gdf.gloc.chromosome_lengths == expected

    def test_gloc_offsets(self, ranged_gdf):
        """Tests computation of offsets."""

        # Check offsets.
        expected = {'1': 0, '2': 39, '_END_': 98}
        assert ranged_gdf.gloc.chromosome_offsets == expected

        # Check offset starts/ends.
        assert list(ranged_gdf.gloc.start_offset) == [20, 30, 49, 89]
        assert list(ranged_gdf.gloc.end_offset) == [30, 40, 64, 99]
        assert list(ranged_gdf.gloc.position_offset) == [25, 35, 56, 94]

    def test_gloc_offsets_predefined(self, ranged_df):
        """Tests use of predefined lengths."""

        # Test example.
        ranged_gdf = GenomicDataFrame(
            ranged_df, chrom_lengths={'1': 120,
                                      '2': 90})

        expected = {'1': 0, '2': 120, '_END_': 210}
        assert ranged_gdf.gloc.chromosome_offsets == expected

        # Test example with extra (unused) chromosome.
        # Should not omit chromosome from offsets.
        ranged_gdf = GenomicDataFrame(
            ranged_df, chrom_lengths={'1': 120,
                                      '2': 90,
                                      '3': 80})

        expected = {'1': 0, '2': 120, '3': 210, '_END_': 290}
        assert ranged_gdf.gloc.chromosome_offsets == expected

    def test_from_csv(self):
        """Tests reading data from tsv."""

        # Read file.
        file_path = pytest.helpers.data_path('frame_ranged.tsv')
        gdf = GenomicDataFrame.from_csv(
            file_path, index_col=['chromosome', 'start', 'end'], sep='\t')

        # Check shape.
        assert len(gdf) == 4
        assert list(gdf.columns) == ['s1', 's2', 's3', 's4']

        # Check some dtypes.
        assert not is_numeric_dtype(gdf.index.get_level_values(0))
        assert is_numeric_dtype(gdf.index.get_level_values(1))
        assert is_numeric_dtype(gdf.index.get_level_values(2))

    def test_from_gtf(self):
        """Tests reading data from gtf."""

        # Read file.
        file_path = pytest.helpers.data_path('reference.gtf.gz')
        gdf = GenomicDataFrame.from_gtf(file_path)

        # Check shape.
        assert len(gdf) > 0
        assert list(gdf.columns[:8]) == [
            'contig', 'source', 'feature', 'start', 'end', 'score', 'strand',
            'frame'
        ]

        # Check some dtypes.
        assert not is_numeric_dtype(gdf['contig'])
        assert is_numeric_dtype(gdf['start'])
        assert is_numeric_dtype(gdf['end'])

    def test_from_gtf_filter(self):
        """Tests reading data from gtf with filter."""

        # Read file.
        file_path = pytest.helpers.data_path('reference.gtf.gz')
        gdf = GenomicDataFrame.from_gtf(
            file_path, filter_=lambda rec: rec.feature == 'gene')

        # Check result.
        assert len(gdf) > 0
        assert set(gdf['feature']) == {'gene'}

    def test_from_records_tuple(self):
        """Tests from_records with tuple input."""

        records = [('1', 20, 30, 'gene', 'GeneA'),
                   ('1', 40, 60, 'gene', 'GeneB')]  # yapf: disable

        # Test with named columns.
        gdf = GenomicDataFrame.from_records(
            records,
            columns=['chromosome', 'start', 'end', 'feature', 'name'],
            index_col=['chromosome', 'start', 'end'])

        assert len(gdf) == 2
        assert list(gdf.columns) == ['feature', 'name']

        # Test un-named columns.
        gdf2 = GenomicDataFrame.from_records(records, index_col=[0, 1, 2])
        assert len(gdf2) == 2
        assert list(gdf2.columns) == [3, 4]

    def test_from_records_tuple_dict(self):
        """Tests from_records with dict input."""

        records = [{'chromosome': '1', 'start': 20, 'end': 30,
                    'feature': 'gene', 'name': 'GeneA'},
                   {'chromosome': '1', 'start': 40, 'end': 60,
                    'feature': 'gene', 'name': 'GeneB'}]  # yapf: disable

        gdf = GenomicDataFrame.from_records(
            records, index_col=['chromosome', 'start', 'end'])

        assert len(gdf) == 2
        assert list(gdf.columns) == ['feature', 'name']

    def test_from_position_df(self, positioned_df):
        """Test from_position_df method."""
        gdf = GenomicDataFrame.from_position_df(positioned_df)

        assert list(gdf.gloc.chromosome) == ['1', '1', '2', '2']
        assert list(gdf.gloc.start) == [20, 30, 10, 50]
        assert list(gdf.gloc.end) == [21, 31, 11, 51]
        assert list(gdf.gloc.position) == [20, 30, 10, 50]

    def test_from_df_positioned(self, positioned_df):
        """Test from_df method with positioned data."""

        gdf = GenomicDataFrame.from_df(positioned_df)

        assert list(gdf.gloc.chromosome) == ['1', '1', '2', '2']
        assert list(gdf.gloc.start) == [20, 30, 10, 50]
        assert list(gdf.gloc.end) == [21, 31, 11, 51]
        assert list(gdf.gloc.position) == [20, 30, 10, 50]
