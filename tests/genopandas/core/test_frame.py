"""Tests for pandas-related classes/functions."""

import pandas as pd
import pytest

from genopandas.core.frame import GenomicDataFrame

# pylint: disable=redefined-outer-name,no-self-use


@pytest.fixture()
def ranged_df():
    """Simple example dataframe."""
    return pd.DataFrame.from_records([
        ('1', 20, 30),
        ('1', 25, 40),
        ('2', 30, 50)
    ], columns=['chromosome', 'start', 'end'])  # yapf: disable


@pytest.fixture()
def ranged_gdf(ranged_df):
    """Simple example genomic dataframe."""
    return GenomicDataFrame(ranged_df)


class TestGenomicDataFrame(object):
    """Tests for GenomicDataFrame class."""

    def test_ranged_init(self, ranged_df):
        """Test init with a ranged frame."""

        gdf = GenomicDataFrame(ranged_df)

        # Check shape of frame.
        assert len(gdf) == 3
        assert list(gdf.columns) == ['chromosome', 'start', 'end']

        # Check index works.
        assert list(gdf.gi.chromosome) == ['1', '1', '2']
        assert list(gdf.gi.start) == [20, 25, 30]
        assert list(gdf.gi.end) == [30, 40, 50]

    def test_ranged_init_with_different_cols(self, ranged_df):
        """Tests init with ranged frame using different column names."""

        ranged_df = ranged_df.rename(columns={
            'chromosome': 'chrom',
            'start': 'chromStart',
            'end': 'chromEnd'
        })

        gdf = GenomicDataFrame(
            ranged_df,
            chromosome_col='chrom',
            start_col='chromStart',
            end_col='chromEnd')

        # Check shape of frame.
        assert len(gdf) == 3
        assert list(gdf.columns) == ['chrom', 'chromStart', 'chromEnd']

        # Check index.
        assert list(gdf.gi.chromosome) == ['1', '1', '2']
        assert list(gdf.gi.start) == [20, 25, 30]
        assert list(gdf.gi.end) == [30, 40, 50]

        # Check chromosomes.
        assert gdf.gi.chromosomes == ['1', '2']

    def test_search(self, ranged_gdf):
        """Tests for search function."""

        # Search on chromosome 1.
        subset = ranged_gdf.gi.search('1', 10, 22)
        assert subset.gi.chromosomes == ['1']
        assert len(subset) == 1
        assert list(subset.columns) == list(ranged_gdf.columns)

        # Search outside existing interval.
        subset = ranged_gdf.gi.search('1', 100, 200)
        assert len(subset) == 0
        assert list(subset.columns) == list(ranged_gdf.columns)

        # Search non-existing chromosome.
        with pytest.raises(KeyError):
            ranged_gdf.gi.search('4', 100, 200)

    def test_chromosome_lengths(self, ranged_gdf):
        """Tests for chromosome_lengths property."""

        # Without specified lengths.
        assert ranged_gdf.gi.chromosome_lengths == {'1': 40, '2': 50}

        # With specified lengths.
        ranged_gdf2 = GenomicDataFrame(
            ranged_gdf, chrom_lengths={'1': 100, '2': 200})  # yapf: disable
        assert ranged_gdf2.gi.chromosome_lengths == {'1': 100, '2': 200}
