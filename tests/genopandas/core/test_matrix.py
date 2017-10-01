import numpy as np
import pandas as pd
import pytest

from genopandas.core.matrix import AnnotatedMatrix, GenomicMatrix


@pytest.fixture
def feature_values(random):
    """Example feature matrix values."""

    return pd.DataFrame(
        np.random.randn(4, 4),
        columns=['s1', 's2', 's3', 's4'],
        index=['f1', 'f2', 'f3', 'f4'])


@pytest.fixture
def sample_data():
    """Example sample data."""

    return pd.DataFrame(
        {
            'phenotype': ['sensitive', 'resistant'] * 2
        },
        index=['s1', 's2', 's3', 's4'])


@pytest.fixture
def annotated_matrix(feature_values, sample_data):
    """Example feature matrix."""

    return AnnotatedMatrix(feature_values, sample_data=sample_data)


class TestAnnotatedMatrix(object):
    """Tests for AnnotatedMatrix class."""

    def test_init(self, feature_values, sample_data):
        """Tests for basic init."""

        matrix = AnnotatedMatrix(feature_values, sample_data=sample_data)

        assert all(matrix.values == feature_values)
        assert matrix.samples == ['s1', 's2', 's3', 's4']
        assert matrix.features == ['f1', 'f2', 'f3', 'f4']

        assert all(matrix.sample_data == sample_data)

    def test_init_without_samples(self, feature_values):
        """Tests for init without sample_data."""

        matrix = AnnotatedMatrix(feature_values)

        assert all(matrix.values == feature_values)
        assert matrix.samples == ['s1', 's2', 's3', 's4']
        assert matrix.features == ['f1', 'f2', 'f3', 'f4']

        dummy = pd.DataFrame({}, index=['s1', 's2', 's3', 's4'])
        assert all(matrix.sample_data == dummy)

    def test_loc(self, annotated_matrix):
        """Tests subsetting matrix using loc."""

        # Test subsetting with list.
        subset = annotated_matrix.loc[['f2', 'f3']]
        assert subset.features == ['f2', 'f3']

        # Test single element, should return series.
        subset2 = annotated_matrix.loc['f2']
        assert isinstance(subset2, pd.Series)

    def test_iloc(self, annotated_matrix):
        """Tests subsetting matrix using iloc."""

        # Test subsetting with list.
        subset = annotated_matrix.iloc[[0, 1]]
        assert subset.features == ['f1', 'f2']
        assert subset.samples == ['s1', 's2', 's3', 's4']

        assert list(subset.values.columns) == ['s1', 's2', 's3', 's4']
        assert list(subset.sample_data.index) == ['s1', 's2', 's3', 's4']

        # Test single element, should return series.
        subset2 = annotated_matrix.iloc[1]
        assert isinstance(subset2, pd.Series)

    def test_get_item(self, annotated_matrix):
        """Tests subsetting samples using get_item."""

        # Tests subsetting with list.
        subset = annotated_matrix[['s2', 's3']]
        assert subset.samples == ['s2', 's3']

        assert list(subset.values.columns) == ['s2', 's3']
        assert list(subset.sample_data.index) == ['s2', 's3']

        # Tests single element.
        subset2 = annotated_matrix['s2']
        assert isinstance(subset2, pd.Series)

    def test_rename(self, annotated_matrix):
        """Tests renaming samples."""

        renamed = annotated_matrix.rename({'s2': 's5'})
        assert renamed.samples == ['s1', 's5', 's3', 's4']

        assert list(renamed.values.columns) == ['s1', 's5', 's3', 's4']
        assert list(renamed.sample_data.index) == ['s1', 's5', 's3', 's4']

    def test_rename_drop(self, annotated_matrix):
        """Tests renaming samples, dropping extra."""

        renamed = annotated_matrix.rename({'s2': 's5', 's3': 's6'}, drop=True)
        assert renamed.samples == ['s5', 's6']

        assert list(renamed.values.columns) == ['s5', 's6']
        assert list(renamed.sample_data.index) == ['s5', 's6']

    def test_query(self, annotated_matrix):
        """Tests subsetting samples with query."""

        subset = annotated_matrix.query('phenotype == "sensitive"')
        assert subset.samples == ['s1', 's3']

        assert list(subset.values.columns) == ['s1', 's3']
        assert list(subset.sample_data.index) == ['s1', 's3']

    def test_dropna(self, feature_values):
        """Tests dropping samples with NAs in sample_data."""

        sample_data = pd.DataFrame(
            {
                'phenotype': ['sensitive', None, None, 'resistant']
            },
            index=['s1', 's2', 's3', 's4'])
        matrix = AnnotatedMatrix(feature_values, sample_data=sample_data)

        matrix = matrix.dropna()

        assert matrix.samples == ['s1', 's4']

    def test_drop_values(self, feature_values, sample_data):
        """Tests dropping features with NAs in values."""

        matrix = AnnotatedMatrix(feature_values, sample_data=sample_data)
        matrix.values.loc['f2', 's2'] = None

        matrix2 = matrix.dropna_values(axis=0)
        assert matrix2.features == ['f1', 'f3', 'f4']
        assert matrix2.samples == ['s1', 's2', 's3', 's4']

        matrix3 = matrix.dropna_values(axis=1)
        assert matrix3.features == ['f1', 'f2', 'f3', 'f4']
        assert matrix3.samples == ['s1', 's3', 's4']

    def test_concat(self, annotated_matrix):
        """Tests concatenation of annotated matrices."""

        # Concat along feature axis.
        mat = AnnotatedMatrix(
            pd.DataFrame(
                np.random.randn(2, 4),
                columns=['s1', 's2', 's3', 's4'],
                index=['f5', 'f6']))

        merged = AnnotatedMatrix.concat([annotated_matrix, mat])
        assert merged.features == ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
        assert merged.samples == ['s1', 's2', 's3', 's4']
        assert list(merged.sample_data.index) == ['s1', 's2', 's3', 's4']

        # Concat along sample axis.
        mat2 = AnnotatedMatrix(
            pd.DataFrame(
                np.random.randn(4, 2),
                columns=['s5', 's6'],
                index=['f1', 'f2', 'f3', 'f4']))

        merged2 = AnnotatedMatrix.concat([annotated_matrix, mat2], axis=1)
        assert merged2.features == ['f1', 'f2', 'f3', 'f4']
        assert merged2.samples == ['s1', 's2', 's3', 's4', 's5', 's6']
        assert list(merged2.sample_data.index) == \
            ['s1', 's2', 's3', 's4', 's5', 's6']

    def test_concat_duplicates(self, annotated_matrix):
        """Tests concatenation with duplciates."""

        # Test with duplicate features.
        mat = AnnotatedMatrix(
            pd.DataFrame(
                np.random.randn(2, 4),
                columns=['s1', 's2', 's3', 's4'],
                index=['f3', 'f4']))

        with pytest.raises(ValueError):
            AnnotatedMatrix.concat([annotated_matrix, mat])

        # Test with duplicate samples.
        mat2 = AnnotatedMatrix(
            pd.DataFrame(
                np.random.randn(4, 2),
                columns=['s3', 's4'],
                index=['f1', 'f2', 'f3', 'f4']))

        with pytest.raises(ValueError):
            AnnotatedMatrix.concat([annotated_matrix, mat2], axis=1)

    def test_from_csv(self, sample_data):
        """Tests from_csv."""

        file_path = pytest.helpers.data_path('matrix_features.tsv')
        matrix = AnnotatedMatrix.from_csv(
            file_path, sep='\t', sample_data=sample_data)

        assert matrix.samples == ['s1', 's2', 's3', 's4']
        assert matrix.features == ['f1', 'f2', 'f3', 'f4']

        assert list(matrix.sample_data.index) == ['s1', 's2', 's3', 's4']
        assert list(matrix.sample_data.columns) == ['phenotype']

    def test_from_csv_extra(self, sample_data):
        """Tests from_csv with an extra column."""

        file_path = pytest.helpers.data_path('matrix_features_extra.tsv')

        matrix = AnnotatedMatrix.from_csv(
            file_path, sep='\t', sample_data=sample_data, drop_cols=['extra'])

        assert matrix.samples == ['s1', 's2', 's3', 's4']
        assert matrix.features == ['f1', 'f2', 'f3', 'f4']


@pytest.fixture
def genomic_values(random):
    """Example feature matrix values."""

    index = pd.MultiIndex.from_tuples(
        [('1', 20, 30), ('1', 30, 40), ('2', 10, 25), ('2', 50, 60)],
        names=['chromosome', 'start', 'end'])

    return pd.DataFrame(
        np.random.randn(4, 4), columns=['s1', 's2', 's3', 's4'], index=index)


@pytest.fixture
def genomic_matrix(genomic_values, sample_data):
    """Example feature matrix."""

    return GenomicMatrix(genomic_values, sample_data=sample_data)


class TestGenomicMatrix(object):
    """Tests for GenomicMatrix base class."""

    def test_init(self, genomic_values, sample_data):
        """Tests basic init."""
        matrix = GenomicMatrix(genomic_values, sample_data)
        assert all(matrix.values == genomic_values)
        assert all(matrix.sample_data == sample_data)

    def test_gloc_subset(self, genomic_matrix):
        """Tests subsetting chromosomes using gloc."""

        subset = genomic_matrix.gloc[['1']]
        assert list(genomic_matrix.values.gloc.chromosomes) == ['1', '2']
        assert list(subset.values.gloc.chromosomes) == ['1']

    def test_gloc_reorder(self, genomic_matrix):
        """Tests reordering chromosomes using gloc."""

        reordered = genomic_matrix.gloc[['2', '1']]

        assert list(genomic_matrix.values.gloc.chromosomes) == ['1', '2']
        assert list(reordered.values.gloc.chromosomes) == ['2', '1']

    def test_gloc_search(self, genomic_matrix):
        """Tests search using gloc."""

        subset = genomic_matrix.gloc.search('1', 10, 30)
        assert isinstance(subset, GenomicMatrix)
        assert subset.shape[0] == 1

    def test_gloc_slice(self, genomic_matrix):
        """Tests slicing using gloc."""

        subset = genomic_matrix.gloc['1'][10:30]
        assert subset.shape[0] == 1

        assert subset == genomic_matrix.gloc.search('1', 10, 30)

    def test_gloc_attributes(self, genomic_matrix):
        """Tests (wrapped) accessor attributes on gloc."""

        values = genomic_matrix.values

        assert all(genomic_matrix.gloc.chromosome == values.gloc.chromosome)
        assert all(genomic_matrix.gloc.start == values.gloc.start)
        assert all(genomic_matrix.gloc.end == values.gloc.end)
        assert genomic_matrix.gloc.chromosome_lengths == \
                   values.gloc.chromosome_lengths
        assert genomic_matrix.gloc.chromosome_offsets == \
                   values.gloc.chromosome_offsets
