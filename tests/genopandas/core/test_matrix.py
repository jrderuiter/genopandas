import numpy as np
import pandas as pd
import pytest

from genopandas.core.frame import GenomicDataFrame
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
def feature_data():
    """Example feature data."""

    return pd.DataFrame(
        {
            'biotype': ['a', 'a', 'b', 'b']
        }, index=['f1', 'f2', 'f3', 'f4'])


@pytest.fixture
def example_matrix(feature_values, sample_data, feature_data):
    """Example feature matrix."""

    return AnnotatedMatrix(
        feature_values, sample_data=sample_data, feature_data=feature_data)


class TestAnnotatedMatrix(object):
    """Tests for AnnotatedMatrix class."""

    def test_init(self, feature_values, sample_data, feature_data):
        """Tests for basic init."""

        matrix = AnnotatedMatrix(
            feature_values, sample_data=sample_data, feature_data=feature_data)

        assert all(matrix.values == feature_values)
        assert list(matrix.columns) == ['s1', 's2', 's3', 's4']
        assert list(matrix.index) == ['f1', 'f2', 'f3', 'f4']

        assert all(matrix.sample_data == sample_data)
        assert all(matrix.feature_data == feature_data)

    def test_init_without_samples(self, feature_values):
        """Tests for init without sample_data."""

        matrix = AnnotatedMatrix(feature_values)

        assert all(matrix.values == feature_values)
        assert list(matrix.columns) == ['s1', 's2', 's3', 's4']
        assert list(matrix.index) == ['f1', 'f2', 'f3', 'f4']

        dummy = pd.DataFrame({}, index=['s1', 's2', 's3', 's4'])
        assert all(matrix.sample_data == dummy)

    def test_loc(self, example_matrix):
        """Tests subsetting matrix using loc."""

        # Test subsetting with list.
        subset = example_matrix.loc[['f2', 'f3']]
        assert list(subset.index) == ['f2', 'f3']

        # Test single element, should return series.
        subset2 = example_matrix.loc['f2']
        assert isinstance(subset2, pd.Series)

    def test_iloc(self, example_matrix):
        """Tests subsetting matrix using iloc."""

        # Test subsetting with list.
        subset = example_matrix.iloc[[0, 1]]
        assert list(subset.index) == ['f1', 'f2']
        assert list(subset.columns) == ['s1', 's2', 's3', 's4']

        # Test single element, should return series.
        subset2 = example_matrix.iloc[1]
        assert isinstance(subset2, pd.Series)

    def test_get_item(self, example_matrix):
        """Tests subsetting samples using get_item."""

        # Tests subsetting with list.
        subset = example_matrix[['s2', 's3']]
        assert list(subset.columns) == ['s2', 's3']

        # Tests single element.
        subset2 = example_matrix['s2']
        assert isinstance(subset2, pd.Series)

    def test_rename(self, example_matrix):
        """Tests renaming."""

        # Test renaming of samples.
        renamed = example_matrix.rename(columns={'s2': 's5'})
        assert list(renamed.columns) == ['s1', 's5', 's3', 's4']
        assert list(renamed.values.columns) == ['s1', 's5', 's3', 's4']
        assert list(renamed.sample_data.index) == ['s1', 's5', 's3', 's4']

        # Test renaming of features.
        renamed = example_matrix.rename(index={'f2': 'f5'})
        assert list(renamed.index) == ['f1', 'f5', 'f3', 'f4']
        assert list(renamed.values.index) == ['f1', 'f5', 'f3', 'f4']
        assert list(renamed.feature_data.index) == ['f1', 'f5', 'f3', 'f4']

    def test_query(self, example_matrix):
        """Tests subsetting samples with query_samples."""

        subset = example_matrix.query_samples('phenotype == "sensitive"')
        assert list(subset.columns) == ['s1', 's3']

        assert list(subset.values.columns) == ['s1', 's3']
        assert list(subset.sample_data.index) == ['s1', 's3']

    def test_dropna_samples(self, feature_values, feature_data):
        """Tests dropping samples with NAs in sample_data."""

        sample_data = pd.DataFrame(
            {
                'phenotype': ['sensitive', None, None, 'resistant']
            },
            index=['s1', 's2', 's3', 's4'])

        matrix = AnnotatedMatrix(
            feature_values, sample_data=sample_data, feature_data=feature_data)

        matrix = matrix.dropna_samples()

        assert list(matrix.columns) == ['s1', 's4']
        assert list(matrix.sample_data.index) == ['s1', 's4']

    def test_dropna(self, example_matrix):
        """Tests dropping features with NAs in values."""

        example_matrix.values.loc['f2', 's2'] = None

        matrix2 = example_matrix.dropna(axis=0)
        assert list(matrix2.index) == ['f1', 'f3', 'f4']
        assert list(matrix2.columns) == ['s1', 's2', 's3', 's4']

        matrix3 = example_matrix.dropna(axis=1)
        assert list(matrix3.index) == ['f1', 'f2', 'f3', 'f4']
        assert list(matrix3.columns) == ['s1', 's3', 's4']
        assert list(matrix3.sample_data.index) == ['s1', 's3', 's4']

    def test_from_csv(self, sample_data):
        """Tests from_csv."""

        file_path = pytest.helpers.data_path('matrix_features.tsv')

        matrix = AnnotatedMatrix.from_csv(
            file_path, sep='\t', sample_data=sample_data)

        assert list(matrix.columns) == ['s1', 's2', 's3', 's4']
        assert list(matrix.index) == ['f1', 'f2', 'f3', 'f4']

        assert list(matrix.sample_data.index) == ['s1', 's2', 's3', 's4']
        assert list(matrix.sample_data.columns) == ['phenotype']

    def test_from_csv_extra(self, sample_data):
        """Tests from_csv with an extra column."""

        file_path = pytest.helpers.data_path('matrix_features_extra.tsv')

        matrix = AnnotatedMatrix.from_csv(
            file_path, sep='\t', sample_data=sample_data, drop_cols=['extra'])

        assert list(matrix.columns) == ['s1', 's2', 's3', 's4']
        assert list(matrix.index) == ['f1', 'f2', 'f3', 'f4']


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


@pytest.fixture
def logratio_matrix():
    """Example dataset containing logratios."""

    file_path = pytest.helpers.data_path('logratios.tsv')

    return GenomicMatrix.from_csv_condensed(
        file_path, sep='\t', is_one_based=True, is_inclusive=False)


@pytest.fixture
def genes():
    """Example genes."""

    file_path = pytest.helpers.data_path('genes.tsv')

    return GenomicDataFrame.from_csv(
        file_path, sep='\t', index_col=['chromosome', 'start', 'end'])


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

    def test_from_csv(self):
        """Tests reading using from_csv."""

        file_path = pytest.helpers.data_path('matrix_ranged.tsv')

        matrix = GenomicMatrix.from_csv(
            file_path, sep='\t', index_col=['chromosome', 'start', 'end'])

        assert list(matrix.columns) == ['s1', 's2', 's3', 's4']
        assert list(matrix.gloc.chromosome) == ['1', '1', '2', '2']
        assert list(matrix.gloc.start) == [20, 30, 10, 50]
        assert list(matrix.gloc.end) == [30, 40, 25, 60]

    def test_from_csv_condensed(self):
        """Tests reading using from_csv_condensed."""

        file_path = pytest.helpers.data_path('matrix_ranged_condensed.tsv')

        matrix = GenomicMatrix.from_csv_condensed(file_path, sep='\t')

        assert list(matrix.columns) == ['s1', 's2', 's3', 's4']
        assert list(matrix.gloc.chromosome) == ['1', '1', '2', '2']
        assert list(matrix.gloc.start) == [20, 30, 10, 50]
        assert list(matrix.gloc.end) == [30, 40, 25, 60]

    def test_from_csv_condensed_pos(self):
        """Tests reading using from_csv_condensed with positions."""

        file_path = pytest.helpers.data_path('matrix_ranged_condensed_pos.tsv')

        matrix = GenomicMatrix.from_csv_condensed(
            file_path,
            sep='\t',
            index_regex=r'(?P<chromosome>\w+):(?P<position>\d+)')

        assert list(matrix.columns) == ['s1', 's2', 's3', 's4']
        assert list(matrix.gloc.chromosome) == ['1', '1', '2', '2']
        assert list(matrix.gloc.start) == [20, 30, 10, 50]
        assert list(matrix.gloc.end) == [21, 31, 11, 51]
        assert list(matrix.gloc.position) == [20, 30, 10, 50]

    def test_expand(self, logratio_matrix):
        """Test expansion using expand."""

        assert logratio_matrix.shape[0] == 9

        # Check shape.
        expanded = logratio_matrix.expand()
        assert expanded.shape[0] == 10

        # Check imputed row.
        assert expanded.index[2] == ('1', 3100000, 3150000)
        assert all(expanded.iloc[2].isnull())

        # Check not everything is null (misaligned index).
        assert not expanded.values.isnull().all().all()

    def test_impute(self, logratio_matrix):
        """Tests imputation."""

        assert logratio_matrix.shape[0] == 9

        # Check shape.
        imputed = logratio_matrix.impute()
        assert imputed.shape[0] == 10

        # Check imputed row.
        assert imputed.index[2] == ('1', 3100000, 3150000)
        assert not any(imputed.iloc[2].isnull())

        # Check not everything is null (misaligned index).
        assert not imputed.values.isnull().all().all()

    def test_resample(self, logratio_matrix):
        """Tests resampling."""

        # Check shape.
        resampled = logratio_matrix.resample(bin_size=100000)
        assert resampled.shape[0] == 5

        # Check indices.
        assert list(resampled.gloc.start) == [
            3000000, 3100000, 3200000, 3300000, 3400000
        ]
        assert list(resampled.gloc.end) == [
            3100000, 3200000, 3300000, 3400000, 3500000
        ]

        # Check values.
        assert not resampled.isnull().all().all()

    def test_annotate(self, logratio_matrix, genes):
        """Tests annotation of logratio matrix."""

        annotated = logratio_matrix.annotate(genes, feature_id='name')

        assert list(annotated.index) == ['gene a']
        assert list(annotated.columns) == list(logratio_matrix.columns)

        expected = logratio_matrix.iloc[:2].mean(axis=0).values
        assert all((annotated.values.iloc[0].values - expected) < 1e-6)
