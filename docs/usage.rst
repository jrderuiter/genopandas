=====
Usage
=====

GenoPandas provides two main data structures for storing genomics data, the
``GenomicDataFrame`` class and an ``AnnotatedMatrix`` class. GenomicDataFrames
can be used to store any type of location-based genomic data and provides
efficient querying a genomic intervaltree structure. AnnotatedMatrices are used
to store various types of numeric data in feature-by-sample matrices, together
with an optional sample annotation. Various specializations of the
AnnotatedMatrix provide further specializations for specific data types
(such as gene-expression or copy number data) and include support for various
manipulations and visualizations of these data.

GenomicDataFrames
-----------------

The ``GenomicDataFrame`` class is a subclass of the pandas DataFrame class and
therefore supports the same basic interface as a normal pandas DataFrame.
However, in contrast to pandas DataFrames, GenomicDataFrames are required to
have a MultiIndex containing three levels describing the genomic range of
each row as (chromosome, start_position, end_position). These positions
are zero-based and the start position is inclusive, whilst the end position is
exclusive. This means that positioned data (which does not span a range, but
is located at an exact genomic position) can be described as (chromosome,
start_position, start_position + 1). GenomicDataFrames use this MultiLevel
index to provide efficient querying of genomic ranges using a ``gloc`` indexer,
which is backed by an intervaltree data structure.

Construction
~~~~~~~~~~~~

From existing DataFrames
========================

A GenomicDataFrame can easily be constructed from an existing DataFrame as
follows:

.. code-block:: python

    from genopandas import GenomicDataFrame

    df = pd.DataFrame.from_records(
        [('1', 10, 20, 'a'),
         ('2', 10, 20, 'b'),
         ('2', 30, 40, 'c')],
        columns=['chromosome', 'start', 'end', 'name'])

    df = df.set_index(['chromosome', 'start', 'end'])

    GenomicDataFrame(df)

Note the setting of the index, which is required for querying the data later.
The constructor does currently not check of the presence of the index, but
any querying will result in errors if a proper index is missing.

If you want to check for the presence of a suitable index, you can use the
``from_df`` classmethod, which explicitly checks the index of the given
dataframe:

.. code-block:: python

    GenomicDataFrame.from_df(df)

If a positioned dataframe (with two index levels, chromosome and
position) is given to ``from_df``, this index is automatically expanded to
three levels containing start/end positions. Alternatively,
``from_position_df`` can be used to explicitly expand positioned data,
which allows the width of the expanded items to be specified using the
``width`` parameter:

.. code-block:: python

    df = pd.DataFrame.from_records(
        [('1', 10, 'a'),
         ('2', 10, 'b'),
         ('2', 30, 'c')],
        columns=['chromosome', 'position', 'name'])

    df = df.set_index(['chromosome', 'position'])

    GenomicDataFrame.from_position_df(df, width=10)

From (delimited) files
======================

GenomicDataFrames can also be read from delimited files using the ``from_csv``
method. This method mimics the ``pd.read_csv`` method, but requires the
``index_col`` argument to indicate which three columns to use for constructing
the index. Using this approach, a GTF file (for example) can be read as follows:

.. code-block:: python

    gdf = GenomicDataFrame.from_csv(
        '../data/example.gtf',
        sep='\t',
        names=('contig', 'source', 'feature', 'start', 'end', 'score',
               'strand', 'frame', 'attributes'),
        index_col=['contig', 'start', 'end'])

    gdf.head()

For several common file types (e.g. BED, GTF), we provide specialized functions
for reading data into a GenomicDataFrame. For example, a GTF file can be read
more easily than above using the ``from_gtf`` method:

.. code-block:: python

    gdf = gpd.GenomicDataFrame.from_gtf('../data/example.gtf')
    gdf.head()

Similarly, bed files can be read using the ``from_bed`` method. See the API
for the full list of supported file formats. Any unsupported file formats can
of course be read using the ``from_csv`` method or using the pandas API, as
illustrated above.

Querying
~~~~~~~~

As a subclass of the pandas ``DataFrame`` class, GenomicDataFrames can be
queried in the same manner as normal pandas DataFrames using loc and iloc.
However, GenomicDataFrames also provide an additional indexer under the
``gloc`` property, which can be used to perform ranged queries over
the GenomicDataFrame.

A simple range query can be performed as follows, in which we select any rows
overlapping with bases 79935353-79935455 on chromosome 12:

.. code-block:: python

    gdf = gpd.GenomicDataFrame.from_gtf('../data/example.gtf')
    gdf.gloc['12'][79935353:79935455]

In this query, the ``gloc`` indexer returns a slice object for chromosome 12,
which we then slice using the given range coordinates to only select rows within
the specified range.

Entire chromosomes can be selected by passing a list of chromosome names to
``gloc``, which also reorders chromosomes to match the given order:

.. code-block:: python

    gdf = gpd.GenomicDataFrame.from_gtf('../data/example.gtf')
    gdf.gloc[['10', '12']]

More complex queries (e.g. with left/right inclusiveness) can be performed using
the ``gloc.search`` method, which provides the above functionality with some
extra options.

Positioning
~~~~~~~~~~~

The ``gloc`` indexer can also be used to extract genomic positions directly,
using the ``chromosome``, ``start`` and ``end`` attributes of the indexer:

.. code-block:: python

    gdf.gloc.chromosome  # Chromosome values.
    gdf.gloc.start       # Start positions.
    gdf.gloc.end         # End positions.

The available chromosomes and their lengths are available through the
``chromosomes`` and ``chromosome_lengths`` attributes. Note that
chromosome lengths are inferred from the data if these were not given to
the GenomicDataFrame constructor.

The chromosome lengths can be used to calculate offset start/end positions, in
which the lengths of preceding chromosomes are included in the start/end
positions. This can be useful for plotting data linearly across multiple
chromosomes. These offset positions can be accessed using the ``start_offset``
and the ``end_offset`` attributes.

AnnotatedMatrices
-----------------

``AnnotatedMatrix`` classes provides functionality for storing a numeric
matrix with features along the rows and samples along the columns, together
with additional metadata describing the samples/features. This format
is ideal for storing data from different types of high-throughput measurements
(such as gene-expression counts or copy number calls) together with the
corresponding sample/feature data.

The base ``AnnotatedMatrix`` class can be used to store values that are indexed
by a set of (named) features, such as gene expression matrices (which contain
counts summarized per gene).

Construction
~~~~~~~~~~~~

The easiest way to construct an AnnotatedMatrix is using a pre-existing
DataFrame. Sample and feature information can be included by passing a DataFrame
using the ``sample_data`` and ``feature_data`` arguments, respectively. Note
that the indices of these annotations should correspond with the matrix
row/column indices:

.. code-block:: python

    from genopandas import AnnotatedMatrix

    df = pd.DataFrame({
            'sample_1': [1, 2, 3],
            'sample_2': [4, 5, 6]
        },
        index=['gene_a', 'gene_b', 'gene_c'])

    sample_data = pd.DataFrame(
        {'condition': ['control', 'treated']},
        index=['sample_1', 'sample_2'])

    AnnotatedMatrix(df, sample_data=sample_data)

Once constructed,  the matrix values can be accessed using the ``values``
property, which returns the matrix in DataFrame format. The sample and feature
annotations can be retrieved using the ``sample_data`` and ``feature_data``
attributes.

Subsetting samples/features
~~~~~~~~~~~~~~~~~~~~~~~~~~~

An AnnotatedMatrix can be subset using the same column/index accessors as
pandas DataFrames (e.g., .loc, .iloc and [] for selecting columns). In this
case, the AnnotatedMatrix class ensures that feature/sample annotations are
kept in line with the subsetted matrix.

Besides this, a number of specialized methods allow subsetting of the matrix
based on the sample/feature annotations. Currently this includes the
``query_samples`` and ``dropna_samples`` methods, which can be used to query
for specific samples or drop samples with NA values in their annotations,
respectively. In general, these methods follow the API of their pandas
equivalents.

Renaming samples/features
~~~~~~~~~~~~~~~~~~~~~~~~~

Features and/or samples can be renamed using the ``rename`` method, using
the ``index`` parameter for features and the ``columns`` parameter for samples.
The corresponding sample/feature annotations are renamed accordingly.

Melting to 'tidy' format
~~~~~~~~~~~~~~~~~~~~~~~~

Matrices can be 'melted' into a tidy format (a.k.a. long format), which may
be more suitable for certain types of processing/visualization than the matrix
format. This type of transformation is performed using the ``melt`` method,
which returns a 'tidy' pandas DataFrame. Optionally, the parameters
``with_sample_data`` and ``with_feature_data`` can be used to indicate whether
sample/feature annotations should be included in the produced DataFrame.

.. code-block:: python

    import seaborn as sns

    df_long = matrix.melt(with_sample_data=True)
    sns.boxplot(data=df_long, x='condition', y='value')

Plotting
~~~~~~~~

Several high-level plotting functions are provided for plotting matrix values
in different representations. For example, the ``plot_heatmap`` method is most
useful for plotting a (clustered) overview of the matrix values, with optional
feature/sample annotations:

.. code-block:: python

    matrix.plot_heatmap(sample_cols=['condition'])

Similarly, the ``plot_pca`` method can be used to plot a PCA transform of the
matrix values. This transformation can be performed along either the sample or
feature axes and can be colored according to specific sample/feature
annotations:

.. code-block:: python

    matrix.plot_pca(hue='condition', axis='samples')

Additionally, the ``plot_feature`` method can be used to create categorical
plots (boxplot, swarmplot or violin plot) of feature values. These plots
can be grouped by different feature characteristics to compare distributions
of feature values across different sample groups:

.. code-block:: python

    matrix.plot_feature('gene_a', group='condition')

GenomicMatrices
---------------

Similar to the GenomicsDataFrame, the ``GenomicMatrix`` class is a specialized
version ``AnnotatedMatrix`` class that supports storing and querying of
genomically-positioned data. Besides this, the GenomicMatrix class provides
additional functionality specific to manipulating and plotting
genomically-oriented data.

Construction
~~~~~~~~~~~~

GenomicMatrix instances can be constructed in the same manner as
AnnotatedMatrices, although the matrix values should be supplied as a
GenomicsDataFrame (with a MultiIndex containing three levels):

.. code-block:: python

    import numpy as np
    import pandas as pd

    from genopandas import GenomicMatrix


    data = pd.DataFrame({
        'chromosome': ['1'] * 50 + ['2'] * 50,
        'start': np.hstack([range(0, 500, 10),
                            range(0, 500, 10)]),
        'end': np.hstack([range(10, 510, 10),
                          range(10, 510, 10)]),
        'sample_1': np.hstack([np.random.randn(50),
                            np.random.randn(50) + 10]),
        'sample_2': np.hstack([np.random.randn(20) + 15,
                            np.random.randn(30) + 0,
                            np.random.randn(50) + -10])
    })

    matrix = GenomicMatrix(data)
    matrix.head()

Matrix values can also be read from delimited files using the ``from_csv``
method. Besides this, the class also provides a ``from_csv_condensed``
method, which can expand a 'condensed' index (such as 1:10-20) to a multi-level
index suitable for GenomicsDataFrames. The regex used for this expansion
can be defined using the ``index_regex`` parameter of this method.

Querying ranges
~~~~~~~~~~~~~~~

Similar to the GenomicDataFrame class, GenomicMatrices can be subset to
specific genomic ranges using the ``gloc`` indexer. For more details, see the
GenomicDataFrame documentation.

Resampling/imputation
~~~~~~~~~~~~~~~~~~~~~

For certain analyses or visualizations, it can be useful to resample a
GenomicMatrix at a lower resolution or using specific bin sizes. The
``resample`` method can be used to resample a matrix to a given bin_size,
optionally starting from a given start position:

.. code-block:: python

    matrix.resample(bin_size=20, start=0)

Imputation can be used to impute missing values from surrounding
bins. The ``impute`` method can be used for this purpose, which uses the
rolling median functionality from pandas to impute values from surrounding bins:

.. code-block:: python

    matrix.impute(window=11, min_probes=5)

Plotting
~~~~~~~~

Similar to the AnnotatedMatrix class, the GenomicMatrix class provides a
``plot_heatmap`` method for plotting a heatmap of matrix values along a genomic
axis:

.. code-block:: python

    resampled = matrix.resample(bin_size=20, start=0)
    resampled.plot_heatmap()

The ``plot_sample`` method can be used to plot values for a single sample:

.. code-block:: python

    matrix.plot_sample('sample_1', markersize=5)


Specialized matrices
--------------------

Besides the basic ``AnnotatedMatrix`` and ``GenomicMatrix`` classes, a number
of more specalized matrix sub-classes are provided in the ``genopandas.ngs``
module. This currently includes the ``CnvValueMatrix`` and ``CnvCallMatrix``
classes for CNV data and the ``ExpressionMatrix`` class for RNA-seq expression
data. See the respective class documentation for more information on
class-specific functionality.

