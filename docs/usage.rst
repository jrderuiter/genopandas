=====
Usage
=====

GenoPandas implements two main data structures, a ``GenomicDataFrame`` and an
``AnnotatedMatrix``. GenomicDataFrames can be used to store any type of
location-based genomic data and provides efficient querying a genomic
intervaltree structure. AnnotatedMatrices are used to store various types of
numeric data in feature-by-sample matrices, together with an optional sample
annotation. Various specializations of the AnnotatedMatrix provide further
specializations for specific data types (such as gene-expression or copy number
data) and include support for various manipulations and visualizations of
these data.

GenomicDataFrames
-----------------

The ``GenomicDataFrame`` class is a subclass of the pandas DataFrame class and
therefore supports the same basic interface. Data stored in a GenomicDataFrame
is however required to contain columns describing the genomic positions of each
observation.

Construction
~~~~~~~~~~~~

From column-based dataframes
============================

For many genomic data types, such as GTF and BED formatted data, information
about genomic regions is contained with the columns of the DataFrame. For this
type of data, the easiest way to create a GenomicDataFrame is to use an
existing DataFrame and specify the columns that contain ``chromosome``,
``start`` and ``end`` positions for each observation. By default, these data
are assumed to be in the 'chromosome', 'start' and 'end' columns:

.. code-block:: python

    df = pd.DataFrame.from_records(
        [('1', 10, 20), ('2', 10, 20), ('2', 30, 40)],
        columns=['chromosome', 'start', 'end'])

    GenomicDataFrame(df)

If the ``chromosome``, ``start`` and ``end`` positions are in different
columns, the names of these columns can be provided for each of these columns
upon construction as follows:

.. code-block:: python

    df = pd.DataFrame.from_records(
        [('1', 10, 20), ('2', 10, 20), ('2', 30, 40)],
        columns=['chrom', 'chromStart', 'chromEnd'])

    GenomicDataFrame(
        df,
        chromosome_col='chrom',
        start_col='chromStart',
        end_col='chromEnd')

Finally, for positional data, which is located at a single base position rather
than spanning a larger region, it is possible to omit the end position by
specifying ``end_col`` as None:

.. code-block:: python

    GenomicDataFrame(
        df,
        chromosome_col='chrom',
        start_col='position',
        end_col=None)

From index-based dataframes
===========================

For matrix-based data formats, it is more convienent to store positional
information in the index of the DataFrame than in its columns. For this reason,
GenomicDataFrames can also be instantiated using DataFrames that contain
genomic positions in the index of the DataFrame. For region-based data, the
DataFrame should contain a MultiIndex with three levels (chromosome, start end
end):

.. code-block:: python

    df = pd.DataFrame.from_records(
        [('1', 10, 20, 4), ('2', 10, 20, 5), ('2', 30, 40, 6)],
        columns=['chromosome', 'start', 'end', 'value'])

    df = df.set_index(['chromosome', 'start', 'end'])

    GenomicDataFrame(df, use_index=True)

For positional data, the index should contain only two levels (chromosome
and position):

.. code-block:: python

    df = pd.DataFrame.from_records(
        [('1', 10, 4), ('2', 10, 5), ('2', 30, 6)],
        columns=['chromosome', 'position', 'value'])

    df = df.set_index(['chromosome', 'position'])

    GenomicDataFrame(df, use_index=True)


From various file formats
=========================

GenomicDataFrames can also be constructed directly from a various number of
genomic file formats. Generic table-based formats can be read direclty using
the ``from_csv`` method, which is similar to the pandas ``read_csv`` function.
Specialized functions are provided for common genomic file formats, including
the BED and GTF file formats.

Querying by position
~~~~~~~~~~~~~~~~~~~~

As a sub-class of the pandas ``DataFrame`` class GenomicDataFrames can be
queried and manipulated in the same manner as normal DataFrames. However,
GenomicDataFrames also provide an additional ``GenomicIndexer`` under the
``gi`` property, which uses a ``GenomicIntervalTree`` data structure to
efficiently select rows of the DataFrame by a genomic position or range.

The main method of the ``GenomicIndexer`` is the ``search`` method, which
returns a new frame containing all rows within the specified genomic range:

.. code-block:: python

    subset = gdf.gi.query('2', 10, 20)


Accessing positions
~~~~~~~~~~~~~~~~~~~

The main method of the ``GenomicIndexer`` is the ``search`` method, which
returns a new frame containing all rows within the specified genomic range:

.. code-block:: python

    subset = gdf.gi.query('2', 10, 20)

The indexer also provides direct access to the chromosome, start and
end values using its ``chromosome``, ``start`` and ``end`` properties.

Finally, the indexer also provides access to offset start/end positions, which
are offset by the lengths of the preceding chromosomes. This is particularly
useful when visualizing data on a genomic axis over multiple chromosomes. The
offset positions can be accessed using the ``start_offset`` and ``end_offset``
properties. The chromosome lengths and offsets are also available via the
``chromosome_lengths`` and ``chromosome_offsets`` properties.

By default, chromosome lengths are extrapolated from the genomic positions in
the GenomicDataFrame. However, these positions may underestimate the chromosome
length if they do not span the entire chromosome. To avoid this issue, the
correct chromosomal lengths can be supplied using the ``chrom_lengths``
argument when constructing the frame.


AnnotatedMatrices
-----------------

The ``AnnotatedMatrix`` base class provides basic functionality for storing
a numeric matrix (with 'features' along the rows and samples along the columns),
together with additional metadata describing the samples. This format is ideal
for storing data from different types of high-throughput measurements (such as
gene-expression counts or copy number calls) together with the corresponding
sample phenotypes and other properties.

FeatureMatrix
~~~~~~~~~~~~~

GenoPandas currently provides two main subclasses of the ``AnnotatedMatrix``
class: the ``FeatureMatrix`` and the ``RegionMatrix``. FeatureMatrices are used
to store values that are indexed by a set of (named) features, such as
gene expression matrices (which contain counts summarized per gene).

Construction
============
The easiest way to construct a feature matrix is using a pre-existing DataFrame:

.. code-block:: python

    df = pd.DataFrame({
            'sample_1': [1, 2, 3],
            'sample_2': [4, 5, 6]
        },
        index=['gene_a', 'gene_b', 'gene_c'])

    matrix = FeatureMatrix(df)

Sample information can be included by passing a DataFrame using the ``design``
argument. Note that the sample metadata should be indexed by sample name to
match the matrix correctly:

.. code-block:: python

    df = pd.DataFrame({
            'sample_1': [1, 2, 3],
            'sample_2': [4, 5, 6]
        },
        index=['gene_a', 'gene_b', 'gene_c'])

    design = pd.DataFrame(
        {'condition': ['control', 'treated']},
        index=['sample_a', 'sample_b'])

    matrix = FeatureMatrix(df, design=design)

Once constructed,  the matrix values can be accessed using the ``values``
property, which returns the matrix in DataFrame format. The sample design can
be retrieved using the ``design`` property. The list of available features and
samples can be obtained using the ``features`` and ``samples`` properties,
respectively.

Subsetting
==========

 - loc
 - __getitem__
 - rename
 - query
 - dropna
 - dropna_values

 - concat

Plotting
========

- plot_heatmap
- plot_pca?

RegionMatrix
~~~~~~~~~~~~

In constrast, RegionMatrices are used to store values that are indexed by genomic
regions, such as copy-number ratios measured in bins across the genome. Besides
these two base classes, a number of specialized derivatives (such as the
``ExpressionMatrix`` class are provided for these more specific use cases.


Specialized matrices
~~~~~~~~~~~~~~~~~~~~
