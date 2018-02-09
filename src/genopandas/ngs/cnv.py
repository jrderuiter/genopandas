import numpy as np
import pandas as pd

from genopandas.core.frame import GenomicDataFrame
from genopandas.core.matrix import AnnotatedMatrix, GenomicMatrix


class CnvValueMatrix(GenomicMatrix):
    """CnvMatrix containing (segmented) logratio values (positions-by-samples).
    """

    def as_segments(self, drop_columns=True):
        """Returns matrix as segments (consecutive stetches with same value).

        Assumes that values have already been segmented, i.e. that bins
        in the same segment have been assigned same numeric value.

        Parameters
        ----------
        drop_columns : bool
            Whether to drop chromosome, start, end and sample columns
            after setting the index.

        Returns
        -------
        GenomicDataFrame
            GenomicDataFrame describing genomic segments. Indexed by
            chromosome, start, end and sample.

            Note that the sample index is included to avoid duplicate index
            errors when reindexing in cases where samples have identical
            segments.
        """

        values = self._values.sort_index()

        # Get segments per sample.
        segment_data = pd.concat(
            (self._segments_for_sample(sample_values)
             for _, sample_values in values.items()),
            axis=0, ignore_index=True)  # yapf: disable

        # Set index. Note that we add sample here, to avoid running into
        # duplicate index errors when re-indexing later down the road.
        # This is a bit of a hack, but GenomicDataFrames shouldn't suffer
        # from having any extra index entries.
        segment_data = segment_data.set_index(
            ['chromosome', 'start', 'end', 'sample'], drop=drop_columns)

        segments = GenomicDataFrame(segment_data)
        segments = segments.gloc[self.gloc.chromosomes]

        return segments

    @staticmethod
    def _segments_for_sample(sample_values):
        # Calculate segment ids (distinguished by diff values).
        segment_ids = np.cumsum(_padded_diff(sample_values) != 0)

        # Get sample and position columns.
        sample = sample_values.name
        chrom_col, start_col, end_col = sample_values.index.names

        # Group and determine positions + values.
        grouped = sample_values.reset_index().groupby(
            by=[chrom_col, segment_ids])

        segments = grouped.agg({
            chrom_col: 'first',
            start_col: 'min',
            end_col: 'max',
            sample: ['first', 'size']
        })

        # Flatten column levels and rename.
        segments.columns = ['_'.join(s) for s in segments.columns]
        segments = segments.rename(columns={
            chrom_col + '_first': 'chromosome',
            start_col + '_min': 'start',
            end_col + '_max': 'end',
            sample + '_first': 'value',
            sample + '_size': 'size'
        })

        # Add sample name and reorder columns.
        segments = segments.reindex(
            columns=['chromosome', 'start', 'end', 'value', 'size'])
        segments['sample'] = sample

        return segments.reset_index(drop=True)

    def to_igv(self, file_path):
        """Saves data for viewing in IGV."""

        igv_data = self._values.reset_index()

        # Rename index columns.
        igv_columns = ['Chromosome', 'Start', 'End']
        column_map = dict(zip(self._values.index.names, igv_columns))

        igv_data = igv_data.rename(columns=column_map)

        # Add 'Feature' column.
        feature_names = ['P{}'.format(i + 1) for i in range(igv_data.shape[0])]
        igv_data.insert(4, 'Feature', feature_names)

        # Write file.
        with open(file_path, 'w') as file_:
            print('#type=COPY_NUMBER', file=file_)
            igv_data.to_csv(file_, sep='\t', index=False, header=True)


class CnvCallMatrix(AnnotatedMatrix):
    """Cnv matrix containing CNV calls (genes-by-samples)."""

    def mask_with_controls(self, column, mask_value=0.0):
        """Masks calls present in control samples.

        Calls are retained if (a) no call is present in the matched control
        sample, (b) if the sample call is more extreme than the control sample
        or (c) the sample and control have calls with different signs
        (loss/gain).

        Matched control samples should be indicated by the given column
        in the sample_data annotation.
        """

        control_samples = self._sample_data[column].dropna()

        new_values = self._values.copy()
        for sample, ctrl in dict(control_samples).items():
            mask = self._call_mask(self._values[ctrl], self._values[sample])
            new_values.loc[~mask, sample] = mask_value

        return self._constructor(new_values)

    @staticmethod
    def _call_mask(ctrl_values, sample_values):
        """Returns mask in which entries are True where ctrl and sample
           have different signs or the sample has a more extreme value.
        """

        ctrl_sign = np.sign(ctrl_values)
        sample_sign = np.sign(sample_values)

        diff_sign = (ctrl_sign - sample_sign).abs() > 1e-8
        higher_val = sample_values.abs() > ctrl_values.abs()

        return diff_sign | (~diff_sign & higher_val)


def _padded_diff(values, pad_value=0):
    """Same as np.diff, with leading 0 to keep same length as input."""
    diff = np.diff(values)
    return np.pad(
        diff, pad_width=(1, 0), mode='constant', constant_values=pad_value)
