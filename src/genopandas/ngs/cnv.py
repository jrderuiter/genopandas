import numpy as np
import pandas as pd

from genopandas.core.frame import GenomicDataFrame
from genopandas.core.matrix import AnnotatedMatrix, GenomicMatrix


class CnvValueMatrix(GenomicMatrix):
    """CnvMatrix containing (segmented) logratio values (positions-by-samples).
    """

    @classmethod
    def as_segments(cls, values):
        """Returns matrix as segments (consecutive stetches with same value).

        Assumes that values have already been segmented, i.e. that bins
        in the same segment have been assigned same numeric value.
        """
        values = values.sort_index()

        # Get segments per sample.
        segments = pd.concat(
            (cls._segments_for_sample(sample_values)
             for _, sample_values in values.items()),
            axis=0, ignore_index=True)  # yapf: disable

        segments = segments.set_index(
            ['chromosome', 'start', 'end'], drop=False)

        return GenomicDataFrame(segments)

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
