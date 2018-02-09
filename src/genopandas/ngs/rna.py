import numpy as np

from genopandas.core.matrix import AnnotatedMatrix


class ExpressionMatrix(AnnotatedMatrix):
    """Matrix containing (gene) expression values (features-by-samples)."""

    @classmethod
    def from_subread(cls,
                     file_path,
                     sample_data=None,
                     sample_mapping=None,
                     **kwargs):
        """Reads expression from a subread output file."""

        return super().from_csv(
            file_path,
            sample_data=sample_data,
            sample_mapping=sample_mapping,
            drop_cols=['Chr', 'Start', 'End', 'Strand', 'Length'],
            index_col=0,
            sep='\t',
            **kwargs)

    def normalize(self, size_factors=None, log2=False):
        """Normalizes expression counts for sequencing depth.

        Normalizes by dividing sample counts using the given (sample) size
        factors. If no size factors are given, they are calculated using the
        median-of-ratios approach used by DESeq2.

        Parameters
        ----------
        size_factors : np.array
            Array of size factors, length should be equal to the number
            of samples.
        log2 : bool
            Whether to also log2-transform the normalized counts.

        Returns
        -------
        ExpressionMatrix
            ExpressionMatrix containing normalized counts.

        """

        with np.errstate(divide="ignore"):
            if size_factors is None:
                size_factors = self._estimate_size_factors(self._values)
            normalized = self._values.divide(size_factors, axis=1)

        if log2:
            normalized = np.log2(normalized + 1)

        return self._constructor(normalized)

    @staticmethod
    def _estimate_size_factors(counts):
        """Calculate size factors for DESeq's median-of-ratios normalization."""

        def _estimate_size_factors_col(counts, log_geo_means):
            log_counts = np.log(counts)
            mask = np.isfinite(log_geo_means) & (counts > 0)
            return np.exp(np.median((log_counts - log_geo_means)[mask]))

        with np.errstate(divide="ignore"):
            log_geo_means = np.mean(np.log(counts), axis=1)

            size_factors = np.apply_along_axis(
                _estimate_size_factors_col,
                axis=0,
                arr=counts,
                log_geo_means=log_geo_means)

        return size_factors
