import numpy as np

from genopandas.core.matrix import AnnotatedMatrix


class CnvCallMatrix(AnnotatedMatrix):
    def mask_with_controls(self, column, mask_value=0.0):
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
