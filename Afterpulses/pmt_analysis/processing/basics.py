import numpy as np
import pandas as pd
import warnings
from typing import Tuple, List, Optional


class FixedWindow:
    """Basic processing functions using a fixed window for baseline / peak calculations.
    Applicable for (externally) triggered data taking, such as during LED calibrations.

    Attributes:
        bounds_baseline: Lower (included) and upper (excluded) index of window for baseline calculations.
            Unbound if element(s) `None`.
        bounds_peak: Lower (included) and upper (excluded) index of window for peak calculations.
            Unbound if element(s) `None`.
    """

    def __init__(self, bounds_baseline: Tuple[Optional[int], Optional[int]],
                 bounds_peak: Tuple[Optional[int], Optional[int]]):
        """Init of the FixedWindow class.

        Defines baseline and peak window boundaries.

        Args:
            bounds_baseline: Lower (included) and upper (excluded) index of window for baseline calculations.
            bounds_peak: Lower (included) and upper (excluded) index of window for peak calculations.
        """
        self.bounds_baseline = self.set_bounds(bounds_baseline, 'bounds_baseline')
        self.bounds_peak = self.set_bounds(bounds_peak, 'bounds_peak')

    def set_bounds(self, input_tuple: Tuple[Optional[int], Optional[int]],
                   input_tuple_name: str) -> Tuple[Optional[int], Optional[int]]:
        """Check and convert boundary tuples.

        Used for `bounds_baseline` and `bounds_peak`. Assert that tuple of length 2 with integers.

        Args:
            input_tuple: Tuple to check and convert.
            input_tuple_name: Name of the tuple, used in error messages.

        Returns:
            output_tuple: Tuple of length 2 with integers.
        """
        # Check that iterable is passed.
        if not hasattr(input_tuple, '__iter__'):
            raise TypeError('Expected iterable (ideally tuple) for {}.'.format(input_tuple_name))
        # Check length of passed iterable.
        if len(input_tuple) != 2:
            raise ValueError('Expected tuple of length 2 for {}.'.format(input_tuple_name))
        # Convert to list to make mutable.
        output_list = list(input_tuple)
        # Try to convert values to int if necessary.
        for i, el in enumerate(output_list):
            if (type(el) not in [int, np.int64]) and (el is not None):
                try:
                    output_list[i] = int(el)
                    warnings.warn('Converted element {} in {} to int. This may lead to unwanted behavior. '
                                  'Recommended to directly pass tuples of integers.'.format(i, input_tuple_name))
                except Exception:
                    raise TypeError('Expected tuple of integers / Nones for {}.'.format(input_tuple_name))
        # Convert to tuple to make immutable.
        output_tuple = tuple(output_list)
        # Assert that tuple elements of ascending order.
        if np.all([el is not None for el in output_tuple]) and output_tuple[0] >= output_tuple[1]:
            raise ValueError('Elements in {} must be of ascending order or None.format(input_tuple_name)')
        return output_tuple

    def get_baseline(self, input_data: np.ndarray) -> np.ndarray:
        """Calculate baselines of individual waveforms.

        Args:
            input_data: Array with ADC data of shape (number of waveforms, time bins per waveform).

        Returns:
            bsl: Array with baseline values of shape (number of waveforms, ).
        """
        bsl = np.median(input_data[:, self.bounds_baseline[0]:self.bounds_baseline[-1]], axis=1)
        return bsl

    def get_baseline_std(self, input_data: np.ndarray) -> np.ndarray:
        """Calculate baseline standard deviations of individual waveforms.

        Args:
            input_data: Array with ADC data of shape (number of waveforms, time bins per waveform).

        Returns:
            bsl: Array with baseline standard deviation values of shape (number of waveforms, ).
        """
        bsl_std = np.std(input_data[:, self.bounds_baseline[0]:self.bounds_baseline[-1]], axis=1, ddof=1)
        return bsl_std

    def get_amplitude(self, input_data: np.ndarray) -> np.ndarray:
        """Calculate naive peak amplitudes as baseline subtracted maximum outlier in window of individual waveforms.

        Args:
            input_data: Array with ADC data of shape (number of waveforms, time bins per waveform).

        Returns:
            amp_val: Array with amplitude values of shape (number of waveforms, ).
        """
        bsl_val = self.get_baseline(input_data)
        min_val = np.min(input_data[:, self.bounds_peak[0]:self.bounds_peak[-1]], axis=1)
        amp_val = bsl_val - min_val
        return amp_val

    def get_peak_position(self, input_data: np.ndarray) -> np.ndarray:
        """Calculate naive peak position index as maximum outlier in window of individual waveforms.

        Args:
            input_data: Array with ADC data of shape (number of waveforms, time bins per waveform).

        Returns:
            amp_val: Array with peak position index values of shape (number of waveforms, ).
        """
        min_pos = np.argmin(input_data[:, self.bounds_peak[0]:self.bounds_peak[-1]], axis=1)
        # Correct for lower peak window boundary.
        # Zero offset if lower boundary None (defined to falsify in boolean or).
        min_pos_corrected = min_pos + (self.bounds_peak[0] or 0)
        return min_pos_corrected

    def get_area(self, input_data: np.ndarray) -> np.ndarray:
        """Calculate naive baseline subtracted peak areas.

        Args:
            input_data: Array with ADC data of shape (number of waveforms, time bins per waveform).

        Returns:
            area_val: Array with peak area values of shape (number of waveforms, ).
        """
        gross_area = np.sum(input_data[:, self.bounds_peak[0]:self.bounds_peak[-1]], axis=1)
        window_width_peak = np.sum(np.ones(input_data.shape[1])[self.bounds_peak[0]:self.bounds_peak[-1]])
        bsl_val = self.get_baseline(input_data)
        bsl_area = window_width_peak * bsl_val
        area_val = bsl_area - gross_area
        return area_val

    def get_basic_properties(self, input_data: np.ndarray,
                             parameters: List[str] = ['baseline', 'amplitude', 'area']) -> pd.DataFrame:
        """Calculate basic waveform properties, such as baseline, peak amplitude and area,...

        Args:
            input_data: Array with ADC data of shape (number of waveforms, time bins per waveform).
            parameters: List of names of the parameters to be calculated.

        Returns:
            df_out: Pandas data frame with parameters of interest as columns.
        """
        if type(input_data) != np.ndarray:
            if type(input_data) == list:
                input_data = np.array(input_data)
            else:
                raise TypeError('input_data must be of type numpy.ndarray.')
        if input_data.ndim != 2:
            raise ValueError('input_data must have ndim = 2 dimension.')
        lower_bounds_list = [self.bounds_peak[0], self.bounds_baseline[0], input_data.shape[1]-1]
        lower_bounds_list = np.array([el for el in lower_bounds_list if el is not None])
        lower_bounds_list = lower_bounds_list + input_data.shape[1] * (lower_bounds_list < 0)
        if max(lower_bounds_list) >= input_data.shape[1]:
            raise ValueError('A lower window boundary was selected that extends beyond the data array length.')

        df_out = pd.DataFrame()
        if 'baseline' in parameters:
            df_out['baseline'] = self.get_baseline(input_data)
        if 'baseline_std' in parameters:
            df_out['baseline_std'] = self.get_baseline_std(input_data)
        if 'amplitude' in parameters:
            df_out['amplitude'] = self.get_amplitude(input_data)
        if 'peak_position' in parameters:
            df_out['peak_position'] = self.get_peak_position(input_data)
        if 'area' in parameters:
            df_out['area'] = self.get_area(input_data)
        if df_out.shape == (0, 0):
            warnings.warn("Output from get_basic_properties seems to be empty.")
        return df_out


class FullWindow(FixedWindow):
    """Basic processing functions using the full waveform window for baseline / peak calculations
    without dedicated peak finding. Applicable especially for naive random dark count finding.

    Attributes:
        n_slices: Number of slices for robust baseline and baseline standard deviation calculation.
    """

    def __init__(self, n_slices: int = 7):
        """Init of the FullWindow class.

        Defines baseline and peak window boundaries as None to use full waveform window.

        Args:
            n_slices: Number of slices for robust baseline and baseline standard deviation calculation.
        """
        FixedWindow.__init__(self, bounds_baseline=(None, None), bounds_peak=(None, None))
        self.n_slices = n_slices

    def get_baseline(self, input_data: np.ndarray) -> np.ndarray:
        """Calculate baselines of individual waveforms with reduced peaks impact.

        Apply more robust approach reducing the pull by localized peaks by calculating the median
        in `n_slices` slices of approximately (but not necessarily exactly)
        equal size and taking the median of those values as baseline value.

        Args:
            input_data: Array with ADC data of shape (number of waveforms, time bins per waveform).

        Returns:
            bsl: Array with baseline values of shape (number of waveforms, ).
        """
        bsl = np.median(np.asarray([np.median(el, axis=1) for el in
                                    np.array_split(input_data, self.n_slices, axis=1)]).T, axis=1)
        return bsl

    def get_baseline_std(self, input_data: np.ndarray) -> np.ndarray:
        """Calculate baseline standard deviations of individual waveforms with reduced peaks impact.

        Apply more robust approach reducing the pull by localized peaks by calculating the
        standard deviation in `n_slices` slices of approximately (but not necessarily exactly)
        equal size and taking the median of those values as baseline standard deviation value.

        Args:
            input_data: Array with ADC data of shape (number of waveforms, time bins per waveform).

        Returns:
            bsl: Array with baseline standard deviation values of shape (number of waveforms, ).
        """
        bsl_std = np.median(np.asarray([np.std(el, axis=1) for el in
                                        np.array_split(input_data, self.n_slices, axis=1)]).T, axis=1)
        return bsl_std
