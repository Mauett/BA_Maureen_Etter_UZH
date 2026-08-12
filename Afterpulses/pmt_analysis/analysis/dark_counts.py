import numpy as np
import warnings


class DarkCountRate:
    """Class for dark count rate estimations through search for random dark counts above threshold in waveforms.

    Attributes:
        amplitudes: Array with maximum data pulse amplitudes per waveform.
        threshold: Amplitude threshold to consider pulse as dark count. Usually set to amplitude values
            corresponding to pulses of 0.25 PE or 0.5 PE area, deduced from an approximately linear
            readout-specific dependence between amplitude and area.
        samples_per_wf: Samples per waveform used for amplitude determination.
        adc_f: ADC sampling frequency in samples per second.
    """

    def __init__(self, amplitudes: np.ndarray, threshold: float,
                 samples_per_wf: int, adc_f: float = 500e6):
        """Init of the DarkCountRate class.

        Args:
            amplitudes: Array with maximum data pulse amplitudes per waveform.
            threshold: Amplitude threshold to consider pulse as dark count. Usually set to amplitude values
                corresponding to pulses of 0.25 PE or 0.5 PE area, deduced from an approximately linear
                readout-specific dependence between amplitude and area.
            samples_per_wf: Samples per waveform used for amplitude determination.
            adc_f: ADC sampling frequency in samples per second as provided by same attribute of
                `pmt_analysis.utils.input.ADCRawData`. Default: 500e6
                (ADC sampling frequency of 500 MS/s for CAEN v1730d).
        """
        if threshold < 0:
            threshold = - threshold
            warnings.warn('Amplitudes are defined to be positive, the passed threshold value is hence '
                          'converted to a positive float.')
        self.amplitudes = amplitudes
        self.threshold = threshold
        self.samples_per_wf = samples_per_wf
        self.adc_f = adc_f

    @staticmethod
    def correction_poisson(fraction_above_thr: float) -> float:
        """Poisson-correction for dark count rates where the probability of more than one dark count per
        wave function is not negligible.

        As `fraction_above_thr` represents an estimator for the probability of at least one dark count per waveform,
        the mean probability of a dark count in a waveform can be calculated assuming Poisson statistics.
        Using the discrete probability distribution

        .. math::
            m = P(k > 0, \lambda) = 1 - P(k = 0, \lambda) = 1 - \lambda^0/0! \cdot e^{-\lambda} = 1 - e^{-\lambda}

        one obtains the corrected dark count probability as

        .. math::
            \lambda = -ln(1-m).

        Args:
            fraction_above_thr: Fraction of waveforms with amplitude above threshold, hence assumed to
                contain at least one dark count.

        Returns:
            dc_prob_per_wf: Dark count probability per waveform.
        """
        if (fraction_above_thr >= 1) or (fraction_above_thr < 0):
            raise ValueError('Argument fraction_above_thr must be between 0 and 1.')
        dc_prob_per_wf = -np.log(1 - fraction_above_thr)
        return dc_prob_per_wf

    def compute(self) -> tuple:
        """Compute dark count rate and corresponding uncertainty in units of dark counts / second.

        Returns:
            dc_rate: Dark count rate in units of dark counts / second.
            dc_rate_unc: Dark count rate uncertainty in units of dark counts / second.
        """
        # Fraction of waveforms with amplitude above threshold
        fraction_above_thr = float(np.mean(self.amplitudes > self.threshold))
        # Assume Poisson uncertainty sqrt(n) on counts above threshold
        fraction_above_thr_unc = np.sqrt(fraction_above_thr/self.amplitudes.shape[0])
        # Poisson-correction for non-negligible multiple dark count probability per waveform
        dc_prob_per_wf = self.correction_poisson(fraction_above_thr)
        # Error propagation with d(-ln(1-x))/dx = 1/(1-x)
        dc_prob_per_wf_unc = fraction_above_thr_unc/(1-fraction_above_thr)
        # Temporal width per waveform from number of samples and sampling frequency
        wf_time_width = self.samples_per_wf / self.adc_f
        # Dark count rate and uncertainty
        dc_rate = dc_prob_per_wf / wf_time_width
        dc_rate_unc = dc_prob_per_wf_unc / wf_time_width

        return dc_rate, dc_rate_unc
