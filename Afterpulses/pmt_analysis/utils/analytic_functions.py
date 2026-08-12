import numpy as np
import warnings


def func_gain_vs_voltage(voltage: np.ndarray, a: float, k: float, n: float = 10) -> np.ndarray:
    """Power-law dependence of the PMT gain on the supply voltage according to
    `T. Hakamata, Photomultiplier tubes: basics and applications, 3a (Hamamatsu Photonics K.K., 2007)`.

    Args:
        voltage: Absolute PMT supply voltage.
        a: Empirical constant.
        k: Empirical constant.
        n: Number of dynode stages. Default: 10 (value for Hamamatsu R12699-406-M4).

    Returns:
        gain: PMT gain for the given supply voltage and PMT parameters.
    """
    if np.any(np.array(voltage) < 0):
        raise ValueError('Function assumes absolute (i.e., positive) voltages, '
                         'even if the PMT is physically operated at negative high voltage.')
    params = np.array([a, k, n])
    params_names = np.array(['a', 'k', 'n'])
    if np.any(params < 0):
        raise ValueError('Parameter(s) {} in func_gain_vs_voltage is / are < 0 which is '
                         'unphysical.'.format(', '.join(params_names[params < 0])))
    if n not in [10]:
        warnings.warn('k different from default value of 10 for Hamamatsu R12699-406-M4 was passed.')
    gain = a**n / (n+1)**(k*n) * voltage**(k*n)
    return gain
