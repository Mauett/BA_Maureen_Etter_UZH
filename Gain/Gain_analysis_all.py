#!/usr/bin/env python3
"""
gain_analysis_one_plot.py

Model-independent PMT gain plus model-dependent Poisson-constrained multi-PE fit.

This version makes one combined figure:
  top: entries vs signal electrons with the fitted components
  bottom: residual precision plot using the same x-axis

The fit model and fit calls are intentionally kept the same as in the supplied
script; only plotting/output layout is changed.
"""

import argparse
import json
import math
import os
import re
import tempfile

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 24,
})

from scipy.stats import chi2, exponnorm

from pmt_analysis.utils.input import ADCRawData
from pmt_analysis.processing.basics import FixedWindow
from pmt_analysis.analysis.model_independent import GainModelIndependent

try:
    from scipy.optimize import curve_fit
except Exception:
    curve_fit = None


from pathlib import Path

BASE_OUTPUT_DIR = Path(__file__).resolve().parent / "Gain_Results"
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_MAX_PE = 6
PE_PLOT_MIN_FRAC = 0.005
TAIL_W_MAX = 0.50
CHI2_MIN_EXPECTED = 5.0

# X-axis settings display range, not fit.
X_AXIS_DENSITY_WINDOW_BINS = 25
X_AXIS_MODEL_EXTRA_PE = 1.20
X_AXIS_MODEL_SIGMA_MARGIN = 6.0


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "t", "1", "yes", "y"}:
        return True
    if value in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


parser = argparse.ArgumentParser(
    description="Model-independent occupancy/gain plus Poisson-constrained model-dependent PMT gain."
)
parser.add_argument("-pon", "--input_path_on", type=str, required=True)
parser.add_argument("-poff", "--input_path_off", type=str, required=True)
parser.add_argument("-bbl", "--bsl_bound_lower", type=int, default=0)
parser.add_argument("-bbu", "--bsl_bound_upper", type=int, required=True)
parser.add_argument("-bpl", "--pks_bound_lower", type=int, required=True)
parser.add_argument("-bpu", "--pks_bound_upper", type=int, default=None)
parser.add_argument("-c", "--channel", type=int, required=True)
parser.add_argument("-v", "--verbose", nargs="?", const=True, default=True, type=str2bool)
parser.add_argument("-tr", "--trim_outliers_bool", nargs="?", const=True, default=True, type=str2bool)
args = parser.parse_args()


def _to_builtin(x):
    if x is None:
        return None
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, (bool, int, float, str)):
        if isinstance(x, float) and not np.isfinite(x):
            return None
        return x
    if isinstance(x, np.ndarray):
        return [_to_builtin(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [_to_builtin(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_builtin(v) for k, v in x.items()}
    try:
        return str(x)
    except Exception:
        return None


def compute():
    data_on = ADCRawData(args.input_path_on, verbose=args.verbose).get_branch_data(args.channel)
    data_off = ADCRawData(args.input_path_off, verbose=args.verbose).get_branch_data(args.channel)

    bsl_bounds = (args.bsl_bound_lower, args.bsl_bound_upper)
    pks_bounds = (args.pks_bound_lower, args.pks_bound_upper)

    fw = FixedWindow(bsl_bounds, pks_bounds)
    areas_on = np.array(fw.get_area(data_on), dtype=float)
    areas_off = np.array(fw.get_area(data_off), dtype=float)

    gain_model = GainModelIndependent(
        areas_on,
        areas_off,
        verbose=args.verbose,
        trim_outliers_bool=args.trim_outliers_bool,
    )

    adc_to_e = ADCRawData(args.input_path_on).adc_area_to_e

    try:
        estimates = gain_model.compute(gain_model.areas_led_on,
        gain_model.areas_led_off,
        adc_to_e)
    except ValueError as e:
        if args.verbose:
            print(f"ERROR: model-independent compute failed: {e}")
        estimates = {
            "mi_success": False,
            "mi_error": str(e),
            "gain": None,
            "occupancy": None,
            "threshold_occupancy_determination": None,
            "iterations": {},
        }

    return estimates, areas_on, areas_off

def gauss(x, A, mu, sigma):
    sigma = max(float(sigma), 1e-12)
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def emg_peak_scaled(x, A, mu, sigma, tau):
    sigma = max(float(sigma), 1e-12)
    tau = max(float(tau), 1e-12)
    y = exponnorm.pdf(x, K=tau, loc=mu, scale=sigma)
    peak = float(np.nanmax(y)) if np.size(y) else 0.0
    if not np.isfinite(peak) or peak <= 0:
        return np.zeros_like(np.asarray(x, dtype=float))
    return A * y / peak


def npe_model_tail_poisson(x, Anoise, munoise, sigmanoise, w, tau, mu1, sigma1, A_pe, occ):
    w = float(np.clip(w, 0.0, 1.0))
    occ = max(float(occ), 1e-12)
    sigma1 = max(float(sigma1), 1e-12)
    dmu = mu1 - munoise

    y = (
        (1.0 - w) * gauss(x, Anoise, munoise, sigmanoise)
        + w * emg_peak_scaled(x, Anoise, munoise, sigmanoise, tau)
    )

    for i in range(1, N_MAX_PE + 1):
        Ai = A_pe * (occ ** i) / math.factorial(i)
        mui = munoise + i * dmu
        sigmai = np.sqrt(i) * sigma1
        y += gauss(x, Ai, mui, sigmai)

    return y


def fit_pedestal_from_off(x, counts_off, window_bins=15):
    if curve_fit is None:
        return None

    x = np.asarray(x, float)
    y = np.asarray(counts_off, float)
    if x.size < 10 or np.nanmax(y) <= 0:
        return None

    i0 = int(np.nanargmax(y))
    lo = max(0, i0 - window_bins)
    hi = min(len(x), i0 + window_bins + 1)
    xf = x[lo:hi]
    yf = y[lo:hi]
    m = np.isfinite(xf) & np.isfinite(yf) & (yf > 0)
    xf = xf[m]
    yf = yf[m]
    if xf.size < 6:
        return None

    A0 = float(np.max(yf))
    mu0 = float(xf[np.argmax(yf)])
    weights = yf / np.sum(yf)
    sig0 = float(np.sqrt(np.sum(weights * (xf - np.sum(weights * xf)) ** 2)))
    if not np.isfinite(sig0) or sig0 <= 0:
        sig0 = float((xf.max() - xf.min()) / 6.0) if xf.max() > xf.min() else 50.0

    try:
        sigma_y = np.sqrt(np.maximum(yf, 1.0))
        popt, _ = curve_fit(
            gauss,
            xf,
            yf,
            p0=(A0, mu0, sig0),
            bounds=([0.0, xf.min(), 1e-9], [np.inf, xf.max(), np.inf]),
            sigma=sigma_y,
            absolute_sigma=True,
            maxfev=50000,
        )
        A, mu, sig = map(float, popt)
        if not (np.isfinite(A) and np.isfinite(mu) and np.isfinite(sig) and sig > 0):
            return None
        return A, mu, sig
    except Exception:
        return None


def first_signal_peak_guess(xr, yr):
    if xr.size == 0:
        return None

    kernel = np.ones(7, dtype=float) / 7.0
    ys = np.convolve(yr, kernel, mode="same")
    if ys.size < 3 or np.nanmax(ys) <= 0:
        return float(xr[int(np.nanargmax(yr))])

    local_max = np.where((ys[1:-1] > ys[:-2]) & (ys[1:-1] >= ys[2:]))[0] + 1
    if local_max.size:
        min_height = max(5.0, 0.03 * float(np.nanmax(ys)))
        good = [idx for idx in local_max if ys[idx] >= min_height]
        if good:
            return float(xr[good[0]])

    return float(xr[int(np.nanargmax(ys))])


def fit_constrained_poisson_pe_tail(x, counts_on, ped_params, initial_dmu_adc=None):
    if curve_fit is None:
        return None

    x = np.asarray(x, float)
    y = np.asarray(counts_on, float)
    m = np.isfinite(x) & np.isfinite(y) & (y >= 0)
    x = x[m]
    y = y[m]
    if x.size < 50 or np.max(y) <= 0:
        return None

    Anoise0, munoise0, sigmanoise0 = ped_params
    Anoise0 = max(float(Anoise0), 1.0)
    sigmanoise0 = max(float(sigmanoise0), 1.0)

    ymax = float(np.max(y))
    keep = y > (1e-9 * ymax)
    if np.count_nonzero(keep) > 40:
        xf = x[keep]
        yf = y[keep]
    else:
        xf = x
        yf = y



    start = munoise0 + 5.0 * sigmanoise0
    stop = munoise0 + 9000.0
    region = (xf > start) & (xf < stop)
    if np.count_nonzero(region) < 10:
        return None

    xr = xf[region]
    yr = yf[region]

    if initial_dmu_adc is not None and np.isfinite(initial_dmu_adc) and initial_dmu_adc > 100:
        mu1_guess = munoise0 + float(initial_dmu_adc)
    else:
        guessed_peak = first_signal_peak_guess(xr, yr)
        if guessed_peak is None:
            return None
        mu1_guess = guessed_peak

    mu1_guess = max(float(mu1_guess), munoise0 + 200.0)
    dmu_guess = max(mu1_guess - munoise0, 200.0)
    sigma1_guess = max(0.25 * dmu_guess, 80.0)

    def amp_at(mu_guess):
        k = int(np.argmin(np.abs(xf - mu_guess)))
        return float(max(yf[k], 1.0))

    A1_guess = amp_at(munoise0 + dmu_guess)
    A2_guess = amp_at(munoise0 + 2.0 * dmu_guess)
    if A1_guess > 0 and A2_guess > 0:
        occ_guess = max(0.05, min(3.0, 2.0 * A2_guess / A1_guess))
    else:
        occ_guess = 0.2

    A_pe_guess = max(A1_guess / occ_guess, 1.0)
    w_guess = min(0.03, TAIL_W_MAX)
    tau_guess = min(max(0.55 * dmu_guess, 1.0), 3000.0)


    p0 = [
        Anoise0,
        munoise0,
        sigmanoise0,
        w_guess,
        tau_guess,
        mu1_guess,
        sigma1_guess,
        A_pe_guess,
        occ_guess,
    ]

    sigma1_upper = max(1.2 * dmu_guess, 2.0 * sigma1_guess, 200.0)
    lower = [
        0.5 * Anoise0,
        munoise0 - 2.5 * sigmanoise0,
        0.5 * sigmanoise0,
        0.0,
        1e-6,
        munoise0 + 100.0,
        30.0,
        0.0,
        0.001,
    ]
    upper = [
        max(1.5 * Anoise0, Anoise0 + 1.0),
        munoise0 + 2.5 * sigmanoise0,
        1.8 * sigmanoise0,
        TAIL_W_MAX,
        50.0,
        munoise0 + 10000.0,
        sigma1_upper,
        np.inf,
        5.0,
    ]

    p0 = np.minimum(np.maximum(np.asarray(p0, dtype=float), np.asarray(lower) + 1e-9), np.asarray(upper) - 1e-9)

    try:
        sigma_y = np.sqrt(np.maximum(yf, 1.0))
        popt, pcov = curve_fit(
            npe_model_tail_poisson,
            xf,
            yf,
            p0=p0,
            bounds=(lower, upper),
            sigma=sigma_y,
            absolute_sigma=True,
            maxfev=300000,
        )
        return popt, pcov
    except Exception as e:
        if args.verbose:
            print(f"WARNING: Poisson PE fit failed: {e}")
        return None
    

def popt_from_model_params(params):
    return [
        params["Anoise"],
        params["munoise"],
        params["sigmanoise"],
        params["w"],
        params["tau"],
        params["mu1"],
        params["sigma1"],
        params["A_pe"],
        params["occupancy_fit"],
    ]


def compute_chi_square_test(x, observed, popt, min_expected=CHI2_MIN_EXPECTED):
    x = np.asarray(x, dtype=float)
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(npe_model_tail_poisson(x, *popt), dtype=float)

    valid = (
        np.isfinite(x)
        & np.isfinite(observed)
        & np.isfinite(expected)
        & (observed >= 0)
        & (expected >= float(min_expected))
    )

    contribution = np.full_like(expected, np.nan, dtype=float)
    contribution[valid] = (observed[valid] - expected[valid]) ** 2 / expected[valid]
    used_bins = int(np.count_nonzero(valid))
    n_fit_params = int(len(popt))
    ndf = used_bins - n_fit_params

    if used_bins == 0:
        chi2_value = None
        reduced_chi2 = None
        p_value = None
    else:
        chi2_value = float(np.sum(contribution[valid]))
        reduced_chi2 = float(chi2_value / ndf) if ndf > 0 else None
        p_value = float(chi2.sf(chi2_value, ndf)) if ndf > 0 else None

    stats = {
        "test": "Pearson chi-squared goodness-of-fit",
        "chi2": chi2_value,
        "ndf": int(ndf),
        "reduced_chi2": reduced_chi2,
        "p_value": p_value,
        "used_bins": used_bins,
        "fit_parameters": n_fit_params,
        "min_expected_count": float(min_expected),
        "observed_sum_used_bins": float(np.sum(observed[valid])) if used_bins else None,
        "expected_sum_used_bins": float(np.sum(expected[valid])) if used_bins else None,
    }
    return stats, expected, contribution, valid


def sci_latex(value, uncertainty=None, unit="", ndigits=2):
    if value is None or not np.isfinite(value):
        return r"\mathrm{n/a}"
    value = float(value)
    if value == 0:
        return f"{0:.{ndigits}f}" + unit
    exponent = int(np.floor(np.log10(abs(value))))
    scale = 10.0 ** exponent
    mantissa = value / scale
    if uncertainty is not None and np.isfinite(uncertainty) and uncertainty > 0:
        uncertainty_mantissa = float(uncertainty) / scale
        return (
            rf"({mantissa:.{ndigits}f} \pm "
            rf"{uncertainty_mantissa:.{ndigits}f})\times 10^{{{exponent}}}{unit}"
        )
    return rf"{mantissa:.{ndigits}f}\times 10^{{{exponent}}}{unit}"


def decimal_latex(value, decimals=2, unit=""):
    if value is None or not np.isfinite(value):
        return r"\mathrm{n/a}"
    return f"{float(value):.{decimals}f}" + unit


def p_value_latex(value):
    if value is None or not np.isfinite(value):
        return r"\mathrm{n/a}"
    if value < 1e-3:
        return sci_latex(value, ndigits=2)
    return f"{value:.3f}"


def add_scientific_summary(ax, text):
    ax.text(
        0.00,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14.0,
        linespacing=1.28,
    )
    ax.axvline(0.0, color="0.70", linewidth=0.8)


def area_to_signal_me(area, pedestal_mu, adc_to_e):
    return (np.asarray(area, dtype=float) - float(pedestal_mu)) * float(adc_to_e) / 1e6


def automatic_high_edge_area(centers, counts_on, occupied_hi_area):
    """
    Find where the useful high-x histogram ends.

    Two automatic limits are combined:
      1. a cumulative-count limit that drops only a statistically tiny tail,
      2. a local-density limit that rejects long stretches of isolated bins.
    """
    centers = np.asarray(centers, dtype=float)
    counts_on = np.asarray(counts_on, dtype=float)

    finite = np.isfinite(centers) & np.isfinite(counts_on) & (counts_on > 0)
    if not np.any(finite):
        return float(occupied_hi_area)

    ordered = np.argsort(centers[finite])
    x = centers[finite][ordered]
    y = counts_on[finite][ordered]
    total = float(np.sum(y))

    if total <= 0:
        return float(occupied_hi_area)


    tail_entries_to_drop = min(max(5.0, np.sqrt(total)), 0.002 * total)
    keep_count = max(total - tail_entries_to_drop, 1.0)
    cumulative = np.cumsum(y)
    count_idx = int(np.searchsorted(cumulative, keep_count))
    count_hi = float(x[min(count_idx, x.size - 1)])

    dense_hi = count_hi
    all_finite = np.isfinite(centers) & np.isfinite(counts_on)
    if np.count_nonzero(all_finite) >= 3:
        x_all = centers[all_finite]
        y_all = counts_on[all_finite]
        order_all = np.argsort(x_all)
        x_all = x_all[order_all]
        y_all = y_all[order_all]

        window = min(X_AXIS_DENSITY_WINDOW_BINS, max(3, y_all.size // 8))
        kernel = np.ones(window, dtype=float)
        local_sum = np.convolve(y_all, kernel, mode="same")
        density_threshold = max(3.0, 2.5 * np.sqrt(max(total / max(y_all.size, 1), 1.0)))
        dense = local_sum >= density_threshold
        if np.any(dense):
            dense_hi = float(x_all[np.where(dense)[0][-1]])

    return min(float(occupied_hi_area), max(count_hi, dense_hi))


def choose_signal_window_me(centers, counts_on, counts_off, pedestal_mu, adc_to_e, model_dep=None):
    """
    Choose a tight display range in signal-electron units.

    This is only a plotting choice. The low edge keeps the full beginning of
    the occupied histogram; the high edge may cut only a few far-tail entries.
    """
    centers = np.asarray(centers, dtype=float)
    counts_on = np.asarray(counts_on, dtype=float)
    counts_off = np.asarray(counts_off, dtype=float)

    occupied = (
        np.isfinite(centers)
        & np.isfinite(counts_on)
        & np.isfinite(counts_off)
        & ((counts_on > 0) | (counts_off > 0))
    )

    if np.any(occupied):
        occupied_lo_area = float(np.nanmin(centers[occupied]))
        occupied_hi_area = float(np.nanmax(centers[occupied]))
    else:
        finite_centers = centers[np.isfinite(centers)]
        if finite_centers.size == 0:
            return None
        occupied_lo_area = float(np.nanmin(finite_centers))
        occupied_hi_area = float(np.nanmax(finite_centers))

    lo_area = occupied_lo_area
    hi_area = automatic_high_edge_area(centers, counts_on, occupied_hi_area)

    if model_dep is not None and model_dep.get("fit_success", False):
        p = model_dep["params"]
        dmu = float(p["dmu_adc"])
        mu0 = float(p["munoise"])
        sigma0 = float(p["sigmanoise"])
        sigma1 = float(p["sigma1"])
        a1 = float(p.get("A1", 1.0))
        shown_pe = [
            i for i in range(1, N_MAX_PE + 1)
            if float(p.get(f"A{i}", 0.0)) >= PE_PLOT_MIN_FRAC * max(a1, 1.0)
        ]
        last_pe = max(shown_pe) if shown_pe else 1
        model_lo = mu0 - 3.5 * sigma0
        model_hi = (
            mu0
            + (last_pe + X_AXIS_MODEL_EXTRA_PE) * dmu
            + X_AXIS_MODEL_SIGMA_MARGIN * np.sqrt(last_pe) * sigma1
        )
        lo_area = min(lo_area, model_lo)
        min_signal_hi = mu0 + 1.2 * dmu + 3.0 * sigma1
        hi_area = max(hi_area, min_signal_hi)
        hi_area = min(hi_area, max(model_hi, min_signal_hi))

    span = max(hi_area - lo_area, 1.0)
    lo_area -= 0.035 * span
    hi_area += 0.180 * span

    lo = float(area_to_signal_me(lo_area, pedestal_mu, adc_to_e))
    hi = float(area_to_signal_me(hi_area, pedestal_mu, adc_to_e))
    return (lo, hi) if hi > lo else None


def plot_entries_and_precision(
    areas_on,
    areas_off,
    threshold,
    out_dir,
    basename,
    pmt_name,
    adc_to_e,
    gain_independent=None,
    gain_independent_error=None,
):

    lamp_v = "unknown"
    hv_v = "unknown"
    name = basename.replace("-", "_")

    m_lamp = re.search(r"Lamp_?([0-9]+(?:\.[0-9]+)?)_?Vpp", name, re.IGNORECASE)
    if m_lamp:
        lamp_v = m_lamp.group(1) + "Vpp"

    m_hv = re.search(r"(?<![A-Za-z0-9.])(-?\d{3,4})V(?!pp)", basename, re.IGNORECASE)
    if not m_hv:
        m_hv = re.search(r"(?<![A-Za-z0-9.])(-?\d{3,4})V(?!pp)", name, re.IGNORECASE)
    if m_hv:
        hv_v = m_hv.group(1) + "V"

    m = re.search(r"(LV\d{4})", os.path.basename(args.input_path_on))
    if m:
        pmt_serial = m.group(1)
    else:
        m2 = re.search(r"(LV\d{4})", os.path.basename(os.path.dirname(args.input_path_on)))
        pmt_serial = m2.group(1) if m2 else pmt_name

    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(15.5, 9.2))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[3.4, 1.15],
        height_ratios=[3.2, 1.15],
        left=0.07,
        right=0.98,
        bottom=0.08,
        top=0.92,
        wspace=0.14,
        hspace=0.08,
    )

    ax = fig.add_subplot(gs[0, 0])
    ax_res = fig.add_subplot(gs[1, 0], sharex=ax)
    ax_summary = fig.add_subplot(gs[:, 1])
    ax_summary.axis("off")

    all_areas = np.concatenate([areas_off, areas_on]).astype(float)
    a_min = float(np.min(all_areas))
    a_max_data = float(np.percentile(all_areas, 99.9999))
    a_max = min(25000.0, a_max_data)
    if a_max <= a_min + 50:
        a_max = min(25000.0, float(np.max(all_areas)))

    bin_width = 10.0
    bin_edges = np.arange(a_min, a_max + bin_width, bin_width)
    counts_off, _ = np.histogram(areas_off, bins=bin_edges)
    counts_on, _ = np.histogram(areas_on, bins=bin_edges)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    lamp_num = lamp_v.replace("Vpp", "").strip() if lamp_v != "unknown" else "?"

    h_off = ax.step(
        bin_edges[:-1],
        counts_off,
        where="post",
        linewidth=1.2,
        color="darkred",
        label=r"$\mathrm{LED\ OFF}$",
    )[0]
    h_on = ax.step(
        bin_edges[:-1],
        counts_on,
        where="post",
        linewidth=1.2,
        color="royalblue",
        label=rf"$\mathrm{{LED\ ON}}\ (V_{{\mathrm{{Lamp}}}}={lamp_num}\,\mathrm{{V_{{pp}}}})$",
    )[0]

    positive_counts = np.concatenate([counts_off[counts_off > 0], counts_on[counts_on > 0]])
    ymin = float(np.min(positive_counts)) if positive_counts.size else 1.0
    ymax = float(max(np.max(counts_off), np.max(counts_on), 1.0))

    model_dep = None
    legend_handles = [h_off, h_on]
    legend_labels = [
        r"$\mathrm{LED\ OFF}$",
        rf"$\mathrm{{LED\ ON}}\ (V_{{\mathrm{{Lamp}}}}={lamp_num}\,\mathrm{{V_{{pp}}}})$",
    ]
    summary_text = (
        r"$\mathbf{PMT\ gain\ and\ fit\ summary}$" "\n"
        r"$G_{\mathrm{MD}} = \mathrm{n/a}$" "\n"
        r"$G_{\mathrm{MI}} = \mathrm{n/a}$" "\n"
        r"$\chi^2/\mathrm{ndf} = \mathrm{n/a}$" "\n"
        r"$p_{\chi^2} = \mathrm{n/a}$"
    )

    ped = fit_pedestal_from_off(centers, counts_off, window_bins=15)
    xlim = None

    if ped is not None:
        Anoise, munoise, sigmanoise = ped
        xfine = np.linspace(centers.min(), centers.max(), 3000)
        x_hist_me = area_to_signal_me(bin_edges[:-1], munoise, adc_to_e)
        x_centers_me = area_to_signal_me(centers, munoise, adc_to_e)
        xfine_me = area_to_signal_me(xfine, munoise, adc_to_e)
        h_off.set_xdata(x_hist_me)
        h_on.set_xdata(x_hist_me)

        h_ped = ax.plot(
            xfine_me,
            gauss(xfine, Anoise, munoise, sigmanoise),
            "--",
            linewidth=1.3,
            color="firebrick",
            label=r"$\mathrm{Pedestal\ fit}$",
        )[0]
        legend_handles.append(h_ped)
        legend_labels.append(r"$\mathrm{Pedestal\ fit}$")

        initial_dmu_adc = None
        if gain_independent is not None and np.isfinite(gain_independent) and adc_to_e > 0:
            initial_dmu_adc = float(gain_independent) / float(adc_to_e)

        fit = fit_constrained_poisson_pe_tail(
            centers,
            counts_on,
            ped,
            initial_dmu_adc=initial_dmu_adc,
        )

        if fit is not None:
            popt, pcov = fit
            (
                Anoise_f,
                munoise_f,
                sigmanoise_f,
                w,
                tau,
                mu1,
                sigma1,
                A_pe,
                occ,
            ) = map(float, popt)

            x_hist_me = area_to_signal_me(bin_edges[:-1], munoise_f, adc_to_e)
            x_centers_me = area_to_signal_me(centers, munoise_f, adc_to_e)
            xfine_me = area_to_signal_me(xfine, munoise_f, adc_to_e)
            h_off.set_xdata(x_hist_me)
            h_on.set_xdata(x_hist_me)
            h_ped.set_xdata(xfine_me)

            dmu = mu1 - munoise_f
            gain_adc = float(dmu)
            gain_e = float(gain_adc * float(adc_to_e))

            amps = [A_pe * (occ ** i) / math.factorial(i) for i in range(1, N_MAX_PE + 1)]
            gain_adc_err = None
            gain_e_err = None
            MU1_INDEX = 5
            MUNOISE_INDEX = 1
            if pcov is not None and pcov.shape == (len(popt), len(popt)) and np.all(np.isfinite(pcov)):
                var = float(
                    pcov[MU1_INDEX, MU1_INDEX]
                    + pcov[MUNOISE_INDEX, MUNOISE_INDEX]
                    - 2.0 * pcov[MU1_INDEX, MUNOISE_INDEX]
                )
                if var >= 0 and np.isfinite(var):
                    gain_adc_err = float(np.sqrt(var))
                    gain_e_err = float(gain_adc_err * float(adc_to_e))

            rel_width = sigma1 / gain_adc if gain_adc > 0 else np.inf
            chi_square_stats, _, _, _ = compute_chi_square_test(centers, counts_on, popt)
            ytot = npe_model_tail_poisson(xfine, *popt)

            h_tot = ax.plot(
                xfine_me,
                ytot,
                "-",
                linewidth=1.5,
                color="navy",
                label=rf"$0$-${N_MAX_PE}\,\mathrm{{PE}}\ \mathrm{{model}}$",
            )[0]
            h_gped = ax.plot(
                xfine_me,
                (1.0 - w) * gauss(xfine, Anoise_f, munoise_f, sigmanoise_f),
                "--",
                linewidth=1.3,
                color="indianred",
                label=r"$\mathrm{Gaussian\ pedestal}$",
            )[0]
            h_tail = ax.plot(
                xfine_me,
                w * emg_peak_scaled(xfine, Anoise_f, munoise_f, sigmanoise_f, tau),
                "--",
                linewidth=1.3,
                color="orange",
                label=r"$\mathrm{EMG\ tail}$",
            )[0]

            pe_handles = []
            pe_labels = []
            A1_ref = amps[0] if len(amps) > 0 and amps[0] > 0 else 1.0
            pe_colors = [
                "dodgerblue",
                "deepskyblue",
                "steelblue",
                "cornflowerblue",
                "lightskyblue",
                "slateblue",
                "mediumblue",
                "darkturquoise",
            ]
            for i, Ai in enumerate(amps, start=1):
                if Ai < PE_PLOT_MIN_FRAC * A1_ref:
                    continue
                mui = munoise_f + i * dmu
                sigmai = np.sqrt(i) * sigma1
                h = ax.plot(
                    xfine_me,
                    gauss(xfine, Ai, mui, sigmai),
                    "--",
                    linewidth=1.3,
                    color=pe_colors[(i - 1) % len(pe_colors)],
                    label=rf"${i}\,\mathrm{{PE}}$",
                )[0]
                pe_handles.append(h)
                pe_labels.append(rf"${i}\,\mathrm{{PE}}$")

            legend_handles.extend([h_tot, h_gped, h_tail, *pe_handles])
            legend_labels.extend([
                rf"$0$-${N_MAX_PE}\,\mathrm{{PE}}\ \mathrm{{model}}$",
                r"$\mathrm{Gaussian\ pedestal}$",
                r"$\mathrm{EMG\ tail}$",
                *pe_labels,
            ])

            params = {
                "Anoise": Anoise_f,
                "munoise": munoise_f,
                "sigmanoise": sigmanoise_f,
                "w": w,
                "tau": tau,
                "mu1": mu1,
                "sigma1": sigma1,
                "A_pe": A_pe,
                "occupancy_fit": occ,
                "dmu_adc": gain_adc,
                "dmu_adc_err": gain_adc_err,
                "gain_e": gain_e,
                "gain_e_err": gain_e_err,
                "rel_width": rel_width,
                "n_max_pe": N_MAX_PE,
                "pe_plot_min_frac": PE_PLOT_MIN_FRAC,
            }
            for i, Ai in enumerate(amps, start=1):
                params[f"A{i}"] = Ai
                params[f"mu{i}"] = munoise_f + i * dmu
                params[f"sigma{i}"] = np.sqrt(i) * sigma1

            model_dep = {
                "fit_success": True,
                "params": params,
                "covariance": pcov,
                "chi_square": chi_square_stats,
            }

            fit_counts = npe_model_tail_poisson(centers, *popt)
            sigma = np.sqrt(np.maximum(fit_counts, 1.0))
            residual_sigma = (counts_on - fit_counts) / sigma
            residual_mask = (
                np.isfinite(residual_sigma)
                & np.isfinite(x_centers_me)
                & (counts_on >= 1)
                & (fit_counts >= 1)
            )
            if np.any(residual_mask):
                y_res = residual_sigma[residual_mask]
                frac_gt3 = float(np.mean(np.abs(y_res) > 3.0))
                rms_pull = float(np.sqrt(np.mean(y_res ** 2)))
                ax_res.axhspan(-1.0, 1.0, color="green", alpha=0.2, label=r"$\pm 1\sigma$")
                ax_res.axhspan(-2.0, -1.0, color="gold", alpha=0.3, label=r"$\pm2\sigma$")
                ax_res.axhspan(1.0, 2.0, color="gold", alpha=0.3)
                ax_res.scatter(
                    x_centers_me[residual_mask],
                    y_res,
                    s=10,
                    color="black",
                    alpha=0.75,
                    label="Residuals",
                )
                ax_res.axhline(0, color="green", linestyle="--", linewidth=1.5, label=r"$0\sigma$")
                ax_res.axhline(2, color="gold", linestyle="--", linewidth=1.2, label=r"$\pm2\sigma$")
                ax_res.axhline(-2, color="gold", linestyle="--", linewidth=1.2)
                ax_res.axhline(3, color="red", linestyle="--", linewidth=1.2, label=r"$\pm3\sigma$")
                ax_res.axhline(-3, color="red", linestyle="--", linewidth=1.2)
                ax_res.set_title(
                     rf"$\mathrm{{RMS}}={rms_pull:.2f}$, "
                    rf"$f_{{|r|>3\sigma}}={frac_gt3:.1%}$",
                    fontsize=12,
                )
                ax_res.legend(loc="upper right", fontsize=12)
            else:
                ax_res.text(0.5, 0.5, "No residual bins passed the display cuts.", transform=ax_res.transAxes, ha="center", va="center")

            electron_unit = r"\,e^{-}"
            adc_unit = r"\,\mathrm{ADC}"
            md_line = rf"$G_{{\mathrm{{MD}}}} = {sci_latex(gain_e, gain_e_err, unit=electron_unit)}$"
            if gain_independent is not None and np.isfinite(gain_independent):
                mi_line = (
                    rf"$G_{{\mathrm{{MI}}}} = "
                    rf"{sci_latex(gain_independent, gain_independent_error, unit=electron_unit)}$"
                )
            else:
                mi_line = r"$G_{\mathrm{MI}} = \mathrm{n/a}$"

            shown_pe = [i for i, Ai in enumerate(amps, start=1) if Ai >= PE_PLOT_MIN_FRAC * A1_ref]
            shown_pe_text = ", ".join(str(i) for i in shown_pe) if shown_pe else r"\mathrm{none}"
            if chi_square_stats["chi2"] is not None and chi_square_stats["reduced_chi2"] is not None:
                chi_line = (
                    rf"$\chi^2/\nu = {chi_square_stats['chi2']:.1f}/"
                    rf"{chi_square_stats['ndf']} = {chi_square_stats['reduced_chi2']:.2f}$"
                )
                p_line = rf"$p(\chi^2,\nu) = {p_value_latex(chi_square_stats['p_value'])}$"
            else:
                chi_line = r"$\chi^2/\nu = \mathrm{n/a}$"
                p_line = r"$p(\chi^2,\nu) = \mathrm{n/a}$"

            hv_tex = r"\mathrm{unknown}" if hv_v == "unknown" else rf"{hv_v.replace('V', '')}\,\mathrm{{V}}"
            lamp_tex = r"\mathrm{unknown}" if lamp_num == "?" else rf"{lamp_num}\,\mathrm{{V_{{pp}}}}"
            summary_text = (
                r"$\mathbf{PMT\ gain\ and\ fit\ summary}$" "\n"
                rf"$\mathrm{{PMT}} = \mathrm{{{pmt_serial}}}$" "\n"
                rf"$V_{{\mathrm{{HV}}}} = {hv_tex}$" "\n"
                rf"$V_{{\mathrm{{lamp}}}} = {lamp_tex}$" "\n"
                "\n"
                r"$\mathbf{Gain}$" "\n"
                f"{md_line}\n"
                f"{mi_line}\n"
                rf"$\Delta\mu = {decimal_latex(gain_adc, 1, unit=adc_unit)}$" "\n"
                "\n"
                r"$\mathbf{Goodness\ of\ fit}$" "\n"
                f"{chi_line}\n"
                f"{p_line}\n"
                rf"$N_{{\mathrm{{bins}}}} = {chi_square_stats['used_bins']},\ "
                rf"N_{{\mathrm{{par}}}} = {chi_square_stats['fit_parameters']}$" "\n"
                "\n"
                r"$\mathbf{Model\ parameters}$" "\n"
                rf"$A_0 = {sci_latex(Anoise_f)}$" "\n"
                rf"$\mu_0 = {decimal_latex(munoise_f, 1, unit=adc_unit)}$" "\n"
                rf"$\sigma_0 = {decimal_latex(sigmanoise_f, 1, unit=adc_unit)}$" "\n"
                rf"$f_{{\mathrm{{tail}}}} = {w:.3f},\ "
                rf"K_{{\mathrm{{tail}}}} = {tau:.2f}$" "\n"
                rf"$\mu_1 = {decimal_latex(mu1, 1, unit=adc_unit)}$" "\n"
                rf"$\sigma_1 = {decimal_latex(sigma1, 1, unit=adc_unit)}$" "\n"
                rf"$\sigma_1/\Delta\mu = {rel_width:.3f}$" "\n"
                rf"$\lambda_{{\mathrm{{fit}}}} = {occ:.3f}$" "\n"
                rf"$\mathrm{{PE\ terms}} = {shown_pe_text}$"
            )
            xlim = choose_signal_window_me(centers, counts_on, counts_off, munoise_f, adc_to_e, model_dep=model_dep)
        else:
            xlim = choose_signal_window_me(centers, counts_on, counts_off, munoise, adc_to_e)
            ax_res.text(0.5, 0.5, "Model-dependent fit failed; residuals unavailable.", transform=ax_res.transAxes, ha="center", va="center")
    else:
        x_hist_me = area_to_signal_me(bin_edges[:-1], 0.0, adc_to_e)
        h_off.set_xdata(x_hist_me)
        h_on.set_xdata(x_hist_me)
        ax_res.text(0.5, 0.5, "Pedestal fit failed; residuals unavailable.", transform=ax_res.transAxes, ha="center", va="center")

    if threshold is not None:
        if ped is not None and model_dep is not None and model_dep.get("fit_success", False):
            threshold_x = area_to_signal_me(threshold, model_dep["params"]["munoise"], adc_to_e)
        elif ped is not None:
            threshold_x = area_to_signal_me(threshold, ped[1], adc_to_e)
        else:
            threshold_x = threshold * adc_to_e / 1e6
        h_thr = ax.axvline(threshold_x, linestyle="--", linewidth=1.5, color="gray")
        legend_handles.append(h_thr)
        legend_labels.append(r"$A_{\mathrm{thr}}$")

    if xlim is not None and np.all(np.isfinite(xlim)) and xlim[1] > xlim[0]:
        ax.set_xlim(*xlim)

    ax.set_ylabel(r"Entries")
    ax.set_title(f"{pmt_serial} - HV {hv_v}: Distribution of integrated ADC area")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax * 2.0)
    plt.setp(ax.get_xticklabels(), visible=False)

    ax_res.set_xlabel(r"Signal electrons $[(A-\mu_0)\,c_{\mathrm{ADC}\to e}/10^6]$")
    ax_res.set_ylabel(r"Pull [$\sigma$]")
    ax_res.set_ylim(-3.5, 3.5)
    ax_res.grid(True, alpha=0.25)

    add_scientific_summary(ax_summary, summary_text)
    legend = ax.legend(
        legend_handles,
        legend_labels,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        fontsize=14.0,
        frameon=True,
        framealpha=0.92,
        edgecolor="0.45",
        borderpad=0.40,
        labelspacing=0.36,
        handlelength=2.6,
        handletextpad=0.65,
    )
    legend.get_frame().set_linewidth(0.8)

    plot_path = os.path.join(out_dir, f"{basename}_entries_and_precision.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    if args.verbose:
        print(f"Saved combined entries/residual precision plot: {plot_path}")
    plt.close(fig)

    hist_dict = {
        "bin_edges": bin_edges,
        "counts_off": counts_off,
        "counts_on": counts_on,
        "bin_width": bin_width,
    }
    return hist_dict, model_dep


if __name__ == "__main__":
    estimates, areas_on, areas_off = compute()

    m = re.search(r"(LV\d{4})", os.path.basename(args.input_path_on))
    if m:
        pmt_name = m.group(1)
    else:
        pmt_name = os.path.basename(os.path.dirname(args.input_path_on))

    out_dir = os.path.join(BASE_OUTPUT_DIR, pmt_name)
    os.makedirs(out_dir, exist_ok=True)

    basename = os.path.basename(args.input_path_on).replace(".root", "")
    adc_to_e = ADCRawData(args.input_path_on).adc_area_to_e
    threshold = estimates.get("threshold_occupancy_determination", None)

    hist_data, model_dep = plot_entries_and_precision(
        areas_on,
        areas_off,
        threshold,
        out_dir,
        basename,
        pmt_name,
        adc_to_e,
        gain_independent=estimates.get("gain"),
        gain_independent_error=estimates.get("gain_err"),
    )


    if model_dep is None:
        estimates["model_dependent_fit"] = {
            "fit_success": False,
            "reason": "fit not performed or failed",
        }
    else:
        estimates["model_dependent_fit"] = _to_builtin(model_dep)

    cleaned_estimates = _to_builtin(estimates)
    results_json = os.path.join(out_dir, f"{basename}_results.json")
    fd_res, tmp_res = tempfile.mkstemp(prefix=".tmp_results_", dir=out_dir)
    try:
        with os.fdopen(fd_res, "w") as f:
            json.dump(cleaned_estimates, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_res, results_json)
        if args.verbose:
            print(f"Saved results JSON: {results_json}")
    finally:
        try:
            if os.path.exists(tmp_res):
                os.remove(tmp_res)
        except Exception:
            pass

    cleaned_hist = _to_builtin(hist_data)
    hist_json = os.path.join(out_dir, f"{basename}_hist_entries.json")
    fd_hist, tmp_hist = tempfile.mkstemp(prefix=".tmp_hist_", dir=out_dir)
    try:
        with os.fdopen(fd_hist, "w") as f:
            json.dump(cleaned_hist, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_hist, hist_json)
        if args.verbose:
            print(f"Saved histogram JSON: {hist_json}")
    finally:
        try:
            if os.path.exists(tmp_hist):
                os.remove(tmp_hist)
        except Exception:
            pass

    if args.verbose:
        print("Done.")
