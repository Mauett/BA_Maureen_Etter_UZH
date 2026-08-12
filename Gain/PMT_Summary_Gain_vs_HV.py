#!/usr/bin/env python3
"""
Plot PMT gain versus high voltage from the *_results.json files produced by
run_tails_coupled.py / run_tails_coupled_robust.py.

The script scans:
  BASE_OUTPUT_DIR/<PMT>/*_results.json

and saves:
  BASE_OUTPUT_DIR/pmt_gain_vs_high_voltage.png
  BASE_OUTPUT_DIR/pmt_gain_vs_high_voltage.csv
  BASE_OUTPUT_DIR/pmt_gain_vs_high_voltage_fits.csv

For each PMT, the gain-voltage dependence is fitted with the usual power law

  G(V) = A * V**k

by performing a linear fit in log-log space.
"""

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

BASE_OUTPUT_DIR = Path(__file__).resolve().parent / "Gain_Results"
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PLOT = BASE_OUTPUT_DIR / "pmt_gain_vs_high_voltage_fits.png"
OUTPUT_CSV = BASE_OUTPUT_DIR / "pmt_gain_vs_high_voltage_fits.csv"
OUTPUT_FIT_CSV = BASE_OUTPUT_DIR / "pmt_gain_vs_high_voltage_fits_fits.csv"

PMTS = ["LV2480", "LV2481", "LV2483", "LV2485"]

# Use the same target lamp settings as your analysis runner.
# Set a PMT value to None if you want to use all lamp voltages for that PMT.
TARGET_LAMP_PER_PMT = {
    "LV2480": 1.90,
    "LV2481": 1.95,
    "LV2483": 1.95,
    "LV2485": 1.95,
}

# Choose "model_dependent" or "model_independent".
GAIN_SOURCE = "model_dependent"

# Optional filters. Set to None to disable.
HVS_TO_PLOT = None
# HVS_TO_PLOT = [700, 750, 800, 850, 900]

# Legend will show the gain at this HV when available.
LEGEND_REFERENCE_HV = 800

USE_ABS_HV = True
LAMP_TOL = 1e-3

DO_FITS = True
USE_GAIN_ERRORS_FOR_FIT = True
FIT_N_POINTS = 250
FIT_LINESTYLE = "--"
FIT_LINEWIDTH = 1.6

PMT_COLORS = {
    "LV2480": "mediumseagreen",
    "LV2481": "dodgerblue",
    "LV2483": "slateblue",
    "LV2485": "purple",
}


# ------------------------------------------------------------------
# Filename helpers
# ------------------------------------------------------------------

_PMT_RE = re.compile(r"(LV\d{4})")
_LAMP_RE = re.compile(r"Lamp_?([0-9]+(?:\.[0-9]+)?)_?Vpp", re.IGNORECASE)
_HV_RE = re.compile(r"(?<![A-Za-z0-9.])(-?\d{3,4})V(?!pp)", re.IGNORECASE)


def normalized_name(name):
    return name.replace("-", "_")


def extract_pmt(path):
    m = _PMT_RE.search(path.name)
    if m:
        return m.group(1)

    m = _PMT_RE.search(str(path.parent))
    return m.group(1) if m else None


def extract_lamp_vpp(path):
    m = _LAMP_RE.search(normalized_name(path.name))
    if not m:
        return None

    try:
        return float(m.group(1))
    except ValueError:
        return None


def extract_hv(path):
    matches = _HV_RE.findall(path.name)
    if not matches:
        matches = _HV_RE.findall(normalized_name(path.name))

    if not matches:
        return None

    try:
        hv = int(matches[-1])
    except ValueError:
        return None

    return abs(hv) if USE_ABS_HV else hv


# ------------------------------------------------------------------
# JSON helpers
# ------------------------------------------------------------------

def finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    return value if math.isfinite(value) else None


def gain_from_result(data):
    if GAIN_SOURCE == "model_independent":
        gain = finite_float(data.get("gain"))
        gain_err = (
            finite_float(data.get("gain_err"))
            or finite_float(data.get("gain_error"))
            or finite_float(data.get("gain_std"))
        )
        return gain, gain_err

    if GAIN_SOURCE != "model_dependent":
        raise ValueError("GAIN_SOURCE must be 'model_dependent' or 'model_independent'")

    md = data.get("model_dependent_fit", {})
    if not md.get("fit_success", False):
        return None, None

    params = md.get("params", {})
    gain = finite_float(params.get("gain_e"))
    gain_err = finite_float(params.get("gain_e_err"))
    return gain, gain_err


def include_result(path, pmt, hv, lamp):
    if pmt not in PMTS:
        return False

    if hv is None:
        return False

    if HVS_TO_PLOT is not None and hv not in HVS_TO_PLOT:
        return False

    target_lamp = TARGET_LAMP_PER_PMT.get(pmt)
    if target_lamp is not None:
        if lamp is None or abs(lamp - target_lamp) > LAMP_TOL:
            return False

    return True


def combine_records(records):
    """
    Combine duplicate measurements for the same PMT/HV.
    """
    gains = np.array([r["gain"] for r in records], dtype=float)
    errs = np.array([
        r["gain_err"] if r["gain_err"] is not None and r["gain_err"] > 0 else np.nan
        for r in records
    ], dtype=float)

    if gains.size == 1:
        return float(gains[0]), None if not np.isfinite(errs[0]) else float(errs[0])

    good_err = np.isfinite(errs) & (errs > 0)

    if np.all(good_err):
        weights = 1.0 / errs**2
        gain = float(np.sum(weights * gains) / np.sum(weights))
        gain_err = float(np.sqrt(1.0 / np.sum(weights)))
        return gain, gain_err

    gain = float(np.median(gains))
    gain_err = float(np.std(gains, ddof=1) / np.sqrt(gains.size)) if gains.size > 1 else None
    return gain, gain_err


def load_gain_points():
    grouped = defaultdict(list)

    for path in sorted(BASE_OUTPUT_DIR.glob("LV*/*_results.json")):
        pmt = extract_pmt(path)
        hv = extract_hv(path)
        lamp = extract_lamp_vpp(path)

        if not include_result(path, pmt, hv, lamp):
            continue

        with open(path, "r") as f:
            data = json.load(f)

        gain, gain_err = gain_from_result(data)
        if gain is None or gain <= 0:
            continue

        grouped[(pmt, hv)].append({
            "pmt": pmt,
            "hv": hv,
            "lamp": lamp,
            "gain": gain,
            "gain_err": gain_err,
            "path": path,
        })

    points = defaultdict(list)
    rows = []

    for (pmt, hv), records in sorted(grouped.items()):
        gain, gain_err = combine_records(records)
        lamp_values = sorted({
            round(r["lamp"], 6) for r in records
            if r["lamp"] is not None
        })

        point = {
            "pmt": pmt,
            "hv": hv,
            "gain": gain,
            "gain_err": gain_err,
            "lamp_values": lamp_values,
            "n_files": len(records),
        }
        points[pmt].append(point)
        rows.append(point)

    return points, rows


def write_csv(rows):
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pmt", "hv", "gain", "gain_err", "lamp_values", "n_files"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fit_gain_power_law(pmt, points):
    """
    Fit G(V) = A * V**k using ln(G) = ln(A) + k ln(V).
    """
    if len(points) < 2:
        return None

    hv = np.array([p["hv"] for p in points], dtype=float)
    gain = np.array([p["gain"] for p in points], dtype=float)
    gain_err = np.array([
        p["gain_err"] if p["gain_err"] is not None else np.nan
        for p in points
    ], dtype=float)

    valid = np.isfinite(hv) & np.isfinite(gain) & (hv > 0) & (gain > 0)
    hv = hv[valid]
    gain = gain[valid]
    gain_err = gain_err[valid]

    if hv.size < 2:
        return None

    x = np.log(hv)
    y = np.log(gain)

    sigma_y = np.full_like(y, np.nan, dtype=float)
    good_gain_err = np.isfinite(gain_err) & (gain_err > 0)
    sigma_y[good_gain_err] = gain_err[good_gain_err] / gain[good_gain_err]

    use_weights = (
        USE_GAIN_ERRORS_FOR_FIT
        and np.all(np.isfinite(sigma_y))
        and np.all(sigma_y > 0)
    )

    if use_weights:
        weights = 1.0 / sigma_y**2
    else:
        weights = np.ones_like(y)

    design = np.column_stack([np.ones_like(x), x])
    weighted_design = design * weights[:, None]
    normal_matrix = design.T @ weighted_design
    covariance_base = np.linalg.inv(normal_matrix)
    beta = covariance_base @ (design.T @ (weights * y))

    log_a = float(beta[0])
    exponent = float(beta[1])

    y_fit = design @ beta
    residuals = y - y_fit
    n_free = int(max(0, y.size - 2))
    residual_scale = (
        float(np.sum(weights * residuals**2) / n_free)
        if n_free > 0
        else 1.0
    )
    covariance = covariance_base * residual_scale
    log_a_err = float(np.sqrt(covariance[0, 0]))
    exponent_err = float(np.sqrt(covariance[1, 1]))

    amplitude = float(np.exp(log_a))
    amplitude_err = float(amplitude * log_a_err) if np.isfinite(log_a_err) else np.nan
    gain_at_reference = float(amplitude * LEGEND_REFERENCE_HV**exponent)

    return {
        "pmt": pmt,
        "amplitude": amplitude,
        "amplitude_err": amplitude_err,
        "exponent": exponent,
        "exponent_err": exponent_err,
        "log_amplitude": log_a,
        "log_amplitude_err": log_a_err,
        "gain_at_reference_hv": gain_at_reference,
        "reference_hv": LEGEND_REFERENCE_HV,
        "used_gain_errors": use_weights,
        "n_points": int(hv.size),
        "hv_min": float(np.min(hv)),
        "hv_max": float(np.max(hv)),
    }


def write_fit_csv(fit_rows):
    OUTPUT_FIT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "pmt",
        "amplitude",
        "amplitude_err",
        "exponent",
        "exponent_err",
        "log_amplitude",
        "log_amplitude_err",
        "gain_at_reference_hv",
        "reference_hv",
        "used_gain_errors",
        "n_points",
        "hv_min",
        "hv_max",
    ]

    with open(OUTPUT_FIT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in fit_rows:
            writer.writerow(row)


def legend_label(pmt, points):
    by_hv = {p["hv"]: p["gain"] for p in points}

    if LEGEND_REFERENCE_HV in by_hv:
        hv = LEGEND_REFERENCE_HV
    else:
        hv = max(by_hv)

    return f"{pmt} ({hv}V: {by_hv[hv]:.2e})"


def fit_legend_label(pmt, fit_result):
    exponent = fit_result["exponent"]
    exponent_err = fit_result["exponent_err"]

    if np.isfinite(exponent_err):
        exponent_text = rf"{exponent:.2f}\pm{exponent_err:.2f}"
    else:
        exponent_text = f"{exponent:.2f}"

    return rf"{pmt} fit: $k={exponent_text}$"


def plot_gain_vs_hv(points, fit_results=None):
    if not any(points.values()):
        raise RuntimeError(f"No gain points found in {BASE_OUTPUT_DIR}")

    fig, ax = plt.subplots(figsize=(10.5, 6.4))

    for pmt in PMTS:
        pmt_points = sorted(points.get(pmt, []), key=lambda item: item["hv"])
        if not pmt_points:
            continue

        hv = np.array([p["hv"] for p in pmt_points], dtype=float)
        gain = np.array([p["gain"] for p in pmt_points], dtype=float)
        gain_err = np.array([
            p["gain_err"] if p["gain_err"] is not None else np.nan
            for p in pmt_points
        ], dtype=float)

        yerr = gain_err if np.any(np.isfinite(gain_err)) else None

        color = PMT_COLORS.get(pmt, None)

        ax.errorbar(
            hv,
            gain,
            yerr=yerr,
            marker="o",
            markersize=6.5,
            linewidth=1.7,
            capsize=3.5,
            color=color,
            markerfacecolor=color,
            ecolor=color,
            label=legend_label(pmt, pmt_points),
        )

        fit_result = fit_results.get(pmt) if fit_results else None
        if fit_result is not None:
            hv_fit = np.linspace(
                fit_result["hv_min"],
                fit_result["hv_max"],
                FIT_N_POINTS,
            )
            gain_fit = fit_result["amplitude"] * hv_fit**fit_result["exponent"]

            ax.plot(
                hv_fit,
                gain_fit,
                linestyle=FIT_LINESTYLE,
                linewidth=FIT_LINEWIDTH,
                color=color,
                alpha=0.95,
                label=fit_legend_label(pmt, fit_result),
            )

    ax.set_title("PMT Gain vs High Voltage", fontsize=18)
    ax.set_xlabel("High Voltage [V]", fontsize=15)
    ax.set_ylabel("Gain [e]", fontsize=15)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.30)

    all_hv = sorted({p["hv"] for pmt_points in points.values() for p in pmt_points})
    if all_hv:
        ax.set_xticks(all_hv)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=12,
    )

    fig.tight_layout()
    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    points, rows = load_gain_points()
    fit_rows = []
    fit_results = {}

    if DO_FITS:
        for pmt in PMTS:
            pmt_points = sorted(points.get(pmt, []), key=lambda item: item["hv"])
            fit_result = fit_gain_power_law(pmt, pmt_points)
            if fit_result is not None:
                fit_results[pmt] = fit_result
                fit_rows.append(fit_result)

    write_csv(rows)
    if DO_FITS:
        write_fit_csv(fit_rows)
    plot_gain_vs_hv(points, fit_results=fit_results)

    print(f"Saved plot: {OUTPUT_PLOT}")
    print(f"Saved CSV : {OUTPUT_CSV}")
    if DO_FITS:
        print(f"Saved fits: {OUTPUT_FIT_CSV}")


if __name__ == "__main__":
    main()