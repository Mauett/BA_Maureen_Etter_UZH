#!/usr/bin/env python3
"""
plot_lv2480_800v_1p9vpp_spectrum_only.py

Make a plain integrated-charge spectrum plot for:
  PMT     = LV2480
  HV      = 800 V
  V_lamp  = any LED ON lamp voltage

The plot contains only the LED ON and LED OFF histograms. No gain extraction,
no model-dependent fit, no residual plot, and no fit-summary box are produced.
"""

import re
from pathlib import Path

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

from pmt_analysis.analysis.model_independent import GainModelIndependent
from pmt_analysis.processing.basics import FixedWindow
from pmt_analysis.utils.input import ADCRawData


# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

BASE_PATH = Path("path_to_data")

PMT_FOLDERS = [
    "data_20260112",
    "data_20260126",
    "data_20260127",
    "data_20260218",
    "data_20260304",
    "data_20260305",
    "data_20260306",
]

OUTPUT_DIR = Path(
    "/home/uzh/mauett/PMT_Analysis_Mauett/Gain/Gain_Plots/Spectrum_Only"
)

TARGET_PMT = "LV2480"
TARGET_HV = 800

# Set this to a number, for example 1.90, only if you want to force one lamp
# voltage. With None, every non-OFF LED ON file at TARGET_HV is allowed.
TARGET_LAMP_VPP = None

# Set this if you want to force one folder, for example "data_20260304".
SELECTED_FOLDER = "data_20260126"

# Set this if you want exactly one LED ON file.
# Example:
# SELECTED_ON_FILE = (
#     "SPELV2480_basev1.00U0-800V_Lamp1.95Vpp_Loff_0.50V_Lwid_20ns_"
#     "Lfreq700Hz_20260126_0_Module_0_0.root"
# )
SELECTED_ON_FILE = (
    "SPELV2480_basev1.00U0-800V_Lamp1.95Vpp_Loff_0.50V_Lwid_20ns_"
    "Lfreq700Hz_20260126_0_Module_0_0.root"
)

BSL_LOW = 0
BSL_HIGH = 100
PK_LOW = 100
PK_HIGH = None
CHANNEL = 0

OFF_LAMP_VPP = 0.50
LAMP_TOL = 1e-3

BIN_WIDTH_ADC = 10.0
HIGH_PERCENTILE = 99.995
MAX_X_ADC = 25000.0


# ------------------------------------------------------------------
# Filename helpers
# ------------------------------------------------------------------

_PMT_RE = re.compile(r"(LV\d{4})")
_LAMP_RE = re.compile(r"Lamp_?([0-9]+(?:\.[0-9]+)?)_?Vpp", re.IGNORECASE)
_HV_RE = re.compile(r"(?<![A-Za-z0-9.])(-?\d{3,4})V(?!pp)", re.IGNORECASE)


def normalized_name(name):
    return str(name).replace("-", "_")


def extract_pmt_serial(filename):
    match = _PMT_RE.search(str(filename))
    return match.group(1) if match else None


def extract_lamp_vpp(filename):
    match = _LAMP_RE.search(normalized_name(filename))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def extract_hv(filename):
    matches = _HV_RE.findall(str(filename))
    if not matches:
        matches = _HV_RE.findall(normalized_name(filename))
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def is_target_hv(filename):
    hv = extract_hv(filename)
    return hv is not None and abs(hv) == abs(TARGET_HV)


def is_target_lamp(filename):
    if TARGET_LAMP_VPP is None:
        return True

    lamp = extract_lamp_vpp(filename)
    return lamp is not None and abs(lamp - TARGET_LAMP_VPP) <= LAMP_TOL


def is_off_file(filename):
    lamp = extract_lamp_vpp(filename)
    return lamp is not None and abs(lamp - OFF_LAMP_VPP) <= LAMP_TOL


def find_input_files():
    pmt_folders = [SELECTED_FOLDER] if SELECTED_FOLDER is not None else PMT_FOLDERS

    on_candidates = []
    off_candidates = []

    for folder_name in pmt_folders:
        folder = BASE_PATH / folder_name
        if not folder.is_dir():
            continue

        for path in sorted(folder.rglob("*.root")):
            filename = path.name

            if not filename.startswith("SPE"):
                continue
            if extract_pmt_serial(filename) != TARGET_PMT:
                continue

            if is_off_file(filename):
                off_candidates.append(path)
                continue

            if SELECTED_ON_FILE is not None and filename != SELECTED_ON_FILE:
                continue

            if is_target_hv(filename) and is_target_lamp(filename):
                on_candidates.append(path)

    if not on_candidates:
        lamp_text = (
            "any LED ON lamp voltage"
            if TARGET_LAMP_VPP is None
            else f"Lamp {TARGET_LAMP_VPP:g} Vpp"
        )
        raise FileNotFoundError(
            f"No LED ON file found for {TARGET_PMT}, HV {TARGET_HV} V, "
            f"{lamp_text} below {BASE_PATH}."
        )

    if not off_candidates:
        raise FileNotFoundError(
            f"No LED OFF file found for {TARGET_PMT} below {BASE_PATH}."
        )

    def off_sort_key(path):
        hv = extract_hv(path.name)
        exact_hv_rank = 0 if hv is not None and abs(hv) == abs(TARGET_HV) else 1
        hv_distance = abs(abs(hv) - abs(TARGET_HV)) if hv is not None else float("inf")
        return exact_hv_rank, hv_distance, path.name

    def on_sort_key(path):
        lamp = extract_lamp_vpp(path.name)
        lamp_rank = lamp if lamp is not None else float("inf")
        return lamp_rank, path.name

    file_on = sorted(on_candidates, key=on_sort_key)[0]
    file_off = sorted(off_candidates, key=off_sort_key)[0]

    if len(on_candidates) > 1:
        print("Multiple matching LED ON files found; using:")
        print(f"  {file_on}")

    if len(off_candidates) > 1:
        print("Multiple LED OFF files found; using:")
        print(f"  {file_off}")

    return file_on, file_off


# ------------------------------------------------------------------
# Spectrum plot
# ------------------------------------------------------------------

def integrate_areas(input_path):
    raw = ADCRawData(
        raw_input_path=str(input_path.parent),
        raw_input_filepattern=input_path.name,
        verbose=True,
    )
    data = raw.get_branch_data(CHANNEL)
    fixed_window = FixedWindow(
        (BSL_LOW, BSL_HIGH),
        (PK_LOW, PK_HIGH),
    )
    return np.asarray(fixed_window.get_area(data), dtype=float)


def choose_histogram_edges(areas_on, areas_off):
    all_areas = np.concatenate([areas_on, areas_off]).astype(float)
    finite = all_areas[np.isfinite(all_areas)]

    if finite.size == 0:
        raise RuntimeError("No finite integrated areas found.")

    x_min = float(np.min(finite))
    x_max_percentile = float(np.percentile(finite, HIGH_PERCENTILE))
    x_max = min(MAX_X_ADC, x_max_percentile)

    if x_max <= x_min + BIN_WIDTH_ADC:
        x_max = min(MAX_X_ADC, float(np.max(finite)))

    return np.arange(x_min, x_max + BIN_WIDTH_ADC, BIN_WIDTH_ADC)


def plot_spectrum(file_on, file_off, areas_on, areas_off):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lamp_on = extract_lamp_vpp(file_on.name)
    lamp_label = "unknown" if lamp_on is None else f"{lamp_on:g}"
    lamp_filename = "unknown" if lamp_on is None else f"{lamp_on:g}"

    bin_edges = choose_histogram_edges(areas_on, areas_off)
    counts_on, _ = np.histogram(areas_on, bins=bin_edges)
    counts_off, _ = np.histogram(areas_off, bins=bin_edges)

    positive_counts = np.concatenate([
        counts_on[counts_on > 0],
        counts_off[counts_off > 0],
    ])
    y_min = float(np.min(positive_counts)) if positive_counts.size else 1.0
    y_max = float(max(np.max(counts_on), np.max(counts_off), 1.0))

    fig, ax = plt.subplots(figsize=(10.0, 6.0))

    # ax.step(
    #     bin_edges[:-1],
    #     counts_off,
    #     where="post",
    #     linewidth=1.2,
    #     color="darkred",
    #     label="LED OFF",
    # )
    ax.step(
        bin_edges[:-1],
        counts_on,
        where="post",
        linewidth=1.2,
        color="royalblue",
        label=rf"LED ON, $V_{{\mathrm{{lamp}}}}={lamp_label}\,\mathrm{{Vpp}}$",
    )

    ax.set_xlabel(r"Integrated ADC area $A\,[\mathrm{ADC}]$")
    ax.set_ylabel("Entries")
    ax.set_title(
        rf"{TARGET_PMT} - HV {TARGET_HV} V: integrated charge spectrum"
    )
    ax.set_yscale("log")
    ax.set_ylim(y_min, y_max * 2.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=True)

    output_png = OUTPUT_DIR / (
        f"{TARGET_PMT}_HV{TARGET_HV}V_Lamp{lamp_filename}Vpp_spectrum_only.png"
    )
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"LED ON : {file_on}")
    print(f"LED OFF: {file_off}")
    print(f"Saved spectrum-only plot: {output_png}")


def main():
    file_on, file_off = find_input_files()
    areas_on = integrate_areas(file_on)
    areas_off = integrate_areas(file_off)
    plot_spectrum(file_on, file_off, areas_on, areas_off)


if __name__ == "__main__":
    main()