#!/usr/bin/env python3
"""
Make zoomed diagnostic plots of secondary peaks 30-60 samples after the main pulse.

This script is meant to test the assumption behind the afterpulse cut:
secondary peaks closer than 60 samples to the main peak are likely just ringing
or swinging from the main pulse.

It intentionally does not call AfterPulses.find_ap(), because that method now
skips candidates with p1_position - p0_position < 60. Instead, it repeats the
same baseline subtraction and scipy.find_peaks call, then selects peaks with
30 <= delta_samples < 60.
"""

from pathlib import Path
from typing import Optional
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
from scipy.signal import find_peaks
from tqdm import tqdm

from pmt_analysis.processing.basics import FullWindow
from pmt_analysis.utils.input import ADCRawData


# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

BASE_PATH = Path("path_to_data")

PMT_FOLDERS = [
    "data",
    "data_20260112",
    "data_20260126",
    "data_20260127",
    "data_20260218",
    "data_20260304",
    "data_20260305",
    "data_20260306",
]


BASE_SAVE_DIR = Path(__file__).resolve().parent / "Afterpulse_Results"
BASE_SAVE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_LAMP_PER_PMT = {
    "LV2480": 2.15,
    "LV2481": 2.15,
    "LV2483": 2.15,
    "LV2485": 2.10,
}

# Same peak-finding settings as your full analysis.
HEIGHT = 50
DISTANCE = 25
PROMINENCE_STD = 8

# Region to inspect: second peak delay relative to the first/main peak.
MIN_DELAY_SAMPLES = 30
MAX_DELAY_SAMPLES = 60

# Detailed ringing zoom relative to the main pulse. Start well before the peak
# so the main-pulse rise, peak, tail, and recovery are visible in the same view.
RINGING_ZOOM_START_SAMPLES = -40
RINGING_ZOOM_STOP_SAMPLES = 120

# Keep the diagnostic output manageable.
CHUNK_SIZE = 20000
MAX_WAVEFORMS_PER_RUN: Optional[int] = None
MAX_PLOTS_PER_PMT = 10

# Optional prefilter, equivalent in spirit to AfterPulses.__init__.
PRE_FILTER_THRESHOLD_STD = 3

BASE_SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Small parsing helpers
# ------------------------------------------------------------------

def extract_pmt_from_name(name: str) -> Optional[str]:
    match = re.search(r"(LV\d{4})", str(name))
    return match.group(1) if match else None


def extract_lamp_from_name(name: str) -> Optional[float]:
    match = re.search(r"Lamp_?([0-9]+(?:\.[0-9]+)?)_?Vpp", str(name), re.IGNORECASE)
    return float(match.group(1)) if match else None


def extract_hv_from_name(name: str) -> Optional[int]:
    match = re.search(r"(?<![A-Za-z0-9.])(-?\d{3,4})\s*V(?!pp)", str(name), re.IGNORECASE)
    return int(match.group(1)) if match else None


def format_hv(hv: Optional[int]) -> str:
    return "unknown" if hv is None else f"{hv} V"


def format_lamp(lamp: Optional[float]) -> str:
    return "unknown" if lamp is None else f"{lamp:g} Vpp"


# ------------------------------------------------------------------
# Waveform helpers
# ------------------------------------------------------------------

def convert_to_amplitude(input_data: np.ndarray) -> np.ndarray:
    """Return baseline-subtracted, sign-reversed waveform amplitudes."""
    baselines = FullWindow().get_baseline(input_data)
    return -np.subtract(input_data.T, baselines).T


def pre_filter_waveforms(input_data: np.ndarray, threshold_std: float):
    """Keep waveforms with amplitude above threshold_std * baseline std."""
    input_data_std = FullWindow().get_baseline_std(input_data)
    amplitudes = FullWindow().get_amplitude(input_data)
    keep = amplitudes > threshold_std * input_data_std
    return input_data[keep], input_data_std[keep], np.flatnonzero(keep)


def find_early_secondary_pulses(
    input_data: np.ndarray,
    adc_f: float,
    global_offset: int,
    max_candidates: int,
):
    """
    Find candidate secondary peaks in the 30-60 sample window after the main peak.

    Returns a list of dictionaries containing waveform data and pulse metadata.
    """
    filtered_data, input_data_std, kept_indices = pre_filter_waveforms(
        input_data,
        PRE_FILTER_THRESHOLD_STD,
    )

    if filtered_data.size == 0:
        return []

    amplitudes = convert_to_amplitude(filtered_data)
    prominence = PROMINENCE_STD * input_data_std
    candidates = []

    for local_i, waveform in tqdm(
        enumerate(amplitudes),
        total=len(amplitudes),
        desc="Finding 30-60 sample secondary peaks",
    ):
        peak_positions, properties = find_peaks(
            waveform,
            height=HEIGHT,
            prominence=prominence[local_i],
            distance=DISTANCE,
        )

        if len(peak_positions) < 2:
            continue

        p0_position = int(peak_positions[0])
        peak_heights = properties.get("peak_heights", np.full(len(peak_positions), np.nan))

        for peak_index, p1_position_raw in enumerate(peak_positions[1:], start=1):
            p1_position = int(p1_position_raw)
            delta_samples = p1_position - p0_position

            if MIN_DELAY_SAMPLES <= delta_samples < MAX_DELAY_SAMPLES:
                candidates.append({
                    "global_waveform_index": int(global_offset + kept_indices[local_i]),
                    "p0_position": p0_position,
                    "p1_position": p1_position,
                    "delta_samples": int(delta_samples),
                    "delta_ns": float(delta_samples / (adc_f * 1e-9)),
                    "p0_amplitude_adc": float(waveform[p0_position]),
                    "p1_amplitude_adc": float(waveform[p1_position]),
                    "p1_height_from_find_peaks": float(peak_heights[peak_index]),
                    "baseline_std": float(input_data_std[local_i]),
                    "waveform": waveform,
                })

                if len(candidates) >= max_candidates:
                    return candidates

    return candidates


def make_scientific_figure(figsize=(10.5, 6.0)):
    """Create a main plotting axis and an unboxed right-side information axis."""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[3.35, 1.00],
        left=0.10,
        right=0.98,
        bottom=0.14,
        top=0.88,
        wspace=0.12,
    )

    ax = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis("off")
    ax_info.axvline(0.0, color="0.70", linewidth=0.8)
    return fig, ax, ax_info


def add_side_info(ax_info, info_text: str):
    ax_info.text(
        0.08,
        0.98,
        info_text,
        transform=ax_info.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        linespacing=1.22,
    )


def plot_candidate(
    candidate,
    adc_f: float,
    save_path: Path,
    title_prefix: str,
    pmt_serial: str,
    hv_v: str,
    lamp_v: str,
):
    waveform = candidate["waveform"]
    p0 = candidate["p0_position"]
    p1 = candidate["p1_position"]

    lo = max(0, p0 + RINGING_ZOOM_START_SAMPLES)
    hi = min(len(waveform), p0 + RINGING_ZOOM_STOP_SAMPLES)

    x_samples = np.arange(lo, hi) - p0
    y = waveform[lo:hi]

    fig, ax, ax_info = make_scientific_figure()
    ax.step(x_samples, y, where="mid", linewidth=1.35, label="Waveform")
    ax.plot(x_samples, y, ".", color="C0", markersize=3.0, alpha=0.75, label="ADC samples")

    ax.axvline(0, color="0.25", linestyle="--", linewidth=1.15, label="Main peak")
    ax.axvline(
        p1 - p0,
        color="C3",
        linestyle="--",
        linewidth=1.15,
        label=f"Secondary peak: {p1 - p0} samples",
    )
    ax.axvspan(
        MIN_DELAY_SAMPLES,
        MAX_DELAY_SAMPLES,
        color="C3",
        alpha=0.12,
        label="Inspection window",
    )
    ax.axhline(0, color="0.55", linewidth=0.8)
    ax.axhline(
        candidate["baseline_std"],
        color="0.45",
        linestyle=":",
        linewidth=0.9,
        label=r"Baseline",
    )
    ax.axhline(
        -candidate["baseline_std"],
        color="0.45",
        linestyle=":",
        linewidth=0.9,
    )

    info = (
        r"$\mathbf{Early\ pulse\ check}$" "\n"
        rf"$\mathrm{{PMT}} = {pmt_serial}$" "\n"
        rf"$V_{{\mathrm{{HV}}}} = {hv_v.replace(' V', '')}\,\mathrm{{V}}$" "\n"
        rf"$V_{{\mathrm{{lamp}}}} = {lamp_v.replace(' Vpp', '')}\,\mathrm{{V_{{pp}}}}$" "\n"
        "\n"
        r"$\mathbf{Selection}$" "\n"
        rf"${MIN_DELAY_SAMPLES}\leq\Delta n<{MAX_DELAY_SAMPLES}\,\mathrm{{samples}}$" "\n"
        rf"$H_{{\mathrm{{min}}}} = {HEIGHT}\,\mathrm{{ADC}}$" "\n"
        rf"$d_{{\min}} = {DISTANCE}\,\mathrm{{samples}}$" "\n"
        #rf"$p_{{\mathrm{{prom}}}} = {PROMINENCE_STD}\sigma_{{\mathrm{{baseline}}}}$" "\n"
        "\n"
        r"$\mathbf{Candidate}$" "\n"
        rf"$N_{{\mathrm{{wf}}}} = {candidate['global_waveform_index']}$" "\n"
        rf"$p_0 = {p0}\,\mathrm{{samples}}$" "\n"
        rf"$p_1 = {p1}\,\mathrm{{samples}}$" "\n"
        rf"$\Delta n = {candidate['delta_samples']}\,\mathrm{{samples}}$" "\n"
        rf"$\Delta t = {candidate['delta_ns']:.1f}\,\mathrm{{ns}}$" "\n"
        rf"$A_{{P_0}} = {candidate['p0_amplitude_adc']:.1f}\,\mathrm{{ADC}}$" "\n"
        rf"$A_{{P_1}} = {candidate['p1_amplitude_adc']:.1f}\,\mathrm{{ADC}}$"
    )

    add_side_info(ax_info, info)

    ax.set_xlim(RINGING_ZOOM_START_SAMPLES, RINGING_ZOOM_STOP_SAMPLES)
    ax.set_xlabel("Sample offset from main peak")
    ax.set_ylabel("Amplitude [ADC]")
    ax.set_title(title_prefix)
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.12)
    ax.legend(
        loc="upper right",
        fontsize=9,
        frameon=True,
        fancybox=False,
        framealpha=0.92,
        edgecolor="0.45",
    )

    secax = ax.secondary_xaxis(
        "top",
        functions=(
            lambda samples: samples / (adc_f * 1e-9),
            lambda ns: ns * (adc_f * 1e-9),
        ),
    )
    secax.set_xlabel("Time offset from main peak [ns]")

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def process_run(run_folder: Path, filepattern: str, run_name: str, remaining_plots: int):
    pmt_serial = extract_pmt_from_name(run_name) or "unknown"
    hv_num = extract_hv_from_name(run_name)
    lamp_v_num = extract_lamp_from_name(run_name)
    hv_v = format_hv(hv_num)
    lamp_v = format_lamp(lamp_v_num)

    target_lamp = TARGET_LAMP_PER_PMT.get(pmt_serial)
    if target_lamp is not None:
        if lamp_v_num is None:
            print(f"Skipping {run_name}: no lamp voltage found.")
            return []
        if abs(lamp_v_num - target_lamp) > 1e-6:
            print(
                f"Skipping {run_name}: found Lamp {lamp_v_num:g} Vpp, "
                f"expected {target_lamp:g} Vpp for {pmt_serial}."
            )
            return []

    print("\n================================================")
    print(f"Inspecting early secondary peaks for PMT: {pmt_serial}")
    print(f"Run: {run_name}")
    print(f"Lamp voltage: {lamp_v}")
    print(f"HV: {hv_v}")
    print("================================================")

    raw_data = ADCRawData(
        raw_input_path=str(run_folder),
        raw_input_filepattern=filepattern,
    )
    data = raw_data.get_branch_data(0)

    if data.shape[1] == 3500:
        data = data[:, :1500]
        print(f"Cropped 3500-sample waveform to {data.shape}")

    if MAX_WAVEFORMS_PER_RUN is not None:
        data = data[:MAX_WAVEFORMS_PER_RUN]

    run_save_dir = BASE_SAVE_DIR / pmt_serial / run_name
    run_save_dir.mkdir(parents=True, exist_ok=True)

    if remaining_plots <= 0:
        return []

    all_candidates = []
    n_total = len(data)

    for start in range(0, n_total, CHUNK_SIZE):
        if len(all_candidates) >= remaining_plots:
            print(
                f"Reached PMT plot budget of {MAX_PLOTS_PER_PMT}; "
                "stopping this run early."
            )
            break

        stop = min(start + CHUNK_SIZE, n_total)
        chunk = data[start:stop]
        print(f"Chunk {start}:{stop} / {n_total}")

        chunk_candidates = find_early_secondary_pulses(
            input_data=chunk,
            adc_f=raw_data.adc_f,
            global_offset=start,
            max_candidates=remaining_plots - len(all_candidates),
        )
        all_candidates.extend(chunk_candidates)

    print(f"Found {len(all_candidates)} candidate(s) in the 30-60 sample window.")

    rows = []
    for plot_i, candidate in enumerate(all_candidates[:remaining_plots]):
        save_path = run_save_dir / (
            f"early_secondary_zoom_{plot_i:04d}_"
            f"wf{candidate['global_waveform_index']}_"
            f"d{candidate['delta_samples']}samples.png"
        )
        title = (
            f"Peaks close to main pulse"
        )
        plot_candidate(
            candidate,
            raw_data.adc_f,
            save_path,
            title,
            pmt_serial,
            hv_v,
            lamp_v,
        )

    for candidate in all_candidates:
        row = {key: value for key, value in candidate.items() if key != "waveform"}
        row.update({
            "pmt_serial": pmt_serial,
            "run_name": run_name,
            "hv_v": hv_v,
            "lamp_v": lamp_v,
        })
        rows.append(row)

    run_summary = pd.DataFrame(rows)
    run_summary_path = run_save_dir / "early_secondary_candidates.csv"
    run_summary.to_csv(run_summary_path, index=False)
    print(f"Saved run summary: {run_summary_path}")

    return rows


def main():
    all_rows = []
    plots_by_pmt = {}

    for pmt_folder in PMT_FOLDERS:
        folder = BASE_PATH / pmt_folder
        print(f"\nChecking folder: {folder}")

        if not folder.is_dir():
            print("Folder does not exist, skipping.")
            continue

        subfolders = sorted(
            f for f in folder.iterdir()
            if f.is_dir() and f.name.startswith("APLV")
        )

        for run_folder in subfolders:
            run_name = run_folder.name
            pmt_serial = extract_pmt_from_name(run_name) or "unknown"

            if plots_by_pmt.get(pmt_serial, 0) >= MAX_PLOTS_PER_PMT:
                print(f"Already found {MAX_PLOTS_PER_PMT} candidates for {pmt_serial}, skipping remaining runs.")
                continue

            filepattern = f"{run_name}_Module_0_*.root"
            matching_files = sorted(run_folder.glob(filepattern))

            if not matching_files:
                print(f"No ROOT files found for run: {run_name}")
                continue

            try:
                with uproot.open(matching_files[0]) as root_file:
                    if not root_file.keys():
                        print(f"ROOT file contains no objects: {matching_files[0]}")
                        continue

                remaining = MAX_PLOTS_PER_PMT - plots_by_pmt.get(pmt_serial, 0)
                run_rows = process_run(run_folder, filepattern, run_name, remaining)
                all_rows.extend(run_rows)
                plots_by_pmt[pmt_serial] = plots_by_pmt.get(pmt_serial, 0) + len(run_rows)
            except Exception as exc:
                warnings.warn(f"Skipping {run_name} after error: {exc}")
                continue

    summary = pd.DataFrame(all_rows)
    summary_path = BASE_SAVE_DIR / "early_secondary_candidates_all_runs.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved combined summary: {summary_path}")

    if summary.empty:
        print("No 30-60 sample secondary peak candidates found.")
        return

    counts = (
        summary.groupby(["pmt_serial", "run_name"])
        .size()
        .reset_index(name="n_candidates_30_to_60_samples")
        .sort_values("n_candidates_30_to_60_samples", ascending=False)
    )
    counts_path = BASE_SAVE_DIR / "early_secondary_candidate_counts.csv"
    counts.to_csv(counts_path, index=False)
    print(f"Saved counts summary: {counts_path}")


if __name__ == "__main__":
    main()