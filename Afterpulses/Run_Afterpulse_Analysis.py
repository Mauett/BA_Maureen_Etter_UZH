#!/usr/bin/env python3
"""
Full-data afterpulse analysis with additional readable waveform plots.

The extra waveform plots are produced inside this script and use an automatic
y-axis zoom based on the afterpulse height.
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot

from pmt_analysis.processing.afterpulses import AfterPulses
from pmt_analysis.plotting.afterpulses import PlottingAfterpulses as BasePlottingAfterpulses
from pmt_analysis.utils.input import ADCRawData


# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

BASE_PATH = Path("/disk/gfs_atp/lhoetz/marmotx")

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

GAIN_RESULTS_DIRS = [
    Path(__file__).resolve().parent.parent / "Gain" / "Gain_Results",
]

GAIN_LAMP_TOL = 1e-3

TARGET_LAMP_PER_PMT = {
    "LV2480": 2.15,
    "LV2481": 2.15,
    "LV2483": 2.15,
    "LV2485": 2.10,
}

TARGET_GAIN_LAMP_PER_PMT = {
    "LV2480": 1.90,
    "LV2481": 1.95,
    "LV2483": 1.95,
    "LV2485": 1.95,
}

AREA_THRESHOLD_PE = 2
TIME_THRESHOLD_NS = None
N_WAVEFORMS_PLOT = 10000
CHUNK_SIZE_ANALYSIS = 20000
REQUIRED_N_SAMPLES_PER_WAVEFORM = 2500

HEIGHT = 50
DISTANCE = 25

# Extra readable waveform example plots. These are saved in addition to the
# standard PlottingAfterpulses plots. The main pulse may be clipped so the
# smaller afterpulse is visible.
MAKE_ZOOM_WAVEFORM_PLOTS = True
ZOOM_WAVEFORM_N_EXAMPLES = 5
ZOOM_WAVEFORM_YMIN_ADC = -100.0
ZOOM_WAVEFORM_YMAX_MARGIN = 1.35
ZOOM_WAVEFORM_MIN_YMAX_ADC = 250.0

# Adjustable early-window afterpulse logic.
MIN_AFTERPULSE_DELAY_SAMPLES = 10
CONDITIONAL_DELAY_MIN_SAMPLES = 10
CONDITIONAL_DELAY_MAX_SAMPLES = 60
CONDITIONAL_MIN_PEAK_HEIGHT_ADC = 100

BASE_SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Small parsing helpers
# ------------------------------------------------------------------

def finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    return value if np.isfinite(value) else None


def extract_pmt_from_name(name: str) -> Optional[str]:
    match = re.search(r"(LV\d{4})", str(name))
    return match.group(1) if match else None


def extract_lamp_from_name(name: str) -> Optional[float]:
    match = re.search(
        r"Lamp_?([0-9]+(?:\.[0-9]+)?)_?Vpp",
        str(name),
        re.IGNORECASE,
    )
    return float(match.group(1)) if match else None


def extract_hv_from_name(name: str) -> Optional[int]:
    match = re.search(
        r"(?<![A-Za-z0-9.])(-?\d{3,4})\s*V(?!pp)",
        str(name),
        re.IGNORECASE,
    )
    return int(match.group(1)) if match else None


def format_hv(hv: Optional[int]) -> str:
    return "unknown" if hv is None else f"{hv} V"


def format_lamp(lamp: Optional[float]) -> str:
    return "unknown" if lamp is None else f"{lamp:g} Vpp"


def scientific_latex(value, uncertainty=None, digits=3):
    value = finite_float(value)
    if value is None:
        return r"\mathrm{n/a}"

    if value == 0:
        if uncertainty is not None and finite_float(uncertainty) is not None:
            return rf"(0.000 \pm {float(uncertainty):.{digits}g})"
        return "0"

    exponent = int(np.floor(np.log10(abs(value))))
    scale = 10.0 ** exponent
    mantissa = value / scale

    uncertainty = finite_float(uncertainty)
    if uncertainty is not None and uncertainty > 0:
        uncertainty_mantissa = uncertainty / scale
        return (
            rf"({mantissa:.{digits}f} \pm "
            rf"{uncertainty_mantissa:.{digits}f})\times 10^{{{exponent}}}"
        )

    return rf"{mantissa:.{digits}f}\times 10^{{{exponent}}}"


class PlottingAfterpulses(BasePlottingAfterpulses):
    """Add afterpulse-rate uncertainties to the standard plot information box."""

    def _plot_info_text(self):
        try:
            text = super()._plot_info_text()
        except AttributeError:
            text = r"$\mathbf{Afterpulse\ analysis}$"

        if self.ap_rate_dict is None:
            return text

        rate = self.ap_rate_dict.get("ap_rate_per_pe_separable")
        rate_unc = self.ap_rate_dict.get("ap_rate_per_pe_separable_unc")
        if finite_float(rate) is None:
            return text

        rate_line = (
            rf"$R_{{\mathrm{{AP}}}}/\mathrm{{PE}} = "
            rf"{scientific_latex(rate, rate_unc)}$"
        )

        lines = text.splitlines()
        for idx, line in enumerate(lines):
            if "R_" in line and "AP" in line and "PE" in line:
                lines[idx] = rate_line
                return "\n".join(lines)

        lines.extend(["", r"$\mathbf{Afterpulse\ rate}$", rate_line])
        return "\n".join(lines)


# ------------------------------------------------------------------
# Gain-result JSON lookup
# ------------------------------------------------------------------

def read_gain_result(path: Path):
    try:
        with open(path, "r") as file:
            data = json.load(file)
    except Exception as exc:
        print(f"Could not read gain result JSON: {path} ({exc})")
        return None

    search_name = f"{path.stem}_{path.parent.name}_{path.parent.parent.name}"
    pmt_serial = extract_pmt_from_name(search_name)
    hv_v = extract_hv_from_name(search_name)
    lamp_v = extract_lamp_from_name(search_name)

    model_dep = data.get("model_dependent_fit", {})
    md_params = model_dep.get("params", {}) if isinstance(model_dep, dict) else {}

    md_gain = finite_float(md_params.get("gain_e"))
    md_gain_err = finite_float(md_params.get("gain_e_err"))
    mi_gain = finite_float(data.get("gain"))

    if md_gain is not None:
        selected_gain = md_gain
        selected_gain_err = md_gain_err
        selected_source = "model_dependent"
    elif mi_gain is not None:
        selected_gain = mi_gain
        selected_gain_err = None
        selected_source = "model_independent"
    else:
        return None

    return {
        "path": path,
        "pmt_serial": pmt_serial,
        "hv_v": hv_v,
        "lamp_v": lamp_v,
        "gain": selected_gain,
        "gain_err": selected_gain_err,
        "gain_source": selected_source,
        "model_dependent_gain": md_gain,
        "model_dependent_gain_err": md_gain_err,
        "model_independent_gain": mi_gain,
        "fit_success": (
            bool(model_dep.get("fit_success", False))
            if isinstance(model_dep, dict)
            else False
        ),
        "mtime": path.stat().st_mtime,
    }


def load_gain_results():
    records = []

    for base_dir in GAIN_RESULTS_DIRS:
        if not base_dir.is_dir():
            print(f"Gain results directory does not exist: {base_dir}")
            continue

        for path in sorted(base_dir.rglob("*_results.json")):
            record = read_gain_result(path)
            if record is not None:
                records.append(record)

    print(f"\nLoaded {len(records)} gain result record(s).")
    return records


def select_gain_for_run(gain_records, pmt_serial, hv_v):
    hv_num = extract_hv_from_name(hv_v)
    target_gain_lamp = TARGET_GAIN_LAMP_PER_PMT.get(pmt_serial)

    candidates = [
        record
        for record in gain_records
        if record["pmt_serial"] == pmt_serial and record["gain"] is not None
    ]

    if hv_num is not None:
        candidates = [
            record for record in candidates
            if record["hv_v"] == hv_num
        ]

    if target_gain_lamp is not None:
        candidates = [
            record
            for record in candidates
            if (
                record["lamp_v"] is not None
                and abs(record["lamp_v"] - target_gain_lamp) <= GAIN_LAMP_TOL
            )
        ]

    if not candidates:
        return None

    def score(record):
        source_score = 0 if record["gain_source"] == "model_dependent" else 1
        fit_score = 0 if record["fit_success"] else 1
        lamp_score = 0.0

        if target_gain_lamp is not None and record["lamp_v"] is not None:
            lamp_score = abs(record["lamp_v"] - target_gain_lamp)

        return source_score, fit_score, lamp_score, -record["mtime"]

    return sorted(candidates, key=score)[0]


def gain_lookup_summary(gain_records):
    match_text = r"$\mathrm{PMT+HV+gain\ lamp}$"

    return (
        r"$\mathbf{Summary}$" "\n"
        r"$\mathrm{Metric:}\ R_{\mathrm{AP}}/\mathrm{PE}\ "
        r"\mathrm{(separable)}$" "\n"
        rf"$A_{{\mathrm{{thr}}}} = {AREA_THRESHOLD_PE}\,\mathrm{{PE}}$" "\n"
        + (
            r"$t_{\mathrm{thr}} = \mathrm{None}$" "\n"
            if TIME_THRESHOLD_NS is None
            else rf"$t_{{\mathrm{{thr}}}} = {TIME_THRESHOLD_NS}\,\mathrm{{ns}}$" "\n"
        )
        + r"$\mathrm{Normalization:}\ \sum A_{P0}\ \mathrm{from\ AP\ candidates}$" "\n"
        + r"$N_{\mathrm{wf/run}} = \mathrm{full\ dataset\ (chunked)}$" "\n"
        + rf"$N_{{\mathrm{{frame}}}} = {REQUIRED_N_SAMPLES_PER_WAVEFORM}\,\mathrm{{samples}}$" "\n"
        + rf"$N_{{\mathrm{{wf,plot}}}} = {N_WAVEFORMS_PLOT}$" "\n"
        + rf"$N_{{\mathrm{{wf,chunk}}}} = {CHUNK_SIZE_ANALYSIS}$" "\n"
        + "\n"
        + r"$\mathbf{Gain\ input}$" "\n"
        + r"$\mathrm{Source:}\ \mathrm{gain\ result\ JSONs}$" "\n"
        + r"$\mathrm{Preferred:}\ G_{\mathrm{MD}}$" "\n"
        + rf"$\mathrm{{Match:}}\ {match_text}$" "\n"
        + rf"$N_{{\mathrm{{gain\ files}}}} = {len(gain_records)}$"
    )


def gain_lookup_frame_summary(gain_records):
    return (
        gain_lookup_summary(gain_records)
        + "\n"
        + r"$\mathrm{Frame\ groups: actual\ waveform\ sample\ counts}$"
    )


def add_summary_box_right(fig, text, x=0.69, y=0.88, fontsize=10):
    fig.text(
        x,
        y,
        text,
        ha="left",
        va="top",
        fontsize=fontsize,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="black",
            alpha=0.92,
        ),
    )


# ------------------------------------------------------------------
# Afterpulse processing
# ------------------------------------------------------------------

def make_afterpulses(
    input_data,
    adc_f,
    adc_area_to_e,
    pmt_gain,
    area_threshold_pe,
    time_threshold_ns,
):
    return AfterPulses(
        input_data=input_data,
        adc_f=adc_f,
        verbose=True,
        area_thr_ap=area_threshold_pe,
        t_thr_ap=time_threshold_ns,
        adc_area_to_e=adc_area_to_e,
        gain=pmt_gain,
    )


def compute_chunked_afterpulse_stats(
    data_full,
    adc_f,
    adc_area_to_e,
    pmt_gain,
    area_threshold_pe,
    time_threshold_ns,
    height=50,
    distance=25,
):
    total_n_samples = 0
    total_n_main_pulses = 0
    total_n_ap = 0
    total_n_ap_separable = 0
    total_n_ap_separable_above_thr = 0
    total_n_ap_separable_below_thr = 0
    total_sum_p0_area_pe = 0.0

    n_total = len(data_full)

    for start in range(0, n_total, CHUNK_SIZE_ANALYSIS):
        stop = min(start + CHUNK_SIZE_ANALYSIS, n_total)
        data_chunk = data_full[start:stop]

        print(f"\n--- Full-data chunk: {start}:{stop} / {n_total}")

        ap_chunk = make_afterpulses(
            input_data=data_chunk,
            adc_f=adc_f,
            adc_area_to_e=adc_area_to_e,
            pmt_gain=pmt_gain,
            area_threshold_pe=area_threshold_pe,
            time_threshold_ns=time_threshold_ns,
        )

        ap_chunk.find_ap(height=height, distance=distance)
        total_n_samples += ap_chunk.n_samples
        if ap_chunk.df.empty:
            continue

        ap_chunk.constrain_main_peak(trim=True)
        if ap_chunk.df.empty:
            continue

        ap_chunk.get_ap_properties()
        if ap_chunk.df.empty:
            continue

        ap_chunk.multi_ap()
        ap_chunk.ap_rate()

        rate = ap_chunk.ap_rate_dict
        total_n_ap += int(rate.get("n_ap", 0))
        total_n_ap_separable += int(rate.get("n_ap_separable", 0))
        total_n_ap_separable_above_thr += int(rate.get("n_ap_separable_above_thr", 0) or 0)
        total_n_ap_separable_below_thr += int(rate.get("n_ap_separable_below_thr", 0) or 0)
        total_sum_p0_area_pe += float(rate.get("sum_p0_area_pe", 0.0) or 0.0)

        if "idx" in ap_chunk.df.columns:
            total_n_main_pulses += int(len(np.unique(ap_chunk.df["idx"])))

    if total_n_samples > 0:
        ap_fraction = total_n_ap / total_n_samples
        ap_fraction_unc = (
            np.sqrt(total_n_ap) / total_n_samples
            if total_n_ap > 0
            else 0.0
        )
        ap_fraction_separable = total_n_ap_separable / total_n_samples
        ap_fraction_separable_unc = (
            np.sqrt(total_n_ap_separable) / total_n_samples
            if total_n_ap_separable > 0
            else 0.0
        )
    else:
        ap_fraction = 0.0
        ap_fraction_unc = 0.0
        ap_fraction_separable = 0.0
        ap_fraction_separable_unc = 0.0

    if total_sum_p0_area_pe > 0:
        ap_rate_per_pe = total_n_ap / total_sum_p0_area_pe
        ap_rate_per_pe_unc = (
            np.sqrt(total_n_ap) / total_sum_p0_area_pe
            if total_n_ap > 0
            else 0.0
        )
        ap_rate_per_pe_separable = (
            total_n_ap_separable / total_sum_p0_area_pe
        )
        ap_rate_per_pe_separable_unc = (
            np.sqrt(total_n_ap_separable) / total_sum_p0_area_pe
            if total_n_ap_separable > 0
            else 0.0
        )
        ap_rate_per_pe_above_thr = (
            total_n_ap_separable_above_thr / total_sum_p0_area_pe
        )
        ap_rate_per_pe_above_thr_unc = (
            np.sqrt(total_n_ap_separable_above_thr) / total_sum_p0_area_pe
            if total_n_ap_separable_above_thr > 0
            else 0.0
        )
        ap_rate_per_pe_below_thr = (
            total_n_ap_separable_below_thr / total_sum_p0_area_pe
        )
        ap_rate_per_pe_below_thr_unc = (
            np.sqrt(total_n_ap_separable_below_thr) / total_sum_p0_area_pe
            if total_n_ap_separable_below_thr > 0
            else 0.0
        )
        mean_p0_area_pe = (
            total_sum_p0_area_pe / total_n_main_pulses
            if total_n_main_pulses > 0
            else 0.0
        )
    else:
        ap_rate_per_pe = 0.0
        ap_rate_per_pe_unc = 0.0
        ap_rate_per_pe_separable = 0.0
        ap_rate_per_pe_separable_unc = 0.0
        ap_rate_per_pe_above_thr = 0.0
        ap_rate_per_pe_above_thr_unc = 0.0
        ap_rate_per_pe_below_thr = 0.0
        ap_rate_per_pe_below_thr_unc = 0.0
        mean_p0_area_pe = 0.0

    return {
        "n_samples": total_n_samples,
        "n_main_pulses": total_n_main_pulses,
        "n_ap": total_n_ap,
        "n_ap_separable": total_n_ap_separable,
        "ap_fraction": ap_fraction,
        "ap_fraction_unc": ap_fraction_unc,
        "ap_fraction_separable": ap_fraction_separable,
        "ap_fraction_separable_unc": ap_fraction_separable_unc,
        "mean_p0_area_pe": mean_p0_area_pe,
        "sum_p0_area_pe": total_sum_p0_area_pe,
        "ap_rate_per_pe": ap_rate_per_pe,
        "ap_rate_per_pe_unc": ap_rate_per_pe_unc,
        "ap_rate_per_pe_separable": ap_rate_per_pe_separable,
        "ap_rate_per_pe_separable_unc": ap_rate_per_pe_separable_unc,
        "n_ap_separable_above_thr": total_n_ap_separable_above_thr,
        "n_ap_separable_below_thr": total_n_ap_separable_below_thr,
        "ap_rate_per_pe_above_thr": ap_rate_per_pe_above_thr,
        "ap_rate_per_pe_above_thr_unc": ap_rate_per_pe_above_thr_unc,
        "ap_rate_per_pe_below_thr": ap_rate_per_pe_below_thr,
        "ap_rate_per_pe_below_thr_unc": ap_rate_per_pe_below_thr_unc,
        "area_thr_ap": area_threshold_pe,
        "t_thr_ap": time_threshold_ns,
    }


# ------------------------------------------------------------------
# Extra waveform plots
# ------------------------------------------------------------------

def sample_to_ns(sample, adc_f):
    return float(sample) / float(adc_f) * 1e9


def afterpulse_zoom_ymax(row, waveform):
    p1_height = row.get("p1_amplitude", None)
    try:
        p1_height = float(p1_height)
    except (TypeError, ValueError):
        p1_height = np.nan

    if not np.isfinite(p1_height) or p1_height <= 0:
        try:
            p1_position = int(row["p1_position"])
            p1_height = float(waveform[p1_position])
        except Exception:
            p1_height = np.nan

    if not np.isfinite(p1_height) or p1_height <= 0:
        p1_height = ZOOM_WAVEFORM_MIN_YMAX_ADC

    return max(
        ZOOM_WAVEFORM_MIN_YMAX_ADC,
        ZOOM_WAVEFORM_YMAX_MARGIN * p1_height,
    )


def plot_zoom_afterpulse_waveforms(
    df,
    adc_f,
    save_dir,
    save_name_suffix,
    pmt_serial,
    hv_v,
    lamp_v,
):
    """
    Save additional afterpulse-candidate waveform plots with a y-axis range
    chosen from the afterpulse height. The main pulse may be clipped so smaller
    afterpulses are easier to see.
    """
    if df.empty:
        return

    save_dir = Path(save_dir)
    n_examples = min(ZOOM_WAVEFORM_N_EXAMPLES, df.shape[0])

    for i in range(n_examples):
        row = df.iloc[i]
        waveform = np.asarray(row["input_data_converted"], dtype=float)
        time_ns = np.arange(waveform.size, dtype=float) / float(adc_f) * 1e9

        p0_t = sample_to_ns(row["p0_position"], adc_f)
        p1_t = sample_to_ns(row["p1_position"], adc_f)
        dt_ns = float(row["t_diff_ns"])
        separability = "separable" if bool(row["separable"]) else "non-separable"

        fig, ax = plt.subplots(figsize=(11.5, 6.8))

        ax.axvspan(
            sample_to_ns(row["p0_lower_bound"], adc_f),
            sample_to_ns(row["p0_upper_bound"], adc_f),
            color="tab:orange",
            alpha=0.14,
            linewidth=0,
            zorder=1,
            label="Main pulse",
        )
        ax.axvspan(
            sample_to_ns(row["p1_lower_bound"], adc_f),
            sample_to_ns(row["p1_upper_bound"], adc_f),
            color="tab:red",
            alpha=0.12,
            linewidth=0,
            zorder=1,
            label=f"Afterpulse ({separability})",
        )
        ax.axvline(
            p0_t,
            color="0.25",
            linestyle="--",
            linewidth=1.1,
            alpha=0.65,
            zorder=2,
            label=rf"Peak positions, $\Delta t={dt_ns:.1f}\,\mathrm{{ns}}$",
        )
        ax.axvline(
            p1_t,
            color="0.25",
            linestyle="--",
            linewidth=1.1,
            alpha=0.65,
            zorder=2,
        )
        ax.step(
            time_ns,
            waveform,
            where="mid",
            linewidth=1.35,
            color="tab:blue",
            alpha=0.95,
            zorder=4,
            label="Waveform",
        )

        y_max = afterpulse_zoom_ymax(row, waveform)
        ax.set_ylim(ZOOM_WAVEFORM_YMIN_ADC, y_max)

        ax.set_xlabel(r"Time $t\,[\mathrm{ns}]$", fontsize=18)
        ax.set_ylabel(r"Amplitude $[\mathrm{ADC}]$", fontsize=18)
        ax.set_title(
            f"Afterpulse candidate waveform from PMT {pmt_serial}",
            fontsize=20,
        )
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.tick_params(axis="both", which="minor", labelsize=13)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best", fontsize=12, frameon=True)

        fig.tight_layout()
        output_path = save_dir / (
            f"ap_candidate_wf_zoom_{i}_{save_name_suffix}.png"
        )
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


# ------------------------------------------------------------------
# Main analysis
# ------------------------------------------------------------------

def process_runs(gain_records):
    results = []

    for pmt_folder in PMT_FOLDERS:
        folder = BASE_PATH / pmt_folder
        print(f"\nChecking folder: {folder}")

        if not folder.is_dir():
            print("Folder does not exist, skipping.")
            continue

        subfolders = sorted(
            subfolder
            for subfolder in folder.iterdir()
            if subfolder.is_dir() and subfolder.name.startswith("APLV")
        )

        if not subfolders:
            print("No APLV folders found, skipping.")
            continue

        for run_folder in subfolders:
            run_name = run_folder.name
            filepattern = f"{run_name}_Module_0_*.root"
            matching_files = sorted(run_folder.glob(filepattern))

            if not matching_files:
                print(f"No ROOT files found for run: {run_name}")
                continue

            pmt_serial = extract_pmt_from_name(run_name) or "unknown"
            hv_num = extract_hv_from_name(run_name)
            lamp_v_num = extract_lamp_from_name(run_name)
            hv_v = format_hv(hv_num)
            lamp_v = format_lamp(lamp_v_num)

            target_lamp = TARGET_LAMP_PER_PMT.get(pmt_serial)
            if target_lamp is not None:
                if lamp_v_num is None:
                    print(f"Skipping {run_name}: no lamp voltage found.")
                    continue

                if abs(lamp_v_num - target_lamp) > 1e-6:
                    print(
                        f"Skipping {run_name}: found Lamp {lamp_v_num:g} Vpp, "
                        f"expected {target_lamp:g} Vpp for {pmt_serial}."
                    )
                    continue

            gain_match = select_gain_for_run(
                gain_records=gain_records,
                pmt_serial=pmt_serial,
                hv_v=hv_v,
            )

            if gain_match is None:
                print(
                    f"Skipping {run_name}: no gain result found for "
                    f"{pmt_serial}, HV {hv_v}."
                )
                continue

            pmt_gain = gain_match["gain"]

            print("\n================================================")
            print(f"Run analysis for PMT: {pmt_serial}")
            print(f"Lamp voltage: {lamp_v}")
            print(f"HV: {hv_v}")
            print(f"Gain: {pmt_gain:.3e} e ({gain_match['gain_source']})")
            print(f"Gain file: {gain_match['path']}")
            print(
                "Gain lamp target/result: "
                f"{format_lamp(TARGET_GAIN_LAMP_PER_PMT.get(pmt_serial))} / "
                f"{format_lamp(gain_match['lamp_v'])}"
            )
            print("Afterpulse finder: pmt_analysis.processing.afterpulses.AfterPulses")
            print("================================================")

            try:
                first_file = matching_files[0]
                with uproot.open(first_file) as root_file:
                    if not root_file.keys():
                        print(f"ROOT file contains no objects: {first_file}")
                        continue

                raw_data = ADCRawData(
                    raw_input_path=str(run_folder),
                    raw_input_filepattern=filepattern,
                )

                data = raw_data.get_branch_data(0)
                n_waveforms, n_samples_per_waveform = data.shape
                print(f"{run_name}: original waveform shape = {data.shape}")

                if n_samples_per_waveform != REQUIRED_N_SAMPLES_PER_WAVEFORM:
                    print(
                        f"Skipping {run_name}: waveform length is "
                        f"{n_samples_per_waveform} samples, required "
                        f"{REQUIRED_N_SAMPLES_PER_WAVEFORM} samples."
                    )
                    continue

                frame_type = f"{n_samples_per_waveform}_samples"
                data_full = data
                data_plot = data[:min(N_WAVEFORMS_PLOT, len(data))]

                run_save_dir = BASE_SAVE_DIR / pmt_serial / run_name
                run_save_dir.mkdir(parents=True, exist_ok=True)

                print(f"Using full data for statistics: {data_full.shape}")
                print(f"Using reduced data for plots: {data_plot.shape}")

                adc_area_to_e = raw_data.adc_area_to_e
                adc_f = raw_data.adc_f

                full_stats = compute_chunked_afterpulse_stats(
                    data_full=data_full,
                    adc_f=adc_f,
                    adc_area_to_e=adc_area_to_e,
                    pmt_gain=pmt_gain,
                    area_threshold_pe=AREA_THRESHOLD_PE,
                    time_threshold_ns=TIME_THRESHOLD_NS,
                    height=HEIGHT,
                    distance=DISTANCE,
                )

                results.append({
                    "pmt_serial": pmt_serial,
                    "run_name": run_name,
                    "hv_v": hv_v,
                    "lamp_v": lamp_v,
                    "n_samples": full_stats["n_samples"],
                    "n_main_pulses": full_stats["n_main_pulses"],
                    "gain": pmt_gain,
                    "gain_unc": gain_match["gain_err"],
                    "gain_source": gain_match["gain_source"],
                    "gain_result_file": str(gain_match["path"]),
                    "gain_result_hv_v": format_hv(gain_match["hv_v"]),
                    "gain_result_lamp_v": format_lamp(gain_match["lamp_v"]),
                    "target_gain_lamp_v": format_lamp(
                        TARGET_GAIN_LAMP_PER_PMT.get(pmt_serial)
                    ),
                    "frame_type": frame_type,
                    "n_samples_per_waveform": n_samples_per_waveform,
                    "n_ap": full_stats["n_ap"],
                    "n_ap_separable": full_stats["n_ap_separable"],
                    "ap_fraction": full_stats["ap_fraction"],
                    "ap_fraction_unc": full_stats["ap_fraction_unc"],
                    "ap_fraction_separable": full_stats["ap_fraction_separable"],
                    "ap_fraction_separable_unc": full_stats["ap_fraction_separable_unc"],
                    "mean_p0_area_pe": full_stats["mean_p0_area_pe"],
                    "sum_p0_area_pe": full_stats["sum_p0_area_pe"],
                    "ap_rate_per_pe": full_stats["ap_rate_per_pe"],
                    "ap_rate_per_pe_unc": full_stats["ap_rate_per_pe_unc"],
                    "ap_rate_per_pe_separable": full_stats["ap_rate_per_pe_separable"],
                    "ap_rate_per_pe_separable_unc": full_stats["ap_rate_per_pe_separable_unc"],
                    "n_ap_separable_above_thr": full_stats["n_ap_separable_above_thr"],
                    "n_ap_separable_below_thr": full_stats["n_ap_separable_below_thr"],
                    "ap_rate_per_pe_above_thr": full_stats["ap_rate_per_pe_above_thr"],
                    "ap_rate_per_pe_above_thr_unc": full_stats["ap_rate_per_pe_above_thr_unc"],
                    "ap_rate_per_pe_below_thr": full_stats["ap_rate_per_pe_below_thr"],
                    "ap_rate_per_pe_below_thr_unc": full_stats["ap_rate_per_pe_below_thr_unc"],
                })

                if len(data_plot) == 0:
                    print("No reduced data available for plots, skipping plots.")
                    continue

                ap_plot = make_afterpulses(
                    input_data=data_plot,
                    adc_f=adc_f,
                    adc_area_to_e=adc_area_to_e,
                    pmt_gain=pmt_gain,
                    area_threshold_pe=AREA_THRESHOLD_PE,
                    time_threshold_ns=TIME_THRESHOLD_NS,
                )

                ap_plot.find_ap(height=HEIGHT, distance=DISTANCE)
                ap_plot.constrain_main_peak(trim=True)

                if ap_plot.df.empty:
                    print("No plotting candidates found, skipping plots.")
                    continue

                ap_plot.get_ap_properties()
                if ap_plot.df.empty:
                    print(
                        "No plotting candidates left after get_ap_properties, "
                        "skipping plots."
                    )
                    continue

                ap_plot.multi_ap()
                ap_plot.ap_rate()

                if MAKE_ZOOM_WAVEFORM_PLOTS:
                    plot_zoom_afterpulse_waveforms(
                        df=ap_plot.df,
                        adc_f=adc_f,
                        save_dir=run_save_dir,
                        save_name_suffix=run_name,
                        pmt_serial=pmt_serial,
                        hv_v=hv_v,
                        lamp_v=lamp_v,
                    )

                try:
                    plotting_ap = PlottingAfterpulses(
                        df=ap_plot.df,
                        adc_f=adc_f,
                        ap_rate_dict=full_stats,
                        save_plots=True,
                        show_plots=False,
                        save_dir=str(run_save_dir),
                        save_name_suffix=run_name,
                        adc_area_to_e=adc_area_to_e,
                        gain=pmt_gain,
                        pmt_serial=pmt_serial,
                        hv_v=hv_v,
                        lamp_v=lamp_v,
                        frame_type=frame_type,
                        n_samples_per_waveform=n_samples_per_waveform,
                        n_waveforms_analyzed=len(data_plot),
                    )
                    plotting_ap.plot_essentials()
                except TypeError as exc:
                    print(
                        "Skipping standard PlottingAfterpulses plots because "
                        f"the installed plotting class has a different interface: {exc}"
                    )

            except Exception as exc:
                print("\n----------------------------------------")
                print(f"Error while processing run: {run_name}")
                print(f"Folder: {run_folder}")
                print(f"Reason: {exc}")
                print("Skipping this run.")
                print("----------------------------------------")

    return results


# ------------------------------------------------------------------
# Tables and summary plots
# ------------------------------------------------------------------

def create_summary_plots(results_df, gain_records):
    results_df["split_sum_diff"] = (
        results_df["ap_rate_per_pe_separable"]
        - results_df["ap_rate_per_pe_above_thr"]
        - results_df["ap_rate_per_pe_below_thr"]
    )

    print("\nFrame types found:")
    print(results_df[[
        "pmt_serial",
        "run_name",
        "frame_type",
        "n_samples_per_waveform",
    ]])

    print("\nGain files used:")
    print(results_df[[
        "pmt_serial",
        "hv_v",
        "gain",
        "gain_source",
        "gain_result_file",
    ]])

    print("\nSplit consistency check:")
    print(results_df[["pmt_serial", "run_name", "split_sum_diff"]])

    df_pmt = results_df.groupby("pmt_serial").agg({
        "ap_rate_per_pe_separable": ["mean", "std", "count"],
    }).reset_index()
    df_pmt.columns = [
        "pmt_serial",
        "ap_rate_per_pe_separable_mean",
        "ap_rate_per_pe_separable_std",
        "n_runs",
    ]
    df_pmt["ap_rate_per_pe_separable_std"] = (
        df_pmt["ap_rate_per_pe_separable_std"].fillna(0.0)
    )
    df_pmt["ap_rate_per_pe_separable_sem"] = (
        df_pmt["ap_rate_per_pe_separable_std"] / np.sqrt(df_pmt["n_runs"])
    )

    df_pmt_frame = results_df.groupby(["pmt_serial", "frame_type"]).agg({
        "ap_rate_per_pe_separable": ["mean", "std", "count"],
        "ap_rate_per_pe_above_thr": "mean",
        "ap_rate_per_pe_below_thr": "mean",
    }).reset_index()
    df_pmt_frame.columns = [
        "pmt_serial",
        "frame_type",
        "ap_rate_per_pe_separable_mean",
        "ap_rate_per_pe_separable_std",
        "n_runs",
        "ap_rate_per_pe_above_thr_mean",
        "ap_rate_per_pe_below_thr_mean",
    ]
    df_pmt_frame["ap_rate_per_pe_separable_std"] = (
        df_pmt_frame["ap_rate_per_pe_separable_std"].fillna(0.0)
    )
    df_pmt_frame["ap_rate_per_pe_separable_sem"] = (
        df_pmt_frame["ap_rate_per_pe_separable_std"]
        / np.sqrt(df_pmt_frame["n_runs"])
    ).fillna(0.0)

    df_pmt_split = results_df.groupby("pmt_serial").agg({
        "ap_rate_per_pe_separable": "mean",
        "ap_rate_per_pe_above_thr": "mean",
        "ap_rate_per_pe_below_thr": "mean",
    }).reset_index()

    results_df["frac_above_2pe"] = (
        results_df["ap_rate_per_pe_above_thr"]
        / results_df["ap_rate_per_pe_separable"]
    )
    df_frac = results_df.groupby("pmt_serial").agg({
        "frac_above_2pe": "mean",
    }).reset_index()

    summary_text_general = gain_lookup_summary(gain_records)
    summary_text_frame = gain_lookup_frame_summary(gain_records)

    # 1) PMT comparison bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(left=0.10, right=0.64, bottom=0.18, top=0.90)
    ax.bar(df_pmt["pmt_serial"], df_pmt["ap_rate_per_pe_separable_mean"])
    ax.set_ylabel(r"$R_{\mathrm{AP}}/\mathrm{PE}\ \mathrm{(separable)}$")
    ax.set_xlabel(r"$\mathrm{PMT}$")
    ax.set_title(r"$\mathrm{Afterpulse\ comparison\ per\ PMT}$", fontsize=18)
    ax.tick_params(axis="x", rotation=45)
    add_summary_box_right(fig, summary_text_general, x=0.67, y=0.88)
    fig.savefig(BASE_SAVE_DIR / "PMT_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2) PMT comparison errorbar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(left=0.10, right=0.64, bottom=0.18, top=0.90)
    ax.errorbar(
        df_pmt["pmt_serial"],
        df_pmt["ap_rate_per_pe_separable_mean"],
        yerr=df_pmt["ap_rate_per_pe_separable_sem"],
        fmt="o",
    )
    ax.set_ylabel(r"$R_{\mathrm{AP}}/\mathrm{PE}\ \mathrm{(separable)}$")
    ax.set_xlabel(r"$\mathrm{PMT}$")
    ax.set_title(r"$\mathrm{Afterpulse\ comparison\ per\ PMT}$", fontsize=18)
    ax.tick_params(axis="x", rotation=45)
    add_summary_box_right(fig, summary_text_general, x=0.67, y=0.88)
    fig.savefig(
        BASE_SAVE_DIR / "PMT_comparison_errorbars.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 3) Run-by-run scatter
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(left=0.10, right=0.64, bottom=0.18, top=0.90)
    for pmt in results_df["pmt_serial"].unique():
        df_sub = results_df[results_df["pmt_serial"] == pmt]
        ax.scatter(
            [pmt] * len(df_sub),
            df_sub["ap_rate_per_pe_separable"],
            label=pmt,
        )
    ax.set_ylabel(r"$R_{\mathrm{AP}}/\mathrm{PE}\ \mathrm{(separable)}$")
    ax.set_xlabel(r"$\mathrm{PMT}$")
    ax.set_title(r"$\mathrm{Run\!-\!by\!-\!run\ afterpulse\ rate\ per\ PE}$", fontsize=16)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(1.01, 0.02),
        borderaxespad=0.0,
        fontsize=10,
        frameon=True,
    )
    add_summary_box_right(fig, summary_text_general, x=0.67, y=0.88)
    fig.savefig(BASE_SAVE_DIR / "PMT_runs_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 4) Grouped split-threshold plot
    x = np.arange(len(df_pmt_split["pmt_serial"]))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(left=0.10, right=0.64, bottom=0.18, top=0.90)
    ax.bar(
        x - width,
        df_pmt_split["ap_rate_per_pe_separable"],
        width=width,
        label=r"$\mathrm{Total\ separable}$",
    )
    ax.bar(
        x,
        df_pmt_split["ap_rate_per_pe_above_thr"],
        width=width,
        label=rf"$A_{{\mathrm{{AP}}}} \geq {AREA_THRESHOLD_PE}\,\mathrm{{PE}}$",
    )
    ax.bar(
        x + width,
        df_pmt_split["ap_rate_per_pe_below_thr"],
        width=width,
        label=rf"$A_{{\mathrm{{AP}}}} < {AREA_THRESHOLD_PE}\,\mathrm{{PE}}$",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df_pmt_split["pmt_serial"], rotation=45)
    ax.set_ylabel(r"$R_{\mathrm{AP}}/\mathrm{PE}$")
    ax.set_xlabel(r"$\mathrm{PMT}$")
    ax.set_title(r"$\mathrm{Afterpulse\ rate\ split\ by\ threshold}$", fontsize=16)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(1.01, 0.02),
        borderaxespad=0.0,
        fontsize=10,
        frameon=True,
    )
    add_summary_box_right(fig, summary_text_general, x=0.67, y=0.88)
    fig.savefig(
        BASE_SAVE_DIR / "PMT_comparison_split_threshold.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 5) Stacked threshold plot
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(left=0.10, right=0.64, bottom=0.18, top=0.90)
    ax.bar(
        df_pmt_split["pmt_serial"],
        df_pmt_split["ap_rate_per_pe_below_thr"],
        label=rf"$A_{{\mathrm{{AP}}}} < {AREA_THRESHOLD_PE}\,\mathrm{{PE}}$",
    )
    ax.bar(
        df_pmt_split["pmt_serial"],
        df_pmt_split["ap_rate_per_pe_above_thr"],
        bottom=df_pmt_split["ap_rate_per_pe_below_thr"],
        label=rf"$A_{{\mathrm{{AP}}}} \geq {AREA_THRESHOLD_PE}\,\mathrm{{PE}}$",
    )
    ax.set_ylabel(r"$R_{\mathrm{AP}}/\mathrm{PE}$")
    ax.set_xlabel(r"$\mathrm{PMT}$")
    ax.set_title(r"$\mathrm{Stacked\ afterpulse\ rate\ per\ PE}$", fontsize=16)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(1.01, 0.02),
        borderaxespad=0.0,
        fontsize=10,
        frameon=True,
    )
    add_summary_box_right(fig, summary_text_general, x=0.67, y=0.88)
    fig.savefig(
        BASE_SAVE_DIR / "PMT_comparison_stacked_threshold.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 6) Fraction above 2 PE
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(left=0.10, right=0.64, bottom=0.18, top=0.90)
    ax.bar(df_frac["pmt_serial"], df_frac["frac_above_2pe"])
    ax.set_ylabel(rf"$f(A_{{\mathrm{{AP}}}} \geq {AREA_THRESHOLD_PE}\,\mathrm{{PE}})$")
    ax.set_xlabel(r"$\mathrm{PMT}$")
    ax.set_title(r"$\mathrm{Large\!-\!afterpulse\ fraction\ per\ PMT}$", fontsize=16)
    ax.tick_params(axis="x", rotation=45)
    add_summary_box_right(fig, summary_text_general, x=0.67, y=0.88)
    fig.savefig(
        BASE_SAVE_DIR / "PMT_fraction_above_2pe.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 7) Frame-separated comparison plot
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(left=0.10, right=0.64, bottom=0.18, top=0.90)
    for frame in sorted(df_pmt_frame["frame_type"].unique()):
        df_sub = df_pmt_frame[df_pmt_frame["frame_type"] == frame]
        ax.errorbar(
            df_sub["pmt_serial"],
            df_sub["ap_rate_per_pe_separable_mean"],
            yerr=df_sub["ap_rate_per_pe_separable_sem"],
            fmt="o",
            label=frame,
        )
    ax.set_ylabel(r"$R_{\mathrm{AP}}/\mathrm{PE}\ \mathrm{(separable)}$")
    ax.set_xlabel(r"$\mathrm{PMT}$")
    ax.set_title(r"$\mathrm{Afterpulse\ rate\ per\ PE:\ separated\ by\ frame\ size}$", fontsize=16)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(1.01, 0.02),
        borderaxespad=0.0,
        fontsize=10,
        frameon=True,
    )
    add_summary_box_right(fig, summary_text_frame, x=0.67, y=0.88)
    fig.savefig(
        BASE_SAVE_DIR / "PMT_comparison_frame_types.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 8) Threshold split per frame type
    for frame in sorted(df_pmt_frame["frame_type"].unique()):
        df_sub = df_pmt_frame[df_pmt_frame["frame_type"] == frame].copy()
        x = np.arange(len(df_sub["pmt_serial"]))
        width = 0.35
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.subplots_adjust(left=0.10, right=0.64, bottom=0.18, top=0.90)
        ax.bar(
            x - width / 2,
            df_sub["ap_rate_per_pe_above_thr_mean"],
            width=width,
            label=rf"$A_{{\mathrm{{AP}}}} \geq {AREA_THRESHOLD_PE}\,\mathrm{{PE}}$",
        )
        ax.bar(
            x + width / 2,
            df_sub["ap_rate_per_pe_below_thr_mean"],
            width=width,
            label=rf"$A_{{\mathrm{{AP}}}} < {AREA_THRESHOLD_PE}\,\mathrm{{PE}}$",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(df_sub["pmt_serial"], rotation=45)
        ax.set_ylabel(r"$R_{\mathrm{AP}}/\mathrm{PE}$")
        ax.set_xlabel(r"$\mathrm{PMT}$")
        ax.set_title(
            rf"$\mathrm{{Afterpulse\ split\ by\ threshold}}\ ({frame})$",
            fontsize=16,
        )
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(1.01, 0.02),
            borderaxespad=0.0,
            fontsize=10,
            frameon=True,
        )
        add_summary_box_right(fig, summary_text_frame, x=0.67, y=0.88)
        fig.savefig(
            BASE_SAVE_DIR / f"PMT_split_threshold_{frame}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    print("\nBest PMT:")
    print(df_pmt.loc[df_pmt["ap_rate_per_pe_separable_mean"].idxmin()])

    print("\nBest PMT above 2 PE:")
    print(df_pmt_split.loc[df_pmt_split["ap_rate_per_pe_above_thr"].idxmin()])

    print("\nBest PMT below 2 PE:")
    print(df_pmt_split.loc[df_pmt_split["ap_rate_per_pe_below_thr"].idxmin()])

    for frame in sorted(df_pmt_frame["frame_type"].unique()):
        df_sub = df_pmt_frame[df_pmt_frame["frame_type"] == frame]
        best_frame = df_sub.loc[
            df_sub["ap_rate_per_pe_separable_mean"].idxmin()
        ]
        print(f"\nBest PMT for {frame}:")
        print(best_frame[[
            "pmt_serial",
            "ap_rate_per_pe_separable_mean",
            "ap_rate_per_pe_separable_sem",
            "ap_rate_per_pe_above_thr_mean",
            "ap_rate_per_pe_below_thr_mean",
        ]])


def main():
    gain_records = load_gain_results()

    if not gain_records:
        raise RuntimeError(
            "No gain results were loaded. Check GAIN_RESULTS_DIRS and make sure "
            "your gain scripts have produced *_results.json files."
        )

    results = process_runs(gain_records)
    results_df = pd.DataFrame(results)
    summary_csv = BASE_SAVE_DIR / "afterpulse_summary.csv"
    results_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary table: {summary_csv}")

    if results_df.empty:
        raise RuntimeError("No afterpulse runs were processed successfully.")

    create_summary_plots(results_df, gain_records)


if __name__ == "__main__":
    main()