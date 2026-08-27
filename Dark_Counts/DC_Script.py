#!/usr/bin/env python3
"""
dark_count_analysis_xdet.py

Analyze XDet dark-count pickle files.

The script is made for the files in

    /disk/gfs_atp/lhoetz/marmotx/Xdet_DCrates

with names such as

    raw_data_XDetector_DC_OR_800V_Module_0_0.pkl
    hit_data_XDetector_DC_OR_800V_Module_0_0.pkl

and

    raw_data_XDet_vacuum_800V_matchedThresholdOnFanInOut0mv_ORTrigger_Module_0_0.pkl
    hit_data_XDet_vacuum_800V_matchedThresholdOnFanInOut0mv_ORTrigger_Module_0_0.pkl

It saves dark-count spectra, thesis example waveform plots, area-selection
example plots, comparison plots, and CSV tables with hit counts and rates.
Rates in Hz are only computed when a run duration is available from a
Summary_*.txt file or from --run-duration.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

# Folder containing the raw_data_*.pkl and hit_data_*.pkl files.
DATA_DIR = Path("/disk/gfs_atp/lhoetz/marmotx/Xdet_DCrates")

# Folder where all plots, text summaries, and CSV result tables are saved.
SAVE_DIR = Path(__file__).resolve().parent / "Dark_Counts_Results"

# Default command-line values. Usually you only need to change DATA_DIR and
# SAVE_DIR above.
DEFAULT_DATA_DIR = DATA_DIR
DEFAULT_OUTPUT_DIR = SAVE_DIR

# Thesis example waveform plots are produced automatically. Set this to False
# only if you want a faster run that skips loading raw_data_*.pkl.
MAKE_WAVEFORM_EXAMPLE_PLOTS = True
N_WAVEFORM_EXAMPLES = 5

# Hit-area spectra keep the same displayed ADC window as the original plots,
# but the axis values are converted to PE for readability.
HIT_AREA_PLOT_MIN_ADC = -500.0
HIT_AREA_PLOT_MAX_ADC = 5000.0
HIT_AREA_SPECTRUM_BINS = 101
HIT_AREA_SELECTION_EXAMPLE_BINS = 111

# Physical PMT labels for the two digitizer channels.
CHANNEL_LABELS = {
    0: "LV2483",
    1: "LV2480",
}

# Gain-result JSON files used to convert dark-count hit areas from ADC samples
# to photoelectrons.
GAIN_RESULTS_DIRS = [
    Path("/home/uzh/mauett/PMT_Analysis_Mauett/Gain/Gain_Plots/One_Plot_ErrorMI"),
]

# ADC-area conversion factor used by the V1730D digitizer analysis.
ADC_AREA_TO_E = 3047.6
TARGET_GAIN_HV = 800
LAMP_TOL = 1e-3

# Target lamp settings used for the gain files at 800 V. This avoids picking a
# wrong gain file if several lamp voltages exist for the same PMT and HV.
TARGET_GAIN_LAMP_PER_PMT = {
    "LV2480": 1.90,
    "LV2481": 1.95,
    "LV2483": 1.95,
    "LV2485": 1.95,
}

# Optional manual fallback if the gain JSONs are not available.
MANUAL_GAINS_E = {
    # "LV2480": 4.24e6,
    # "LV2483": 4.06e6,
}

# Plot style matched to the gain-summary figures.
GAIN_STYLE_COLORS = [
    "mediumseagreen",
    "dodgerblue",
    "slateblue",
    "darkmagenta",
]
PMT_COLORS = {
    "LV2483": "mediumseagreen",
    "LV2480": "dodgerblue",
    "LV2481": "slateblue",
    "LV2485": "darkmagenta",
}
GAIN_STYLE_GRID_ALPHA = 0.28
GAIN_STYLE_LEGEND_KWARGS = {
    "frameon": True,
    "framealpha": 0.92,
    "edgecolor": "0.75",
    "fontsize": 12,
}

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 20,
    "axes.labelsize": 17,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.titlesize": 20,
    "axes.linewidth": 1.0,
})

DATASETS = {
    "air": {
        "label": "XDetector DC OR 800 V",
        "filename": "XDetector_DC_OR_800V_Module_0_0",
        "summary_files": [
            "/disk/gfs_atp/lhoetz/marmotx/XDetector_DC_OR_800V/"
            "Summary_XDetector_DC_OR_800V.txt",
        ],
        "area_min": 300.0,
        "area_max": 3500.0,
    },
    "vacuum": {
        "label": "XDet vacuum 800 V",
        "filename": "XDet_vacuum_800V_matchedThresholdOnFanInOut0mv_ORTrigger_Module_0_0",
        "summary_files": [
            "/home/atp/lhoetz/ln_gfsatp/lhoetz/marmotx/data_20260428/"
            "XDet_vacuum_800V_matchedThresholdOnFanInOut0mv_ORTrigger/"
            "Summary_XDet_vacuum_800V_matchedThresholdOnFanInOut0mv_ORTrigger.txt",
        ],
        "area_min": 300.0,
        "area_max": 3500.0,
    },
}


def gain_style_color(index: int) -> str:
    return GAIN_STYLE_COLORS[index % len(GAIN_STYLE_COLORS)]


def channel_label(channel: int) -> str:
    return CHANNEL_LABELS.get(int(channel), f"PMT {int(channel)}")


def pmt_color(pmt: str, fallback_index: int = 0) -> str:
    return PMT_COLORS.get(str(pmt), gain_style_color(fallback_index))


def channel_color(channel: int) -> str:
    return pmt_color(channel_label(channel), int(channel))


def finite_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def normalized_name(name: str) -> str:
    return str(name).replace("-", "_")


def extract_pmt_from_name(name: str) -> str | None:
    match = re.search(r"(LV\d{4})", str(name))
    return match.group(1) if match else None


def extract_lamp_vpp_from_name(name: str) -> float | None:
    match = re.search(
        r"Lamp_?([0-9]+(?:\.[0-9]+)?)_?Vpp",
        normalized_name(name),
        re.IGNORECASE,
    )
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def extract_hv_from_name(name: str) -> int | None:
    matches = re.findall(
        r"(?<![A-Za-z0-9.])(-?\d{3,4})V(?!pp)",
        str(name),
        re.IGNORECASE,
    )
    if not matches:
        matches = re.findall(
            r"(?<![A-Za-z0-9.])(-?\d{3,4})V(?!pp)",
            normalized_name(name),
            re.IGNORECASE,
        )
    if not matches:
        return None
    try:
        return abs(int(matches[-1]))
    except ValueError:
        return None


def read_gain_result(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r") as file:
            data = json.load(file)
    except Exception:
        return None

    search_name = "_".join(
        [
            path.stem,
            path.parent.name,
            path.parent.parent.name if path.parent.parent else "",
        ]
    )
    pmt = extract_pmt_from_name(search_name)
    hv = extract_hv_from_name(search_name)
    lamp = extract_lamp_vpp_from_name(search_name)

    model_dep = data.get("model_dependent_fit", {})
    params = model_dep.get("params", {}) if isinstance(model_dep, dict) else {}
    fit_success = bool(model_dep.get("fit_success", False)) if isinstance(model_dep, dict) else False

    if not fit_success:
        return None

    gain_e = finite_float(params.get("gain_e"))
    gain_e_err = finite_float(params.get("gain_e_err"))
    if gain_e is None or gain_e <= 0:
        return None

    return {
        "path": path,
        "pmt": pmt,
        "hv": hv,
        "lamp": lamp,
        "gain_e": gain_e,
        "gain_e_err": gain_e_err,
        "source": "model_dependent",
        "mtime": path.stat().st_mtime,
    }


def load_gain_records() -> list[dict[str, Any]]:
    records = []
    for base_dir in GAIN_RESULTS_DIRS:
        if not base_dir.is_dir():
            print(f"Gain directory not found: {base_dir}")
            continue
        for path in sorted(base_dir.rglob("*_results.json")):
            record = read_gain_result(path)
            if record is not None:
                records.append(record)
    print(f"Loaded {len(records)} gain result record(s).")
    return records


def select_gain(gain_records: list[dict[str, Any]], pmt: str) -> dict[str, Any] | None:
    candidates = [
        record for record in gain_records
        if record.get("pmt") == pmt
        and record.get("hv") is not None
        and abs(int(record["hv"])) == TARGET_GAIN_HV
        and record.get("gain_e") is not None
    ]

    target_lamp = TARGET_GAIN_LAMP_PER_PMT.get(pmt)
    if target_lamp is not None:
        candidates = [
            record for record in candidates
            if record.get("lamp") is not None
            and abs(float(record["lamp"]) - target_lamp) <= LAMP_TOL
        ]

    if not candidates and pmt in MANUAL_GAINS_E:
        return {
            "gain_e": float(MANUAL_GAINS_E[pmt]),
            "gain_e_err": None,
            "source": "manual",
            "path": None,
            "n_files": 1,
            "lamp_values": [],
        }

    if not candidates:
        return None

    gains = np.array([record["gain_e"] for record in candidates], dtype=float)
    errors = np.array(
        [
            record["gain_e_err"]
            if record.get("gain_e_err") is not None and record["gain_e_err"] > 0
            else np.nan
            for record in candidates
        ],
        dtype=float,
    )

    if gains.size == 1:
        gain_e = float(gains[0])
        gain_e_err = None if not np.isfinite(errors[0]) else float(errors[0])
    elif np.all(np.isfinite(errors) & (errors > 0)):
        weights = 1.0 / errors**2
        gain_e = float(np.sum(weights * gains) / np.sum(weights))
        gain_e_err = float(np.sqrt(1.0 / np.sum(weights)))
    else:
        gain_e = float(np.median(gains))
        gain_e_err = float(np.std(gains, ddof=1) / np.sqrt(gains.size))

    return {
        "gain_e": gain_e,
        "gain_e_err": gain_e_err,
        "source": "model_dependent_target_lamp",
        "path": candidates[0]["path"],
        "n_files": len(candidates),
        "lamp_values": sorted({
            round(float(record["lamp"]), 6)
            for record in candidates
            if record.get("lamp") is not None
        }),
    }


def require_gain(gain_records: list[dict[str, Any]], pmt: str) -> dict[str, Any]:
    gain = select_gain(gain_records, pmt)
    if gain is None:
        raise RuntimeError(
            f"No {TARGET_GAIN_HV} V gain result found for {pmt}. "
            "Check GAIN_RESULTS_DIRS or add a value to MANUAL_GAINS_E."
        )
    return gain


def area_adc_to_pe(area_adc: Any, gain_e: float) -> np.ndarray:
    return np.asarray(area_adc, dtype=float) * ADC_AREA_TO_E / float(gain_e)


def dataset_display_name(dataset_name: str) -> str:
    names = {
        "air": "Air",
        "vacuum": "Vacuum",
    }
    return names.get(str(dataset_name), str(dataset_name).replace("_", " ").title())


def apply_gain_style(ax, which="both", axis="both") -> None:
    ax.grid(True, which=which, axis=axis, alpha=GAIN_STYLE_GRID_ALPHA)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def legend_inside(ax):
    legend = ax.legend(
        loc="best",
        **GAIN_STYLE_LEGEND_KWARGS,
    )
    legend.get_frame().set_linewidth(1.0)
    return legend


def figure_legend_below_axes(fig, axes):
    handles = []
    labels = []

    for ax in np.ravel(axes):
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label and label not in labels:
                handles.append(handle)
                labels.append(label)

    if not handles:
        return None

    legend_kwargs = dict(GAIN_STYLE_LEGEND_KWARGS)
    legend_kwargs.update({
        "fontsize": 9.0,
        "borderpad": 0.30,
        "labelspacing": 0.25,
        "handlelength": 1.8,
        "handletextpad": 0.45,
        "columnspacing": 0.85,
    })

    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.01),
        ncol=min(len(labels), 4),
        **legend_kwargs,
    )
    legend.get_frame().set_linewidth(0.8)
    return legend


def record_get(record: Any, field: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(field, default)

    try:
        return record[field]
    except Exception:
        return default


def raw_fields(raw_data: Any) -> list[str]:
    if raw_data is None:
        return []
    if hasattr(raw_data, "dtype") and raw_data.dtype.names:
        return list(raw_data.dtype.names)
    if len(raw_data) > 0 and isinstance(raw_data[0], dict):
        return list(raw_data[0].keys())
    return []


def load_raw_data(data_dir: Path, filename: str):
    path = data_dir / f"raw_data_{filename}.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing raw-data pickle: {path}")

    with path.open("rb") as file:
        return pickle.load(file), path


def load_hit_data(data_dir: Path, filename: str) -> tuple[pd.DataFrame, Path]:
    path = data_dir / f"hit_data_{filename}.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing hit-data pickle: {path}")

    return pd.read_pickle(path), path


def read_run_duration(summary_file: Path) -> float:
    with summary_file.open("r") as file:
        for line in file:
            if "Total Measuring time" in line:
                return float(line.split(":")[1].strip().split()[0])

    raise ValueError(f"Could not find 'Total Measuring time' in {summary_file}")


def summary_candidates(data_dir: Path, config: dict, explicit_summary: str | None) -> list[Path]:
    candidates = []

    if explicit_summary:
        candidates.append(Path(explicit_summary))

    for value in config.get("summary_files", []):
        candidates.append(Path(value))

    candidates.extend(sorted(data_dir.glob("Summary*.txt")))

    seen = set()
    unique = []
    for path in candidates:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)

    return unique


def infer_duration_from_values(values: np.ndarray, name: str) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size < 2:
        return None

    span = float(np.nanmax(values) - np.nanmin(values))
    if not np.isfinite(span) or span <= 0:
        return None

    lowered = name.lower()
    if "ns" in lowered:
        return span / 1e9
    if "us" in lowered:
        return span / 1e6
    if "ms" in lowered:
        return span / 1e3
    if lowered.endswith("_s") or "seconds" in lowered:
        return span

    return None


def infer_run_duration(raw_data: Any, df: pd.DataFrame) -> tuple[float | None, str | None]:
    time_columns = [
        "time_s",
        "timestamp_s",
        "event_time_s",
        "trigger_time_s",
        "time_ns",
        "timestamp_ns",
        "event_time_ns",
        "trigger_time_ns",
        "time_us",
        "timestamp_us",
        "event_time_us",
        "trigger_time_us",
    ]

    for column in time_columns:
        if column in df.columns:
            duration = infer_duration_from_values(df[column].to_numpy(), column)
            if duration is not None:
                return duration, f"hit column {column}"

    fields = raw_fields(raw_data)
    if not fields:
        return None, None

    for field in time_columns:
        if field not in fields:
            continue
        try:
            values = np.asarray([record_get(row, field) for row in raw_data], dtype=float)
        except Exception:
            continue
        duration = infer_duration_from_values(values, field)
        if duration is not None:
            return duration, f"raw-data field {field}"

    return None, None


def resolve_run_duration(
    data_dir: Path,
    config: dict,
    raw_data: Any,
    df: pd.DataFrame,
    explicit_summary: str | None,
    explicit_duration: float | None,
) -> tuple[float | None, str]:
    if explicit_duration is not None:
        return float(explicit_duration), "command line --run-duration"

    for summary_file in summary_candidates(data_dir, config, explicit_summary):
        if not summary_file.is_file():
            continue
        try:
            return read_run_duration(summary_file), str(summary_file)
        except Exception as exc:
            print(f"WARNING: could not read duration from {summary_file}: {exc}")

    inferred, source = infer_run_duration(raw_data, df)
    if inferred is not None:
        return inferred, source or "inferred"

    return None, "not available"


def print_data_overview(dataset_name: str, raw_data: Any, df: pd.DataFrame) -> None:
    print(f"\n=== {dataset_name} ===")
    print(f"Raw events: {len(raw_data) if raw_data is not None else 'not loaded'}")
    print(f"Hit rows: {len(df)}")

    fields = raw_fields(raw_data)
    print("\nRaw-data fields:")
    for key in fields:
        print(f"  {key}")

    print("\nHit-data columns:")
    for column in df.columns:
        print(f"  {column}")

    print("\nFirst rows of hit data:")
    print(df.head())


def n_channels_from_data(raw_data: Any, df: pd.DataFrame) -> int:
    if raw_data is not None and len(raw_data) > 0:
        wfs = record_get(raw_data[0], "wfs")
        if wfs is not None:
            return int(np.asarray(wfs).shape[0])

    if "channel" in df.columns and len(df) > 0:
        return int(pd.to_numeric(df["channel"], errors="coerce").max()) + 1

    return 1


def n_samples_from_data(raw_data: Any) -> int:
    if raw_data is None:
        return 0
    if len(raw_data) == 0:
        return 0
    wfs = record_get(raw_data[0], "wfs")
    if wfs is None:
        return 0
    return int(np.asarray(wfs).shape[-1])


def plot_example_waveforms(
    dataset_name: str,
    raw_data: Any,
    df: pd.DataFrame,
    output_dir: Path,
    n_events: int = 5,
    n_sigma_hitfinder_threshold: float = 5.0,
) -> None:
    n_channels = n_channels_from_data(raw_data, df)
    n_samples = n_samples_from_data(raw_data)

    if n_samples <= 0:
        print("WARNING: no waveform field found, skipping waveform plots.")
        return

    display_name = dataset_display_name(dataset_name)

    for i, record in enumerate(raw_data[:n_events]):
        event_index = record_get(record, "event_index", i)

        fig, axs = plt.subplots(
            figsize=(5.8 * n_channels, 4.6),
            ncols=n_channels,
            nrows=1,
            squeeze=False,
        )
        axs = axs.flatten()

        for ch in range(n_channels):
            waveform_raw = record_get(record, "wfs_raw")
            if waveform_raw is None:
                waveform_raw = record_get(record, "wfs")
            waveform_raw = np.asarray(waveform_raw)

            axs[ch].step(
                np.arange(n_samples),
                waveform_raw[ch],
                where="post",
                color=channel_color(ch),
                linewidth=1.0,
            )
            axs[ch].set_xlabel("sample number [2 ns]")
            axs[ch].set_ylabel("ADC")
            axs[ch].set_title(channel_label(ch))
            apply_gain_style(axs[ch])

        fig.suptitle(f"{display_name}: Raw waveforms")
        fig.tight_layout()
        out_path = output_dir / f"{dataset_name}_raw_waveforms_event_{i:03d}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        fig, axs = plt.subplots(
            figsize=(5.8 * n_channels, 4.8),
            ncols=n_channels,
            nrows=1,
            squeeze=False,
        )
        axs = axs.flatten()

        waveforms = np.asarray(record_get(record, "wfs"))
        baseline_rms = record_get(record, "baseline_rms")
        baseline_rms = np.asarray(baseline_rms) if baseline_rms is not None else None

        for ch in range(n_channels):
            axs[ch].step(
                np.arange(n_samples),
                waveforms[ch],
                where="post",
                color=channel_color(ch),
                linewidth=1.0,
            )
            axs[ch].axhline(
                0,
                linestyle="--",
                color="0.55",
                linewidth=1.0,
                label="baseline",
            )

            if baseline_rms is not None:
                threshold = baseline_rms[ch] * n_sigma_hitfinder_threshold
                axs[ch].axhline(
                    threshold,
                    linestyle="--",
                    color="darkmagenta",
                    linewidth=1.2,
                    label="pulse finder threshold",
                )

            if {"channel", "event_index"}.issubset(df.columns):
                hits = df[(df["channel"] == ch) & (df["event_index"] == event_index)]
            else:
                hits = pd.DataFrame()
            for j, hit in enumerate(hits.to_dict("records")):
                label_start = "pulse start/end" if j == 0 else None
                label_window = "pulse integration window" if j == 0 else None

                if "hit_start" in hit:
                    axs[ch].axvline(
                        hit["hit_start"],
                        color="slateblue",
                        linestyle="--",
                        linewidth=1.2,
                        label=label_start,
                    )
                if "hit_end" in hit:
                    axs[ch].axvline(
                        hit["hit_end"],
                        color="slateblue",
                        linestyle="--",
                        linewidth=1.2,
                    )
                if "hit_left_bound" in hit and "hit_right_bound" in hit:
                    axs[ch].axvspan(
                        hit["hit_left_bound"],
                        hit["hit_right_bound"],
                        color="dodgerblue",
                        alpha=0.22,
                        label=label_window,
                    )

            axs[ch].set_xlabel("sample number [2 ns]")
            axs[ch].set_ylabel("baseline-corrected ADC")
            axs[ch].set_title(channel_label(ch))
            apply_gain_style(axs[ch])

        fig.suptitle(f"{display_name}: Baseline-corrected waveforms", y=0.98)
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.28, top=0.84, wspace=0.28)
        figure_legend_below_axes(fig, axs)
        out_path = output_dir / f"{dataset_name}_corrected_waveforms_event_{i:03d}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_spectra_and_compute_rates(
    dataset_name: str,
    df: pd.DataFrame,
    run_duration: float | None,
    n_channels: int,
    area_min: float,
    area_max: float,
    output_dir: Path,
    gain_records: list[dict[str, Any]],
) -> pd.DataFrame:
    display_name = dataset_display_name(dataset_name)
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.15, top=0.88)

    if "hit_area_raw" not in df.columns:
        raise KeyError("hit_data dataframe has no 'hit_area_raw' column.")
    if "channel" not in df.columns:
        raise KeyError("hit_data dataframe has no 'channel' column.")

    channel_plot_data = []
    for ch in range(n_channels):
        pmt = channel_label(ch)
        gain = require_gain(gain_records, pmt)
        channel_hits = df[df["channel"] == ch].copy()
        areas_adc = pd.to_numeric(channel_hits["hit_area_raw"], errors="coerce")
        areas_adc = areas_adc[np.isfinite(areas_adc)]
        areas_pe = area_adc_to_pe(areas_adc, gain["gain_e"])
        area_min_pe = float(area_adc_to_pe(area_min, gain["gain_e"]))
        area_max_pe = float(area_adc_to_pe(area_max, gain["gain_e"]))
        channel_plot_data.append({
            "channel": ch,
            "pmt": pmt,
            "gain": gain,
            "areas_adc": areas_adc,
            "areas_pe": areas_pe,
            "area_min_pe": area_min_pe,
            "area_max_pe": area_max_pe,
        })

    plot_min_pe = min(
        float(area_adc_to_pe(HIT_AREA_PLOT_MIN_ADC, item["gain"]["gain_e"]))
        for item in channel_plot_data
    )
    plot_max_pe = max(
        float(area_adc_to_pe(HIT_AREA_PLOT_MAX_ADC, item["gain"]["gain_e"]))
        for item in channel_plot_data
    )
    selection_min_pe = min(item["area_min_pe"] for item in channel_plot_data)
    selection_max_pe = max(item["area_max_pe"] for item in channel_plot_data)
    bins = np.linspace(plot_min_pe, plot_max_pe, HIT_AREA_SPECTRUM_BINS)

    for item in channel_plot_data:
        ch = item["channel"]
        color = pmt_color(item["pmt"], ch)
        ax.hist(
            item["areas_pe"],
            bins=bins,
            histtype="step",
            density=False,
            linewidth=1.8,
            color=color,
            label=item["pmt"],
        )

    ax.axvspan(
        selection_min_pe,
        selection_max_pe,
        color="lightgrey",
        alpha=0.55,
        label="area selection",
    )

    ax.set_yscale("log")
    ax.set_xlim(plot_min_pe, plot_max_pe)
    ax.set_xlabel(r"Hit area $A_{\mathrm{hit}}\,[\mathrm{PE}]$")
    ax.set_ylabel("Counts")
    ax.set_title(f"{display_name}: Dark-count hit-area spectrum in PE")
    apply_gain_style(ax, which="both")
    legend_inside(ax)

    out_path = output_dir / f"{dataset_name}_hit_area_spectrum.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for item in channel_plot_data:
        ch = item["channel"]
        channel_hits = df[df["channel"] == ch]
        hit_area_adc = pd.to_numeric(channel_hits["hit_area_raw"], errors="coerce")
        selected_hits = channel_hits[
            (hit_area_adc > area_min)
            & (hit_area_adc < area_max)
        ]

        n_hits_raw = int(np.count_nonzero(np.isfinite(hit_area_adc)))
        n_hits_area_selected = int(len(selected_hits))

        if run_duration is not None and run_duration > 0:
            dc_rate_raw = n_hits_raw / run_duration
            dc_rate_area_selected = n_hits_area_selected / run_duration
        else:
            dc_rate_raw = np.nan
            dc_rate_area_selected = np.nan

        rows.append({
            "dataset": dataset_name,
            "channel": ch,
            "channel_label": channel_label(ch),
            "gain_e": item["gain"]["gain_e"],
            "gain_e_err": item["gain"]["gain_e_err"],
            "gain_source": item["gain"]["source"],
            "gain_file": "" if item["gain"]["path"] is None else str(item["gain"]["path"]),
            "adc_area_to_e": ADC_AREA_TO_E,
            "area_min_adc_samples": area_min,
            "area_max_adc_samples": area_max,
            "area_min_pe": item["area_min_pe"],
            "area_max_pe": item["area_max_pe"],
            "run_duration_s": run_duration,
            "n_hits_raw": n_hits_raw,
            "n_hits_area_selected": n_hits_area_selected,
            "dc_rate_raw_hz": dc_rate_raw,
            "dc_rate_area_selected_hz": dc_rate_area_selected,
        })

    return pd.DataFrame(rows)


def plot_thesis_area_selection_example(
    dataset_name: str,
    df: pd.DataFrame,
    area_min: float,
    area_max: float,
    output_dir: Path,
    gain_records: list[dict[str, Any]],
) -> None:
    """
    Save a thesis-oriented example showing exactly how the pulse-area cut is
    applied for one representative channel.
    """
    if "hit_area_raw" not in df.columns or "channel" not in df.columns or df.empty:
        return

    display_name = dataset_display_name(dataset_name)
    channel_counts = df["channel"].value_counts()
    example_channel = int(channel_counts.index[0])
    pmt = channel_label(example_channel)
    gain = require_gain(gain_records, pmt)
    channel_hits = df[df["channel"] == example_channel].copy()
    areas_adc = pd.to_numeric(channel_hits["hit_area_raw"], errors="coerce").dropna()

    if areas_adc.empty:
        return

    areas_pe = area_adc_to_pe(areas_adc, gain["gain_e"])
    area_min_pe = float(area_adc_to_pe(area_min, gain["gain_e"]))
    area_max_pe = float(area_adc_to_pe(area_max, gain["gain_e"]))
    selected = areas_pe[(areas_adc > area_min) & (areas_adc < area_max)]

    plot_min_pe = float(area_adc_to_pe(HIT_AREA_PLOT_MIN_ADC, gain["gain_e"]))
    plot_max_pe = float(area_adc_to_pe(HIT_AREA_PLOT_MAX_ADC, gain["gain_e"]))
    bins = np.linspace(plot_min_pe, plot_max_pe, HIT_AREA_SELECTION_EXAMPLE_BINS)

    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.15, top=0.88)
    example_color = pmt_color(pmt, example_channel)

    ax.hist(
        areas_pe,
        bins=bins,
        histtype="step",
        linewidth=2.0,
        color=example_color,
        label="all dark-count candidates",
    )
    ax.hist(
        selected,
        bins=bins,
        histtype="step",
        linewidth=2.0,
        linestyle="--",
        color=example_color,
        label="used for rate calculation",
    )

    ax.axvspan(
        area_min_pe,
        area_max_pe,
        color=example_color,
        alpha=0.16,
        label="accepted area window",
    )
    ax.axvline(area_min_pe, color="slateblue", linestyle="--", linewidth=1.5)
    ax.axvline(area_max_pe, color="slateblue", linestyle="--", linewidth=1.5)

    ax.set_yscale("log")
    ax.set_xlim(plot_min_pe, plot_max_pe)
    ax.set_xlabel(r"Hit area $A_{\mathrm{hit}}\,[\mathrm{PE}]$")
    ax.set_ylabel("Counts")
    ax.set_title(
        f"{display_name}: Dark-count area selection example, "
        f"{pmt}"
    )
    apply_gain_style(ax, which="both")
    legend_inside(ax)

    info_text = (
        rf"$N_{{\mathrm{{all}}}} = {len(areas_adc)}$" "\n"
        rf"$N_{{\mathrm{{selected}}}} = {len(selected)}$" "\n"
        rf"${area_min_pe:.2f} < A_{{\mathrm{{hit}}}} < {area_max_pe:.2f}\,\mathrm{{PE}}$" "\n"
        rf"$({area_min:.0f} < A_{{\mathrm{{hit}}}} < {area_max:.0f}\,\mathrm{{ADC}})$"
    )
    ax.text(
        0.03,
        0.94,
        info_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        bbox=dict(facecolor="white", edgecolor="0.75", alpha=0.90),
    )

    out_path = output_dir / f"{dataset_name}_thesis_area_selection_example.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved thesis area-selection example: {out_path}")


def plot_dark_count_workflow(output_dir: Path) -> None:
    """Save a compact thesis workflow figure for the dark-count analysis."""
    fig, ax = plt.subplots(figsize=(12.5, 3.8))
    ax.axis("off")

    steps = [
        ("Recorded\nwaveforms", "raw PMT traces"),
        ("Baseline\ncorrection", "remove offset"),
        ("Hit finder", "threshold crossing"),
        ("Pulse area", "integrate hit window"),
        ("Area cut", r"$A_{\min}<A_{\rm hit}<A_{\max}$"),
        ("Dark-count\nrate", r"$R_{\rm DC}=N/T$"),
    ]

    x_positions = np.linspace(0.08, 0.92, len(steps))
    box_width = 0.125
    box_height = 0.42

    for i, ((title, subtitle), x) in enumerate(zip(steps, x_positions)):
        color = gain_style_color(i)
        rect = plt.Rectangle(
            (x - box_width / 2, 0.35),
            box_width,
            box_height,
            facecolor=color,
            edgecolor=color,
            alpha=0.18,
            linewidth=1.5,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(
            x,
            0.61,
            title,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )
        ax.text(
            x,
            0.43,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
        )

        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x_positions[i + 1] - box_width / 2 - 0.01, 0.56),
                xytext=(x + box_width / 2 + 0.01, 0.56),
                xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", color="0.35", linewidth=1.5),
            )

    ax.set_title("Dark-Count Analysis Workflow", fontsize=20, pad=14)
    out_path = output_dir / "dark_count_analysis_workflow.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved dark-count workflow plot: {out_path}")


def plot_combined_comparison(combined: pd.DataFrame, output_dir: Path) -> None:
    """
    Save a compact comparison plot across all processed datasets and channels.

    If rates are available, the plot compares dark-count rates in Hz. If rates
    are unavailable because no run duration was found, the same figure compares
    hit counts instead.
    """
    if combined.empty:
        return

    plot_df = combined.copy()
    if "channel_label" not in plot_df.columns:
        plot_df["channel_label"] = plot_df["channel"].map(channel_label)
    plot_df["label_short"] = (
        plot_df["dataset"].map(dataset_display_name)
        + " "
        + plot_df["channel_label"].astype(str)
    )

    rates_available = np.isfinite(
        pd.to_numeric(plot_df["dc_rate_area_selected_hz"], errors="coerce")
    ).any()

    if rates_available:
        raw_column = "dc_rate_raw_hz"
        selected_column = "dc_rate_area_selected_hz"
        ylabel = "Dark-count rate [Hz]"
        title = "Dark-count rate comparison"
        output_name = "xdet_dark_count_rate_comparison.png"
    else:
        raw_column = "n_hits_raw"
        selected_column = "n_hits_area_selected"
        ylabel = "Number of hits"
        title = "Dark-count hit-count comparison"
        output_name = "xdet_dark_count_hit_count_comparison.png"

    x = np.arange(len(plot_df), dtype=float)
    width = 0.38
    bar_colors = [
        pmt_color(pmt, i)
        for i, pmt in enumerate(plot_df["channel_label"].astype(str))
    ]

    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.18, top=0.88)
    ax.bar(
        x - width / 2,
        pd.to_numeric(plot_df[raw_column], errors="coerce"),
        width=width,
        color=bar_colors,
        edgecolor=bar_colors,
        alpha=0.35,
        label="all hits",
    )
    ax.bar(
        x + width / 2,
        pd.to_numeric(plot_df[selected_column], errors="coerce"),
        width=width,
        color=bar_colors,
        edgecolor=bar_colors,
        alpha=0.95,
        label="area-selected hits",
    )

    values = pd.concat([
        pd.to_numeric(plot_df[raw_column], errors="coerce"),
        pd.to_numeric(plot_df[selected_column], errors="coerce"),
    ]).to_numpy(dtype=float)
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size:
        ax.set_yscale("log")
        ax.set_ylim(max(np.min(positive) * 0.5, 0.1), np.max(positive) * 2.0)

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label_short"], rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Dataset and PMT")
    ax.set_title(title.title())
    apply_gain_style(ax, which="both", axis="y")

    pmts_in_plot = []
    for pmt in plot_df["channel_label"].astype(str):
        if pmt not in pmts_in_plot:
            pmts_in_plot.append(pmt)

    legend_handles = []
    for pmt in pmts_in_plot:
        color = pmt_color(pmt, len(legend_handles))
        legend_handles.append(
            Patch(
                facecolor=color,
                edgecolor=color,
                alpha=0.35,
                label=f"{pmt} all hits",
            )
        )
        legend_handles.append(
            Patch(
                facecolor=color,
                edgecolor=color,
                alpha=0.95,
                label=f"{pmt} area-selected hits",
            )
        )

    legend = ax.legend(
        handles=legend_handles,
        loc="upper right",
        **GAIN_STYLE_LEGEND_KWARGS,
    )
    legend.get_frame().set_linewidth(1.0)

    out_path = output_dir / output_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined comparison plot: {out_path}")


def analyze_dataset(
    dataset_name: str,
    config: dict,
    data_dir: Path,
    output_dir: Path,
    gain_records: list[dict[str, Any]],
    summary_file: str | None,
    run_duration: float | None,
    area_min_override: float | None,
    area_max_override: float | None,
    n_events: int,
    no_waveforms: bool,
) -> pd.DataFrame:
    filename = config["filename"]
    dataset_output_dir = output_dir / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    raw_data = None
    raw_path = data_dir / f"raw_data_{filename}.pkl"
    if not no_waveforms:
        print(f"Loading raw waveform data: {raw_path}", flush=True)
        raw_data, raw_path = load_raw_data(data_dir, filename)

    print(f"Loading hit data for {dataset_name}...", flush=True)
    df, hit_path = load_hit_data(data_dir, filename)

    area_min = area_min_override if area_min_override is not None else config["area_min"]
    area_max = area_max_override if area_max_override is not None else config["area_max"]

    print(f"\nProcessing dataset: {dataset_name}")
    print(f"  raw file: {raw_path}")
    print(f"  hit file: {hit_path}")

    print_data_overview(dataset_name, raw_data, df)

    duration, duration_source = resolve_run_duration(
        data_dir=data_dir,
        config=config,
        raw_data=raw_data,
        df=df,
        explicit_summary=summary_file,
        explicit_duration=run_duration,
    )

    if duration is None:
        print(
            "WARNING: run duration not available. Hit counts and spectra will be "
            "saved, but rates in Hz will be NaN. Use --run-duration to set it."
        )
    else:
        print(f"Run duration: {duration:.6g} s ({duration_source})")

    n_channels = n_channels_from_data(raw_data, df)

    overview_path = dataset_output_dir / f"{dataset_name}_overview.txt"
    with overview_path.open("w") as file:
        file.write(f"dataset: {dataset_name}\n")
        file.write(f"label: {config.get('label', dataset_name)}\n")
        file.write(f"raw_file: {raw_path}\n")
        file.write(f"hit_file: {hit_path}\n")
        file.write(f"raw_events: {len(raw_data) if raw_data is not None else 'not loaded'}\n")
        file.write(f"hit_rows: {len(df)}\n")
        file.write(f"n_channels: {n_channels}\n")
        file.write(
            "channel_labels: "
            + ", ".join(f"{ch}={channel_label(ch)}" for ch in range(n_channels))
            + "\n"
        )
        file.write(f"run_duration_s: {duration}\n")
        file.write(f"run_duration_source: {duration_source}\n")
        file.write(f"adc_area_to_e: {ADC_AREA_TO_E}\n")
        file.write(f"target_gain_hv: {TARGET_GAIN_HV}\n")
        for ch in range(n_channels):
            pmt = channel_label(ch)
            gain = select_gain(gain_records, pmt)
            if gain is None:
                file.write(f"gain_channel_{ch}_{pmt}: not found\n")
            else:
                file.write(
                    f"gain_channel_{ch}_{pmt}: {gain['gain_e']} e "
                    f"({gain['source']}, {gain['path']})\n"
                )
        file.write(f"raw_fields: {', '.join(raw_fields(raw_data))}\n")
        file.write(f"hit_columns: {', '.join(str(c) for c in df.columns)}\n")

    if not no_waveforms:
        plot_example_waveforms(
            dataset_name=dataset_name,
            raw_data=raw_data,
            df=df,
            output_dir=dataset_output_dir,
            n_events=n_events,
        )

    plot_thesis_area_selection_example(
        dataset_name=dataset_name,
        df=df,
        area_min=area_min,
        area_max=area_max,
        output_dir=dataset_output_dir,
        gain_records=gain_records,
    )

    rates = plot_spectra_and_compute_rates(
        dataset_name=dataset_name,
        df=df,
        run_duration=duration,
        n_channels=n_channels,
        area_min=area_min,
        area_max=area_max,
        output_dir=dataset_output_dir,
        gain_records=gain_records,
    )

    rates.insert(1, "label", config.get("label", dataset_name))
    rates.insert(2, "filename", filename)
    rates.insert(3, "raw_events", len(raw_data) if raw_data is not None else np.nan)
    rates.insert(4, "hit_rows", len(df))
    rates.insert(5, "run_duration_source", duration_source)

    rates_path = dataset_output_dir / f"{dataset_name}_dark_count_rates.csv"
    rates.to_csv(rates_path, index=False)
    print(f"Saved dataset rates: {rates_path}")

    return rates


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze XDet dark-count pickle files.")
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS.keys(), "all"],
        default="all",
        help="Dataset to process. Default: all.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help=f"Directory containing raw_data_*.pkl and hit_data_*.pkl. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory where plots and CSV files are saved. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--summary-file",
        default=None,
        help="Optional Summary_*.txt file containing Total Measuring time.",
    )
    parser.add_argument(
        "--run-duration",
        type=float,
        default=None,
        help="Optional run duration in seconds. Overrides summary-file lookup.",
    )
    parser.add_argument("--area-min", type=float, default=None)
    parser.add_argument("--area-max", type=float, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = DATASETS.keys() if args.dataset == "all" else [args.dataset]
    gain_records = load_gain_records()

    all_rates = []
    for dataset_name in selected:
        rates = analyze_dataset(
            dataset_name=dataset_name,
            config=DATASETS[dataset_name],
            data_dir=data_dir,
            output_dir=output_dir,
            gain_records=gain_records,
            summary_file=args.summary_file,
            run_duration=args.run_duration,
            area_min_override=args.area_min,
            area_max_override=args.area_max,
            n_events=N_WAVEFORM_EXAMPLES,
            no_waveforms=not MAKE_WAVEFORM_EXAMPLE_PLOTS,
        )
        all_rates.append(rates)

    combined = pd.concat(all_rates, ignore_index=True)
    combined_path = output_dir / "xdet_dark_count_rates_summary.csv"
    combined.to_csv(combined_path, index=False)
    plot_dark_count_workflow(output_dir)
    plot_combined_comparison(combined, output_dir)

    print("\nCombined dark-count rate summary:")
    print(combined)
    print(f"\nSaved combined summary: {combined_path}")


if __name__ == "__main__":
    main()
