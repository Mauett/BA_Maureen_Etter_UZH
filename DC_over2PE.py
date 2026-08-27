#!/usr/bin/env python3
"""
dark_count_check_above_2pe.py

Check whether the dark-count data contain a larger relative contribution of
large pulses in air than in vacuum.

The script converts hit areas to photoelectrons using PMT gains from the gain
result JSON files and then counts pulses above 2 PE for LV2483 and LV2480.
It saves:
  - a CSV summary table,
  - a rate/count comparison for A_hit >= 2 PE,
  - a fraction comparison for A_hit >= 2 PE,
  - PE-area spectra with the 2 PE threshold marked.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

DATA_DIR = Path("/disk/gfs_atp/lhoetz/marmotx/Xdet_DCrates")
SAVE_DIR = Path("/home/uzh/mauett/PMT_Analysis_Mauett/Dark_Counts/Over2PE")

GAIN_RESULTS_DIRS = [
    # Same source as plot_pmt_gain_vs_high_voltage_no_chi2.py.
    Path("/home/uzh/mauett/PMT_Analysis_Mauett/Gain/Gain_Plots/One_Plot_ErrorMI"),
]

# The hit areas are converted as A_PE = A_ADC * ADC_AREA_TO_E / gain_e.
# This default is the ADC-area conversion used for the V1730D digitizer in the
# pmt_analysis plotting code. Change it here if your setup used another value.
ADC_AREA_TO_E = 3047.6

TARGET_HV = 800
PE_THRESHOLD = 2.5
LAMP_TOL = 1e-3

TARGET_LAMP_PER_PMT = {
    "LV2480": 1.90,
    "LV2481": 1.95,
    "LV2483": 1.95,
    "LV2485": 1.95,
}

CHANNEL_TO_PMT = {
    0: "LV2483",
    1: "LV2480",
}

# Optional manual fallback if no gain JSON is found. Leave empty if the gain
# result JSONs are available.
MANUAL_GAINS_E = {
    # "LV2480": 4.24e6,
    # "LV2483": 4.06e6,
}

DATASETS = {
    "air": {
        "label": "Air",
        "filename": "XDetector_DC_OR_800V_Module_0_0",
        "summary_files": [
            "/disk/gfs_atp/lhoetz/marmotx/XDetector_DC_OR_800V/"
            "Summary_XDetector_DC_OR_800V.txt",
        ],
    },
    "vacuum": {
        "label": "Vacuum",
        "filename": "XDet_vacuum_800V_matchedThresholdOnFanInOut0mv_ORTrigger_Module_0_0",
        "summary_files": [
            "/home/atp/lhoetz/ln_gfsatp/lhoetz/marmotx/data_20260428/"
            "XDet_vacuum_800V_matchedThresholdOnFanInOut0mv_ORTrigger/"
            "Summary_XDet_vacuum_800V_matchedThresholdOnFanInOut0mv_ORTrigger.txt",
        ],
    },
}

PMT_COLORS = {
    "LV2483": "#1b9e77",
    "LV2480": "#1f78b4",
    "LV2481": "slateblue",
    "LV2485": "darkmagenta",
}

PMT_DATASET_COLORS = {
    "LV2483": {
        "Air": "#1b9e77",
        "Vacuum": "#a6d854",
    },
    "LV2480": {
        "Air": "#1f78b4",
        "Vacuum": "#4cc9f0",
    },
}

DATASET_STYLE = {
    "Air": {
        "alpha": 0.95,
        "linestyle": "-",
    },
    "Vacuum": {
        "alpha": 0.95,
        "linestyle": "-",
    },
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


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def extract_pmt(name: str):
    match = re.search(r"(LV\d{4})", str(name))
    return match.group(1) if match else None


def normalized_name(name: str) -> str:
    return str(name).replace("-", "_")


def extract_lamp_vpp(path: Path):
    match = re.search(
        r"Lamp_?([0-9]+(?:\.[0-9]+)?)_?Vpp",
        normalized_name(path.name),
        re.IGNORECASE,
    )
    if not match:
        return None

    try:
        return float(match.group(1))
    except ValueError:
        return None


def extract_hv_from_path(path: Path):
    matches = re.findall(r"(?<![A-Za-z0-9.])(-?\d{3,4})V(?!pp)", path.name, re.IGNORECASE)
    if not matches:
        matches = re.findall(r"(?<![A-Za-z0-9.])(-?\d{3,4})V(?!pp)", normalized_name(path.name), re.IGNORECASE)
    if not matches:
        return None
    try:
        return abs(int(matches[-1]))
    except ValueError:
        return None


def apply_style(ax, which="both", axis="both"):
    ax.grid(True, which=which, axis=axis, alpha=0.28)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def pmt_color(pmt, fallback="0.30"):
    return PMT_COLORS.get(str(pmt), fallback)


def pmt_dataset_color(pmt, dataset, fallback="0.30"):
    return PMT_DATASET_COLORS.get(str(pmt), {}).get(str(dataset), pmt_color(pmt, fallback))


def legend_inside(ax):
    legend = ax.legend(
        loc="best",
        frameon=True,
        framealpha=0.92,
        edgecolor="0.75",
        fontsize=12,
    )
    legend.get_frame().set_linewidth(1.0)
    return legend


def add_pmt_dataset_legend(ax, pmts):
    handles = []
    for pmt in pmts:
        for dataset in DATASET_STYLE:
            color = pmt_dataset_color(pmt, dataset)
            handles.append(
                Patch(
                    facecolor=color,
                    edgecolor=color,
                    alpha=DATASET_STYLE[dataset]["alpha"],
                    label=f"{pmt} {dataset}",
                )
            )

    legend = ax.legend(
        handles=handles,
        loc="best",
        frameon=True,
        framealpha=0.92,
        edgecolor="0.75",
        fontsize=12,
    )
    legend.get_frame().set_linewidth(1.0)
    return legend


def read_run_duration(summary_files):
    for path_like in summary_files:
        path = Path(path_like)
        if not path.is_file():
            continue
        with path.open("r") as file:
            for line in file:
                if "Total Measuring time" in line:
                    return float(line.split(":")[1].strip().split()[0]), str(path)
    return None, "not found"


def read_gain_result(path: Path):
    try:
        with path.open("r") as file:
            data = json.load(file)
    except Exception:
        return None

    pmt = extract_pmt(path.name) or extract_pmt(str(path.parent))
    hv = extract_hv_from_path(path)
    lamp = extract_lamp_vpp(path)

    model_dep = data.get("model_dependent_fit", {})
    params = model_dep.get("params", {}) if isinstance(model_dep, dict) else {}

    if not isinstance(model_dep, dict) or not model_dep.get("fit_success", False):
        return None

    gain = finite_float(params.get("gain_e"))
    gain_err = finite_float(params.get("gain_e_err"))
    if gain is None or gain <= 0:
        return None

    return {
        "path": path,
        "pmt": pmt,
        "hv": hv,
        "lamp": lamp,
        "gain_e": gain,
        "gain_e_err": gain_err,
        "source": "model_dependent",
        "fit_success": bool(model_dep.get("fit_success", False)) if isinstance(model_dep, dict) else False,
        "mtime": path.stat().st_mtime,
    }


def load_gain_records():
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


def select_gain(gain_records, pmt):
    candidates = [
        record for record in gain_records
        if record["pmt"] == pmt
        and record["hv"] is not None
        and abs(int(record["hv"])) == TARGET_HV
        and record["gain_e"] is not None
    ]

    target_lamp = TARGET_LAMP_PER_PMT.get(pmt)
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
        }

    if not candidates:
        return None

    gains = np.array([record["gain_e"] for record in candidates], dtype=float)
    errs = np.array([
        record["gain_e_err"] if record["gain_e_err"] is not None and record["gain_e_err"] > 0 else np.nan
        for record in candidates
    ], dtype=float)

    good_err = np.isfinite(errs) & (errs > 0)
    if gains.size == 1:
        gain_e = float(gains[0])
        gain_e_err = None if not np.isfinite(errs[0]) else float(errs[0])
    elif np.all(good_err):
        weights = 1.0 / errs**2
        gain_e = float(np.sum(weights * gains) / np.sum(weights))
        gain_e_err = float(np.sqrt(1.0 / np.sum(weights)))
    else:
        gain_e = float(np.median(gains))
        gain_e_err = float(np.std(gains, ddof=1) / np.sqrt(gains.size)) if gains.size > 1 else None

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


def load_hit_data(filename):
    path = DATA_DIR / f"hit_data_{filename}.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing hit-data pickle: {path}")
    return pd.read_pickle(path), path


def binomial_uncertainty(k, n):
    if n <= 0:
        return np.nan
    fraction = k / n
    return float(np.sqrt(fraction * (1.0 - fraction) / n))


def approximate_peak(values, low_percentile=0.5, high_percentile=99.5, n_bins=160):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan

    lo = float(np.percentile(values, low_percentile))
    hi = float(np.percentile(values, high_percentile))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float(np.median(values))

    counts, edges = np.histogram(values, bins=np.linspace(lo, hi, n_bins + 1))
    if counts.size == 0 or np.max(counts) <= 0:
        return float(np.median(values))

    idx = int(np.argmax(counts))
    return float(0.5 * (edges[idx] + edges[idx + 1]))


# ------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------

def analyze():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    gain_records = load_gain_records()

    rows = []
    spectra = []

    for dataset_key, config in DATASETS.items():
        dataset_label = config["label"]
        df, hit_path = load_hit_data(config["filename"])
        duration_s, duration_source = read_run_duration(config["summary_files"])

        if "channel" not in df.columns or "hit_area_raw" not in df.columns:
            raise KeyError(f"{hit_path} must contain 'channel' and 'hit_area_raw'.")

        print(f"\nDataset: {dataset_label}")
        print(f"  hit file: {hit_path}")
        print(f"  duration: {duration_s} s ({duration_source})")

        for channel, pmt in CHANNEL_TO_PMT.items():
            gain = select_gain(gain_records, pmt)
            if gain is None:
                print(f"  WARNING: no {TARGET_HV} V gain found for {pmt}; skipping.")
                continue

            channel_hits = df[df["channel"] == channel].copy()
            hit_area_adc = pd.to_numeric(channel_hits["hit_area_raw"], errors="coerce")
            hit_area_adc = hit_area_adc[np.isfinite(hit_area_adc)]
            hit_area_pe = hit_area_adc.to_numpy(dtype=float) * ADC_AREA_TO_E / float(gain["gain_e"])

            above = hit_area_pe >= PE_THRESHOLD
            n_total = int(hit_area_pe.size)
            n_above = int(np.count_nonzero(above))
            n_below = int(n_total - n_above)

            if duration_s is not None and duration_s > 0:
                rate_total_hz = n_total / duration_s
                rate_above_hz = n_above / duration_s
                rate_below_hz = n_below / duration_s
                rate_above_unc_hz = np.sqrt(n_above) / duration_s if n_above > 0 else 0.0
            else:
                rate_total_hz = np.nan
                rate_above_hz = np.nan
                rate_below_hz = np.nan
                rate_above_unc_hz = np.nan

            fraction_above = n_above / n_total if n_total > 0 else np.nan
            fraction_above_unc = binomial_uncertainty(n_above, n_total)
            threshold_adc = PE_THRESHOLD * float(gain["gain_e"]) / ADC_AREA_TO_E
            median_adc = float(np.median(hit_area_adc)) if n_total else np.nan
            median_pe = float(np.median(hit_area_pe)) if n_total else np.nan
            peak_adc = approximate_peak(hit_area_adc)
            peak_pe = approximate_peak(hit_area_pe)

            rows.append({
                "dataset": dataset_label,
                "pmt": pmt,
                "channel": channel,
                "gain_e": gain["gain_e"],
                "gain_e_err": gain["gain_e_err"],
                "gain_source": gain["source"],
                "gain_file": "" if gain["path"] is None else str(gain["path"]),
                "gain_n_files": gain.get("n_files", 1),
                "gain_lamp_values": gain.get("lamp_values", []),
                "adc_area_to_e": ADC_AREA_TO_E,
                "pe_threshold": PE_THRESHOLD,
                "threshold_adc_samples": threshold_adc,
                "median_hit_area_adc_samples": median_adc,
                "median_hit_area_pe": median_pe,
                "approx_peak_hit_area_adc_samples": peak_adc,
                "approx_peak_hit_area_pe": peak_pe,
                "run_duration_s": duration_s,
                "duration_source": duration_source,
                "n_total_hits": n_total,
                "n_hits_below_2pe": n_below,
                "n_hits_above_2pe": n_above,
                "fraction_above_2pe": fraction_above,
                "fraction_above_2pe_unc": fraction_above_unc,
                "rate_total_hz": rate_total_hz,
                "rate_below_2pe_hz": rate_below_hz,
                "rate_above_2pe_hz": rate_above_hz,
                "rate_above_2pe_unc_hz": rate_above_unc_hz,
            })

            spectra.append({
                "dataset": dataset_label,
                "pmt": pmt,
                "hit_area_adc": hit_area_adc.to_numpy(dtype=float),
                "hit_area_pe": hit_area_pe,
                "threshold_adc": threshold_adc,
            })

            print(
                f"  {pmt}: N={n_total}, N(A>=2PE)={n_above}, "
                f"f={fraction_above:.3f}, R(A>=2PE)={rate_above_hz:.3g} Hz, "
                f"gain={float(gain['gain_e']):.3e} e, "
                f"1PE={float(gain['gain_e']) / ADC_AREA_TO_E:.1f} ADC, "
                f"peak~{peak_pe:.2f} PE"
            )

    results = pd.DataFrame(rows)
    csv_path = SAVE_DIR / "dark_count_above_2pe_summary.csv"
    results.to_csv(csv_path, index=False)
    print(f"\nSaved summary table: {csv_path}")

    make_rate_or_count_plot(results)
    make_fraction_plot(results)
    make_spectra_plot(spectra)
    make_adc_spectra_plot(spectra)

    return results


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------

def make_rate_or_count_plot(results):
    if results.empty:
        return

    rates_available = np.isfinite(results["rate_above_2pe_hz"]).any()
    pmts = [pmt for pmt in CHANNEL_TO_PMT.values() if pmt in set(results["pmt"])]
    datasets = [config["label"] for config in DATASETS.values()]
    x = np.arange(len(pmts), dtype=float)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.16, top=0.88)

    for i, dataset in enumerate(datasets):
        style = DATASET_STYLE.get(dataset, DATASET_STYLE["Air"])
        values = []
        errors = []
        for pmt in pmts:
            row = results[(results["dataset"] == dataset) & (results["pmt"] == pmt)]
            if row.empty:
                values.append(np.nan)
                errors.append(np.nan)
            elif rates_available:
                values.append(float(row.iloc[0]["rate_above_2pe_hz"]))
                errors.append(float(row.iloc[0]["rate_above_2pe_unc_hz"]))
            else:
                n = float(row.iloc[0]["n_hits_above_2pe"])
                values.append(n)
                errors.append(np.sqrt(n))

        colors = [pmt_dataset_color(pmt, dataset, f"C{j}") for j, pmt in enumerate(pmts)]
        ax.bar(
            x + (i - 0.5) * width,
            values,
            yerr=errors,
            width=width,
            capsize=3,
            color=colors,
            edgecolor=colors,
            alpha=style["alpha"],
            label=dataset,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(pmts)
    ylabel = rf"$R(A_{{\mathrm{{hit}}}}\geq {PE_THRESHOLD:g}\,\mathrm{{PE}})$ [Hz]"
    title = rf"Large dark-count pulses above {PE_THRESHOLD:g} PE"
    if not rates_available:
        ylabel = rf"$N(A_{{\mathrm{{hit}}}}\geq {PE_THRESHOLD:g}\,\mathrm{{PE}})$"
        title += " (counts)"
    ax.set_ylabel(ylabel)
    ax.set_xlabel("PMT")
    ax.set_title(title)
    apply_style(ax, which="both", axis="y")
    add_pmt_dataset_legend(ax, pmts)

    out_path = SAVE_DIR / "dark_count_rate_above_2pe.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def make_fraction_plot(results):
    if results.empty:
        return

    pmts = [pmt for pmt in CHANNEL_TO_PMT.values() if pmt in set(results["pmt"])]
    datasets = [config["label"] for config in DATASETS.values()]
    x = np.arange(len(pmts), dtype=float)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.16, top=0.88)

    for i, dataset in enumerate(datasets):
        style = DATASET_STYLE.get(dataset, DATASET_STYLE["Air"])
        values = []
        errors = []
        for pmt in pmts:
            row = results[(results["dataset"] == dataset) & (results["pmt"] == pmt)]
            if row.empty:
                values.append(np.nan)
                errors.append(np.nan)
            else:
                values.append(float(row.iloc[0]["fraction_above_2pe"]))
                errors.append(float(row.iloc[0]["fraction_above_2pe_unc"]))

        colors = [pmt_dataset_color(pmt, dataset, f"C{j}") for j, pmt in enumerate(pmts)]
        ax.bar(
            x + (i - 0.5) * width,
            values,
            yerr=errors,
            width=width,
            capsize=3,
            color=colors,
            edgecolor=colors,
            alpha=style["alpha"],
            label=dataset,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(pmts)
    ax.set_ylabel(rf"$f(A_{{\mathrm{{hit}}}}\geq {PE_THRESHOLD:g}\,\mathrm{{PE}})$")
    ax.set_xlabel("PMT")
    ax.set_title(rf"Fraction of dark-count pulses above {PE_THRESHOLD:g} PE")
    apply_style(ax, which="both", axis="y")
    add_pmt_dataset_legend(ax, pmts)

    out_path = SAVE_DIR / "dark_count_fraction_above_2pe.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def make_spectra_plot(spectra):
    if not spectra:
        return

    pmts = [pmt for pmt in CHANNEL_TO_PMT.values()]
    fig, axes = plt.subplots(1, len(pmts), figsize=(6.2 * len(pmts), 5.6), squeeze=False)
    axes = axes.flatten()

    all_positive = np.concatenate([
        spec["hit_area_pe"][np.isfinite(spec["hit_area_pe"]) & (spec["hit_area_pe"] >= 0)]
        for spec in spectra
    ])
    if all_positive.size:
        xmax = min(max(np.percentile(all_positive, 99.5), PE_THRESHOLD * 2.0), 15.0)
    else:
        xmax = 10.0
    bins = np.linspace(0.0, xmax, 90)

    for ax, pmt in zip(axes, pmts):
        for spec in spectra:
            if spec["pmt"] != pmt:
                continue
            style = DATASET_STYLE.get(spec["dataset"], DATASET_STYLE["Air"])
            values = spec["hit_area_pe"]
            values = values[np.isfinite(values) & (values >= 0)]
            ax.hist(
                values,
                bins=bins,
                histtype="step",
                linewidth=1.8,
                color=pmt_dataset_color(pmt, spec["dataset"]),
                linestyle=style["linestyle"],
                alpha=style["alpha"],
                label=spec["dataset"],
            )

        ax.axvline(
            PE_THRESHOLD,
            color="slateblue",
            linestyle="--",
            linewidth=1.4,
            label=rf"{PE_THRESHOLD:g} PE",
        )
        ax.set_yscale("log")
        ax.set_xlabel(r"Hit area $A_{\mathrm{hit}}$ [PE]")
        ax.set_ylabel("Counts")
        ax.set_title(pmt)
        apply_style(ax, which="both")
        legend_inside(ax)

    fig.suptitle("Dark-count hit-area spectra in PE")
    fig.tight_layout()
    out_path = SAVE_DIR / "dark_count_hit_area_pe_spectra.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def make_adc_spectra_plot(spectra):
    if not spectra:
        return

    pmts = [pmt for pmt in CHANNEL_TO_PMT.values()]
    fig, axes = plt.subplots(1, len(pmts), figsize=(6.2 * len(pmts), 5.6), squeeze=False)
    axes = axes.flatten()

    all_positive = np.concatenate([
        spec["hit_area_adc"][np.isfinite(spec["hit_area_adc"]) & (spec["hit_area_adc"] >= 0)]
        for spec in spectra
    ])
    if all_positive.size:
        xmax = min(max(np.percentile(all_positive, 99.5), 5000.0), 20000.0)
    else:
        xmax = 10000.0
    bins = np.linspace(0.0, xmax, 110)

    for ax, pmt in zip(axes, pmts):
        for spec in spectra:
            if spec["pmt"] != pmt:
                continue
            style = DATASET_STYLE.get(spec["dataset"], DATASET_STYLE["Air"])
            values = spec["hit_area_adc"]
            values = values[np.isfinite(values) & (values >= 0)]
            ax.hist(
                values,
                bins=bins,
                histtype="step",
                linewidth=1.8,
                color=pmt_dataset_color(pmt, spec["dataset"]),
                linestyle=style["linestyle"],
                alpha=style["alpha"],
                label=spec["dataset"],
            )
            ax.axvline(
                spec["threshold_adc"],
                color=pmt_dataset_color(pmt, spec["dataset"]),
                linestyle=style["linestyle"],
                linewidth=1.2,
                alpha=0.80,
            )

        ax.set_yscale("log")
        ax.set_xlabel(r"Hit area $A_{\mathrm{hit}}$ [ADC samples]")
        ax.set_ylabel("Counts")
        ax.set_title(pmt)
        apply_style(ax, which="both")
        legend_inside(ax)

    fig.suptitle("Dark-count hit-area spectra in raw ADC units")
    fig.tight_layout()
    out_path = SAVE_DIR / "dark_count_hit_area_adc_spectra.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    analyze()