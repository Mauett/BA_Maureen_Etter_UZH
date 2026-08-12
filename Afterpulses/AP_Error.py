#!/usr/bin/env python3
"""
Post-process afterpulse summary tables and add uncertainty estimates.

Run this after AP_run_analysis_full_data*.py has produced:

    afterpulse_summary.csv

The script adds:
  1. Per-run gain uncertainty propagation.
  2. Per-PMT run-to-run standard deviation and SEM.
  3. Per-PMT propagated statistical uncertainty on the mean.
  4. Per-PMT gain systematic uncertainty.
  5. Per-PMT frame-size systematic estimate.
  6. Optional cut-variation systematic estimate from additional summary CSVs.

The original analysis already stores Poisson count uncertainties such as
ap_rate_per_pe_separable_unc. This script does not replace those; it combines
them with additional uncertainty sources in separate columns.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

BASE_SAVE_DIR = Path(__file__).resolve().parent / "Afterpulse_Results"
BASE_SAVE_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = BASE_SAVE_DIR / "afterpulse_summary.csv"

AREA_THRESHOLD_PE = 2

# Optional: add cut-variation summary files here after running the full
# analysis with different thresholds/cuts. The first entry should usually be
# the nominal result.
#
# Example:
# CUT_VARIATION_SUMMARY_FILES = [
#     ("nominal", "/path/to/nominal/afterpulse_summary.csv"),
#     ("area_thr_1p5pe", "/path/to/area_1p5/afterpulse_summary.csv"),
#     ("area_thr_2p5pe", "/path/to/area_2p5/afterpulse_summary.csv"),
#     ("early_adc_80", "/path/to/early_adc_80/afterpulse_summary.csv"),
#     ("early_adc_120", "/path/to/early_adc_120/afterpulse_summary.csv"),
# ]
CUT_VARIATION_SUMMARY_FILES = []

RATE_COLUMNS = [
    "ap_rate_per_pe",
    "ap_rate_per_pe_separable",
    "ap_rate_per_pe_above_thr",
    "ap_rate_per_pe_below_thr",
]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def ensure_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def rate_unc_column(rate_col):
    return rate_col + "_unc"


def gain_unc_column(rate_col):
    return rate_col + "_gain_unc"


def stat_gain_unc_column(rate_col):
    return rate_col + "_stat_gain_unc"


def add_per_run_gain_uncertainties(results_df):
    """Add gain-propagated uncertainty columns for each rate column.

    Since A_PE is proportional to 1/G, the charge-normalized rate R is
    proportional to G for fixed raw areas:

        sigma_R,gain = R * sigma_G / G

    This only accounts for the normalization effect of the gain uncertainty.
    It does not model possible migration across the PE area threshold.
    """
    numeric_cols = ["gain", "gain_unc"] + RATE_COLUMNS
    numeric_cols += [rate_unc_column(col) for col in RATE_COLUMNS]
    ensure_numeric(results_df, numeric_cols)

    rel_gain_unc = results_df["gain_unc"] / results_df["gain"]
    rel_gain_unc = rel_gain_unc.replace([np.inf, -np.inf], np.nan)
    results_df["relative_gain_unc"] = rel_gain_unc

    for rate_col in RATE_COLUMNS:
        if rate_col not in results_df.columns:
            continue

        stat_col = rate_unc_column(rate_col)
        gain_col = gain_unc_column(rate_col)
        total_col = stat_gain_unc_column(rate_col)

        if stat_col not in results_df.columns:
            results_df[stat_col] = np.nan

        results_df[gain_col] = np.abs(results_df[rate_col]) * results_df["relative_gain_unc"]

        stat_unc = results_df[stat_col].fillna(0.0)
        gain_unc = results_df[gain_col].fillna(0.0)
        results_df[total_col] = np.sqrt(stat_unc ** 2 + gain_unc ** 2)

    return results_df


def frame_systematic_by_pmt(results_df, rate_col):
    """Estimate frame-size systematic as max frame-mean deviation from PMT mean."""
    if "frame_type" not in results_df.columns:
        return pd.DataFrame(columns=["pmt_serial", rate_col + "_frame_syst_unc"])

    rows = []
    for pmt, df_pmt in results_df.groupby("pmt_serial"):
        pmt_mean = df_pmt[rate_col].mean()
        frame_means = df_pmt.groupby("frame_type")[rate_col].mean()

        if len(frame_means) <= 1 or not np.isfinite(pmt_mean):
            frame_syst = 0.0
        else:
            frame_syst = np.nanmax(np.abs(frame_means.values - pmt_mean))

        rows.append({
            "pmt_serial": pmt,
            rate_col + "_frame_syst_unc": frame_syst,
            rate_col + "_n_frame_types": len(frame_means),
        })

    return pd.DataFrame(rows)


def cut_variation_systematic(rate_col):
    """Estimate cut systematic from additional afterpulse_summary.csv files.

    The systematic is the standard deviation of the PMT mean rate across the
    configured cut variations. If no variation files are configured, all
    returned values are absent and later treated as zero.
    """
    if not CUT_VARIATION_SUMMARY_FILES:
        return pd.DataFrame(columns=["pmt_serial", rate_col + "_cut_syst_unc"])

    rows = []
    for label, path in CUT_VARIATION_SUMMARY_FILES:
        df = pd.read_csv(path)
        ensure_numeric(df, [rate_col])
        grouped = df.groupby("pmt_serial")[rate_col].mean().reset_index()
        grouped["cut_label"] = label
        rows.append(grouped)

    if not rows:
        return pd.DataFrame(columns=["pmt_serial", rate_col + "_cut_syst_unc"])

    combined = pd.concat(rows, ignore_index=True)
    out = combined.groupby("pmt_serial").agg({
        rate_col: ["std", "min", "max", "count"],
    }).reset_index()
    out.columns = [
        "pmt_serial",
        rate_col + "_cut_syst_unc",
        rate_col + "_cut_min",
        rate_col + "_cut_max",
        rate_col + "_n_cut_variations",
    ]
    out[rate_col + "_cut_syst_unc"] = out[rate_col + "_cut_syst_unc"].fillna(0.0)
    return out


def make_pmt_uncertainty_summary(results_df, rate_col):
    """Build one PMT-level uncertainty table for a selected rate column."""
    stat_col = rate_unc_column(rate_col)
    gain_col = gain_unc_column(rate_col)

    rows = []
    for pmt, df_pmt in results_df.groupby("pmt_serial"):
        rates = pd.to_numeric(df_pmt[rate_col], errors="coerce")
        n_runs = int(rates.notna().sum())
        mean_rate = rates.mean()
        run_std = rates.std(ddof=1) if n_runs > 1 else 0.0
        run_sem = run_std / np.sqrt(n_runs) if n_runs > 0 else np.nan

        if stat_col in df_pmt.columns and n_runs > 0:
            stat_vals = pd.to_numeric(df_pmt[stat_col], errors="coerce").fillna(0.0)
            poisson_unc_on_mean = np.sqrt(np.sum(stat_vals ** 2)) / n_runs
        else:
            poisson_unc_on_mean = np.nan

        rel_gain_vals = pd.to_numeric(df_pmt["relative_gain_unc"], errors="coerce")
        if rel_gain_vals.notna().any() and np.isfinite(mean_rate):
            # The same gain calibration is usually common to all runs of one PMT,
            # so this systematic is treated as correlated and does not average down.
            gain_syst_unc = abs(mean_rate) * rel_gain_vals.mean()
        else:
            gain_syst_unc = 0.0

        rows.append({
            "pmt_serial": pmt,
            rate_col + "_mean": mean_rate,
            rate_col + "_run_std": run_std,
            rate_col + "_n_runs": n_runs,
            rate_col + "_run_sem": run_sem,
            rate_col + "_poisson_unc_on_mean": poisson_unc_on_mean,
            rate_col + "_gain_syst_unc": gain_syst_unc,
        })

    summary = pd.DataFrame(rows)

    frame_syst = frame_systematic_by_pmt(results_df, rate_col)
    cut_syst = cut_variation_systematic(rate_col)

    summary = summary.merge(frame_syst, on="pmt_serial", how="left")
    summary = summary.merge(cut_syst, on="pmt_serial", how="left")

    for suffix in ["_frame_syst_unc", "_cut_syst_unc"]:
        col = rate_col + suffix
        if col not in summary.columns:
            summary[col] = 0.0
        summary[col] = summary[col].fillna(0.0)

    sem = summary[rate_col + "_run_sem"].fillna(0.0)
    poisson = summary[rate_col + "_poisson_unc_on_mean"].fillna(0.0)
    gain = summary[rate_col + "_gain_syst_unc"].fillna(0.0)
    frame = summary[rate_col + "_frame_syst_unc"].fillna(0.0)
    cut = summary[rate_col + "_cut_syst_unc"].fillna(0.0)

    summary[rate_col + "_total_unc_sem_gain_frame"] = np.sqrt(
        sem ** 2 + gain ** 2 + frame ** 2
    )
    summary[rate_col + "_total_unc_sem_gain_frame_cut"] = np.sqrt(
        sem ** 2 + gain ** 2 + frame ** 2 + cut ** 2
    )
    summary[rate_col + "_total_unc_poisson_gain_frame"] = np.sqrt(
        poisson ** 2 + gain ** 2 + frame ** 2
    )
    summary[rate_col + "_total_unc_poisson_gain_frame_cut"] = np.sqrt(
        poisson ** 2 + gain ** 2 + frame ** 2 + cut ** 2
    )

    return summary


def save_selected_uncertainty_plot(pmt_summaries, output_dir):
    """Recreate the threshold-split PMT comparison with uncertainty bars."""
    total = pmt_summaries["ap_rate_per_pe_separable"]
    above = pmt_summaries["ap_rate_per_pe_above_thr"]
    below = pmt_summaries["ap_rate_per_pe_below_thr"]

    pmt_order = total["pmt_serial"].tolist()
    x = np.arange(len(pmt_order))
    width = 0.25

    def ordered(summary, rate_col, value_suffix, unc_suffix):
        df = summary.set_index("pmt_serial").reindex(pmt_order)
        return (
            df[rate_col + value_suffix].values,
            df[rate_col + unc_suffix].values,
        )

    total_y, total_err = ordered(
        total,
        "ap_rate_per_pe_separable",
        "_mean",
        "_total_unc_sem_gain_frame_cut",
    )
    above_y, above_err = ordered(
        above,
        "ap_rate_per_pe_above_thr",
        "_mean",
        "_total_unc_sem_gain_frame_cut",
    )
    below_y, below_err = ordered(
        below,
        "ap_rate_per_pe_below_thr",
        "_mean",
        "_total_unc_sem_gain_frame_cut",
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(left=0.10, right=0.82, bottom=0.18, top=0.90)

    ax.bar(
        x - width,
        total_y,
        width=width,
        yerr=total_err,
        capsize=4,
        label=r"$\mathrm{Total\ separable}$",
    )
    ax.bar(
        x,
        above_y,
        width=width,
        yerr=above_err,
        capsize=4,
        label=rf"$A_{{\mathrm{{AP}}}} \geq {AREA_THRESHOLD_PE}\,\mathrm{{PE}}$",
    )
    ax.bar(
        x + width,
        below_y,
        width=width,
        yerr=below_err,
        capsize=4,
        label=rf"$A_{{\mathrm{{AP}}}} < {AREA_THRESHOLD_PE}\,\mathrm{{PE}}$",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(pmt_order, rotation=45)
    ax.set_ylabel(r"$R_{\mathrm{AP}}/\mathrm{PE}$")
    ax.set_xlabel(r"$\mathrm{PMT}$")
    ax.set_title(r"$\mathrm{Afterpulse\ rate\ split\ by\ threshold}$")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True)

    fig.savefig(
        output_dir / "PMT_comparison_split_threshold_with_uncertainties.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_uncertainty_budget_plot(summary, rate_col, output_dir):
    pmt_order = summary["pmt_serial"].tolist()
    x = np.arange(len(pmt_order))
    width = 0.20

    components = [
        (rate_col + "_run_sem", "run-to-run SEM"),
        (rate_col + "_poisson_unc_on_mean", "Poisson"),
        (rate_col + "_gain_syst_unc", "gain"),
        (rate_col + "_frame_syst_unc", "frame"),
        (rate_col + "_cut_syst_unc", "cuts"),
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    for j, (col, label) in enumerate(components):
        if col not in summary.columns:
            continue
        offset = (j - 2) * width
        ax.bar(x + offset, summary[col].fillna(0.0), width=width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(pmt_order, rotation=45)
    ax.set_ylabel(r"$\sigma(R_{\mathrm{AP}}/\mathrm{PE})$")
    ax.set_xlabel(r"$\mathrm{PMT}$")
    ax.set_title(r"$\mathrm{Uncertainty\ budget:\ separable\ afterpulse\ rate}$")
    ax.legend(frameon=True)

    fig.savefig(
        output_dir / "PMT_uncertainty_budget_separable.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    if not SUMMARY_CSV.is_file():
        raise FileNotFoundError("Summary CSV not found: {}".format(SUMMARY_CSV))

    results_df = pd.read_csv(SUMMARY_CSV)
    results_df = add_per_run_gain_uncertainties(results_df)

    output_with_unc = BASE_SAVE_DIR / "afterpulse_summary_with_uncertainties.csv"
    results_df.to_csv(output_with_unc, index=False)
    print("Saved per-run uncertainty table: {}".format(output_with_unc))

    pmt_summaries = {}
    for rate_col in RATE_COLUMNS:
        if rate_col not in results_df.columns:
            continue

        summary = make_pmt_uncertainty_summary(results_df, rate_col)
        pmt_summaries[rate_col] = summary

        out_path = BASE_SAVE_DIR / "{}_pmt_uncertainty_summary.csv".format(rate_col)
        summary.to_csv(out_path, index=False)
        print("Saved PMT uncertainty summary: {}".format(out_path))

    if all(col in pmt_summaries for col in [
        "ap_rate_per_pe_separable",
        "ap_rate_per_pe_above_thr",
        "ap_rate_per_pe_below_thr",
    ]):
        save_selected_uncertainty_plot(pmt_summaries, BASE_SAVE_DIR)

    if "ap_rate_per_pe_separable" in pmt_summaries:
        save_uncertainty_budget_plot(
            pmt_summaries["ap_rate_per_pe_separable"],
            "ap_rate_per_pe_separable",
            BASE_SAVE_DIR,
        )

    print("\nRecommended thesis reporting:")
    print("- Use run-to-run SEM to show repeatability between runs.")
    print("- Quote gain uncertainty separately or include it as a correlated systematic.")
    print("- Use frame systematic if multiple frame lengths are present.")
    print("- Add CUT_VARIATION_SUMMARY_FILES after running alternative cuts to include cut systematics.")


if __name__ == "__main__":
    main()