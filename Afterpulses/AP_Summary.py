#!/usr/bin/env python3
"""
Make the important afterpulse summary plots for AP_Run_anaysis_better_plots.py.

This script does not rerun the waveform analysis. It reads the
afterpulse_summary.csv produced by AP_Run_anaysis_better_plots.py and creates
only the plots that add distinct information:

  1. PMT_comparison_split_threshold_better_from_csv.png
     Total separable afterpulse rate split into A_AP >= 2 PE and A_AP < 2 PE.

  2. PMT_fraction_above_2pe_better_from_csv.png
     Fraction of separable afterpulse rate from large afterpulses.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------

DEFAULT_SUMMARY_CSV = Path(
    "/home/uzh/mauett/PMT_Analysis_Mauett/pmt/PMT_Analysis/"
    "AP_Plots_full_data/Better_Plots/afterpulse_summary.csv"
)

AREA_THRESHOLD_PE = 2
SHOW_ERROR_BARS = True
ERROR_BAR_KIND = "propagated_unc_on_mean"
REQUIRED_N_SAMPLES_PER_WAVEFORM = 2500

RATE_TOTAL = "ap_rate_per_pe_separable"
RATE_ABOVE = "ap_rate_per_pe_above_thr"
RATE_BELOW = "ap_rate_per_pe_below_thr"
OUTPUT_SUFFIX = "_better_from_csv"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def rate_unc_col(rate_col):
    return rate_col + "_unc"


def count_unc_col(count_col):
    return count_col + "_unc"


def ensure_rate_uncertainties(df):
    """
    Ensure that the statistical uncertainty columns exist.

    AP_Run_anaysis_better_plots.py writes these columns already. This fallback
    reconstructs them from the event counts and normalization if an older CSV is
    used.
    """
    df = df.copy()

    if "sum_p0_area_pe" not in df.columns:
        return df

    denominator = pd.to_numeric(df["sum_p0_area_pe"], errors="coerce")
    denominator = denominator.replace([np.inf, -np.inf], np.nan)

    mappings = [
        (RATE_TOTAL, "n_ap_separable"),
        (RATE_ABOVE, "n_ap_separable_above_thr"),
        (RATE_BELOW, "n_ap_separable_below_thr"),
        ("ap_rate_per_pe", "n_ap"),
    ]

    for rate_col, count_col in mappings:
        unc_col = rate_unc_col(rate_col)
        if unc_col in df.columns:
            df[unc_col] = pd.to_numeric(df[unc_col], errors="coerce")
            continue

        if rate_col not in df.columns or count_col not in df.columns:
            continue

        counts = pd.to_numeric(df[count_col], errors="coerce")
        uncertainty = np.sqrt(counts) / denominator
        uncertainty = uncertainty.replace([np.inf, -np.inf], np.nan)
        uncertainty[denominator <= 0] = np.nan
        df[unc_col] = uncertainty

    return df


def pmt_order_from_serials(serials):
    def key(serial):
        text = str(serial)
        digits = "".join(ch for ch in text if ch.isdigit())
        return int(digits) if digits else text

    return sorted(pd.unique(serials), key=key)


def keep_required_frame_length(df):
    """
    Keep only rows with the configured waveform length.

    This makes summary plots robust even if the CSV still contains older runs
    with 1500- or 3500-sample frames.
    """
    if "n_samples_per_waveform" in df.columns:
        frame_lengths = pd.to_numeric(
            df["n_samples_per_waveform"],
            errors="coerce",
        )
        mask = frame_lengths == REQUIRED_N_SAMPLES_PER_WAVEFORM
    elif "frame_type" in df.columns:
        mask = df["frame_type"].astype(str) == f"{REQUIRED_N_SAMPLES_PER_WAVEFORM}_samples"
    else:
        raise ValueError(
            "CSV has neither n_samples_per_waveform nor frame_type, "
            "so the 2500-sample selection cannot be applied."
        )

    kept = int(np.count_nonzero(mask))
    skipped = int(len(df) - kept)
    print(
        f"Keeping {kept} row(s) with {REQUIRED_N_SAMPLES_PER_WAVEFORM} samples; "
        f"ignoring {skipped} other row(s)."
    )

    return df.loc[mask].copy()


def mean_sem_summary(df, group_cols, value_col, unc_col=None):
    rows = []
    for group_key, group in df.groupby(group_cols):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        values = pd.to_numeric(group[value_col], errors="coerce")
        n = int(values.notna().sum())
        mean = values.mean()
        std = values.std(ddof=1) if n > 1 else 0.0
        sem = std / np.sqrt(n) if n > 0 else np.nan

        if unc_col is not None and unc_col in group.columns and n > 0:
            uncs = pd.to_numeric(group[unc_col], errors="coerce").fillna(0.0)
            propagated_unc_on_mean = np.sqrt(np.sum(uncs ** 2)) / n
        else:
            propagated_unc_on_mean = np.nan

        row = {
            value_col + "_mean": mean,
            value_col + "_std": std,
            value_col + "_sem": sem,
            value_col + "_propagated_unc_on_mean": propagated_unc_on_mean,
            "n_runs": n,
        }
        for col, val in zip(group_cols, group_key):
            row[col] = val
        rows.append(row)

    return pd.DataFrame(rows)


def mean_value_and_uncertainty(group, value_col, unc_col=None):
    values = pd.to_numeric(group[value_col], errors="coerce")
    n = int(values.notna().sum())

    if n == 0:
        return np.nan, np.nan

    mean = values.mean()

    if unc_col is not None and unc_col in group.columns:
        uncs = pd.to_numeric(group[unc_col], errors="coerce").dropna()
        if len(uncs) > 0:
            return mean, np.sqrt(np.sum(uncs ** 2)) / n

    std = values.std(ddof=1) if n > 1 else 0.0
    return mean, std / np.sqrt(n)


def format_value_with_uncertainty(value, uncertainty):
    if not np.isfinite(value):
        return r"\mathrm{n/a}"

    if value == 0:
        if np.isfinite(uncertainty) and uncertainty > 0:
            return rf"(0.000 \pm {uncertainty:.3g})"
        return "0"

    exponent = int(np.floor(np.log10(abs(value))))
    scale = 10.0 ** exponent
    mantissa = value / scale

    if not np.isfinite(uncertainty) or uncertainty <= 0:
        return rf"{mantissa:.3f}\times 10^{{{exponent}}}"

    uncertainty_mantissa = uncertainty / scale
    return (
        rf"({mantissa:.3f} \pm {uncertainty_mantissa:.3f})"
        rf"\times 10^{{{exponent}}}"
    )


def add_fraction_above_with_poisson_unc(df):
    """Add fraction above threshold and binomial/statistical uncertainty.

    The fraction is computed as n_above / n_separable if counts are present.
    This is preferred over the rate ratio for the uncertainty because the
    above-threshold candidates are a subset of all separable candidates.
    """
    df = df.copy()

    if {"n_ap_separable_above_thr", "n_ap_separable"}.issubset(df.columns):
        n_above = pd.to_numeric(df["n_ap_separable_above_thr"], errors="coerce")
        n_total = pd.to_numeric(df["n_ap_separable"], errors="coerce")
        frac = n_above / n_total
        frac = frac.replace([np.inf, -np.inf], np.nan)
        unc = np.sqrt(frac * (1.0 - frac) / n_total)
        unc = unc.replace([np.inf, -np.inf], np.nan)
        df["frac_above_2pe"] = frac
        df["frac_above_2pe_unc"] = unc
    else:
        frac = df[RATE_ABOVE] / df[RATE_TOTAL]
        df["frac_above_2pe"] = frac.replace([np.inf, -np.inf], np.nan)
        df["frac_above_2pe_unc"] = np.nan

    return df


def make_summary_figure(figsize=(12.0, 6.0)):
    """Create plot axis plus unboxed right-side information column."""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[3.7, 1.0],
        left=0.08,
        right=0.98,
        bottom=0.16,
        top=0.88,
        wspace=0.10,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis("off")
    ax_info.axvline(0.0, color="0.75", linewidth=0.8)
    return fig, ax, ax_info


def add_summary_side_text(ax_info, text, fontsize=9.5):
    ax_info.text(
        0.06,
        0.98,
        text,
        transform=ax_info.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        linespacing=1.25,
    )


def build_summary_text(df):
    lines = [
        r"$\mathbf{Summary}$",
        r"$\mathrm{Metric:}\ R_{\mathrm{AP}}/\mathrm{PE}\ \mathrm{(separable)}$",
        rf"$A_{{\mathrm{{thr}}}} = {AREA_THRESHOLD_PE}\,\mathrm{{PE}}$",
        rf"$N_{{\mathrm{{frame}}}} = {REQUIRED_N_SAMPLES_PER_WAVEFORM}\,\mathrm{{samples}}$",
    ]

    if "n_samples" in df.columns:
        lines.append(r"$N_{\mathrm{wf}} = \mathrm{full\ dataset}$")

    if "sum_p0_area_pe" in df.columns:
        lines.append(r"$\mathrm{Normalization:}\ \sum A_{P0}\ \mathrm{in\ PE}$")

    # if "gain_source" in df.columns:
    #     gain_sources = sorted(str(x) for x in df["gain_source"].dropna().unique())
    #     if len(gain_sources) == 1:
    #         lines.append(rf"$\mathrm{{Gain:}}\ \mathrm{{{gain_sources[0]}}}$")

    if {
        "min_afterpulse_delay_samples",
        "conditional_delay_min_samples",
        "conditional_delay_max_samples",
        "conditional_min_peak_height_adc",
    }.issubset(df.columns):
        min_delay = int(pd.to_numeric(df["min_afterpulse_delay_samples"], errors="coerce").dropna().iloc[0])
        win_min = int(pd.to_numeric(df["conditional_delay_min_samples"], errors="coerce").dropna().iloc[0])
        win_max = int(pd.to_numeric(df["conditional_delay_max_samples"], errors="coerce").dropna().iloc[0])
        adc_thr = int(pd.to_numeric(df["conditional_min_peak_height_adc"], errors="coerce").dropna().iloc[0])
        lines.extend([
            "",
            r"$\mathbf{Peak\ selection}$",
            rf"$\Delta n_{{\min}} = {min_delay}\,\mathrm{{samples}}$",
            rf"${win_min}\leq\Delta n\leq{win_max}:"
            rf"\ A_{{\mathrm{{peak}}}}\geq{adc_thr}\,\mathrm{{ADC}}$",
        ])
        
    if {"pmt_serial", RATE_TOTAL, RATE_ABOVE, RATE_BELOW}.issubset(df.columns):
        lines.extend([
            "",
            r"$\mathbf{Afterpulse\ rates}$",
        ])

        pmt_order = pmt_order_from_serials(df["pmt_serial"])

        for pmt in pmt_order:
            df_pmt = df[df["pmt_serial"] == pmt]

            r_total, r_total_unc = mean_value_and_uncertainty(
                df_pmt,
                RATE_TOTAL,
                rate_unc_col(RATE_TOTAL),
            )

            lines.append(
                rf"$\mathrm{{{pmt}}}: "
                rf"{format_value_with_uncertainty(r_total, r_total_unc)}$"
            )

    return "\n".join(lines)


def save_split_threshold_plot(df, output_dir, summary_text):
    total = mean_sem_summary(df, ["pmt_serial"], RATE_TOTAL, rate_unc_col(RATE_TOTAL))
    above = mean_sem_summary(df, ["pmt_serial"], RATE_ABOVE, rate_unc_col(RATE_ABOVE))
    below = mean_sem_summary(df, ["pmt_serial"], RATE_BELOW, rate_unc_col(RATE_BELOW))

    pmt_order = pmt_order_from_serials(df["pmt_serial"])
    x = np.arange(len(pmt_order))
    width = 0.25

    def ordered(summary, rate_col, error_kind=ERROR_BAR_KIND):
        indexed = summary.set_index("pmt_serial").reindex(pmt_order)
        y = indexed[rate_col + "_mean"].values
        if SHOW_ERROR_BARS:
            yerr = indexed[rate_col + "_" + error_kind].fillna(0.0).values
        else:
            yerr = None
        return y, yerr

    total_y, total_err = ordered(total, RATE_TOTAL)
    above_y, above_err = ordered(above, RATE_ABOVE)
    below_y, below_err = ordered(below, RATE_BELOW)

    fig, ax, ax_info = make_summary_figure()

    ax.bar(
        x - width,
        total_y,
        width=width,
        yerr=total_err,
        capsize=4,
        error_kw={"ecolor": "black", "elinewidth": 1.0, "capthick": 1.0},
        label=r"$\mathrm{Total\ separable}$",
    )
    ax.bar(
        x,
        above_y,
        width=width,
        yerr=above_err,
        capsize=4,
        error_kw={"ecolor": "black", "elinewidth": 1.0, "capthick": 1.0},
        label=rf"$A_{{\mathrm{{AP}}}} \geq {AREA_THRESHOLD_PE}\,\mathrm{{PE}}$",
    )
    ax.bar(
        x + width,
        below_y,
        width=width,
        yerr=below_err,
        capsize=4,
        error_kw={"ecolor": "black", "elinewidth": 1.0, "capthick": 1.0},
        label=rf"$A_{{\mathrm{{AP}}}} < {AREA_THRESHOLD_PE}\,\mathrm{{PE}}$",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(pmt_order, rotation=45)
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
    add_summary_side_text(ax_info, summary_text)

    fig.savefig(
        output_dir / f"PMT_comparison_split_threshold{OUTPUT_SUFFIX}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_runs_scatter_plot(df, output_dir, summary_text):
    pmt_order = pmt_order_from_serials(df["pmt_serial"])

    fig, ax, ax_info = make_summary_figure()

    for pmt in pmt_order:
        df_sub = df[df["pmt_serial"] == pmt]
        x = np.full(len(df_sub), pmt_order.index(pmt), dtype=float)
        if len(df_sub) > 1:
            x += np.linspace(-0.06, 0.06, len(df_sub))
        y = pd.to_numeric(df_sub[RATE_TOTAL], errors="coerce").values
        if SHOW_ERROR_BARS and rate_unc_col(RATE_TOTAL) in df_sub.columns:
            yerr = pd.to_numeric(
                df_sub[rate_unc_col(RATE_TOTAL)],
                errors="coerce",
            ).fillna(0.0).values
        else:
            yerr = None

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            capsize=3,
            linewidth=1.0,
            markersize=5,
            label=pmt,
        )

    ax.set_xticks(np.arange(len(pmt_order)))
    ax.set_xticklabels(pmt_order, rotation=45)
    ax.set_ylabel(r"$R_{\mathrm{AP}}/\mathrm{PE}\ \mathrm{(separable)}$")
    ax.set_xlabel(r"$\mathrm{PMT}$")
    ax.set_title(r"$\mathrm{Run\!-\!by\!-\!run\ afterpulse\ rate\ per\ PE}$", fontsize=16)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(1.01, 0.02),
        borderaxespad=0.0,
        fontsize=10,
        frameon=True,
    )
    add_summary_side_text(ax_info, summary_text)

    fig.savefig(
        output_dir / f"PMT_runs_scatter{OUTPUT_SUFFIX}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_fraction_above_plot(df, output_dir, summary_text):
    df = add_fraction_above_with_poisson_unc(df)
    frac = mean_sem_summary(df, ["pmt_serial"], "frac_above_2pe", "frac_above_2pe_unc")

    pmt_order = pmt_order_from_serials(df["pmt_serial"])
    frac = frac.set_index("pmt_serial").reindex(pmt_order).reset_index()

    yerr = (
        frac["frac_above_2pe_" + ERROR_BAR_KIND].fillna(0.0).values
        if SHOW_ERROR_BARS
        else None
    )

    fig, ax, ax_info = make_summary_figure()

    ax.bar(
        frac["pmt_serial"],
        frac["frac_above_2pe_mean"],
        yerr=yerr,
        capsize=4,
        error_kw={"ecolor": "black", "elinewidth": 1.0, "capthick": 1.0},
    )
    ax.set_ylabel(rf"$f(A_{{\mathrm{{AP}}}} \geq {AREA_THRESHOLD_PE}\,\mathrm{{PE}})$")
    ax.set_xlabel(r"$\mathrm{PMT}$")
    ax.set_title(r"$\mathrm{Large\!-\!afterpulse\ fraction\ per\ PMT}$", fontsize=16)
    ax.tick_params(axis="x", rotation=45)
    add_summary_side_text(ax_info, summary_text)

    fig.savefig(
        output_dir / f"PMT_fraction_above_2pe{OUTPUT_SUFFIX}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_frame_type_plot(df, output_dir, summary_text):
    if "frame_type" not in df.columns:
        print("No frame_type column found, skipping frame-type plot.")
        return

    frame = mean_sem_summary(df, ["pmt_serial", "frame_type"], RATE_TOTAL, rate_unc_col(RATE_TOTAL))
    pmt_order = pmt_order_from_serials(df["pmt_serial"])

    fig, ax, ax_info = make_summary_figure()

    for frame_type in sorted(frame["frame_type"].dropna().unique()):
        df_sub = frame[frame["frame_type"] == frame_type].set_index("pmt_serial").reindex(pmt_order)
        x = np.arange(len(pmt_order))
        y = df_sub[RATE_TOTAL + "_mean"].values
        yerr = (
            df_sub[RATE_TOTAL + "_" + ERROR_BAR_KIND].fillna(0.0).values
            if SHOW_ERROR_BARS
            else None
        )
        ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=4, label=frame_type)

    ax.set_xticks(np.arange(len(pmt_order)))
    ax.set_xticklabels(pmt_order, rotation=45)
    ax.set_ylabel(r"$R_{\mathrm{AP}}/\mathrm{PE}\ \mathrm{(separable)}$")
    ax.set_xlabel(r"$\mathrm{PMT}$")
    ax.set_title(r"$\mathrm{Afterpulse\ rate\ per\ PE:\ separated\ by\ frame\ size}$", fontsize=16)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(1.01, 0.02),
        borderaxespad=0.0,
        fontsize=10,
        frameon=True,
    )
    add_summary_side_text(ax_info, summary_text)

    fig.savefig(
        output_dir / f"PMT_comparison_frame_types{OUTPUT_SUFFIX}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_summary_tables(df, output_dir):
    tables = {}

    for rate_col in [RATE_TOTAL, RATE_ABOVE, RATE_BELOW]:
        tables[rate_col] = mean_sem_summary(df, ["pmt_serial"], rate_col, rate_unc_col(rate_col))
        tables[rate_col].to_csv(
            output_dir / f"{rate_col}_summary{OUTPUT_SUFFIX}.csv",
            index=False,
        )

    df_frac = add_fraction_above_with_poisson_unc(df)
    frac_table = mean_sem_summary(df_frac, ["pmt_serial"], "frac_above_2pe", "frac_above_2pe_unc")
    frac_table.to_csv(
        output_dir / f"frac_above_2pe_summary{OUTPUT_SUFFIX}.csv",
        index=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Make better afterpulse summary plots from afterpulse_summary.csv."
    )
    parser.add_argument(
        "summary_csv",
        nargs="?",
        default=str(DEFAULT_SUMMARY_CSV),
        help="Path to afterpulse_summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for plots. Defaults to the CSV parent directory.",
    )
    args = parser.parse_args()

    summary_csv = Path(args.summary_csv)
    output_dir = Path(args.output_dir) if args.output_dir else summary_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_csv)
    needed_numeric = [
        RATE_TOTAL,
        RATE_ABOVE,
        RATE_BELOW,
        rate_unc_col(RATE_TOTAL),
        rate_unc_col(RATE_ABOVE),
        rate_unc_col(RATE_BELOW),
        "gain",
        "gain_unc",
        "n_samples",
        "n_samples_per_waveform",
        "sum_p0_area_pe",
        "n_ap",
        "n_ap_separable",
        "n_ap_separable_above_thr",
        "n_ap_separable_below_thr",
        "min_afterpulse_delay_samples",
        "conditional_delay_min_samples",
        "conditional_delay_max_samples",
        "conditional_min_peak_height_adc",
    ]
    df = numeric(df, needed_numeric)
    df = keep_required_frame_length(df)

    if df.empty:
        raise ValueError(
            f"No rows with {REQUIRED_N_SAMPLES_PER_WAVEFORM} samples found in CSV."
        )

    df = ensure_rate_uncertainties(df)

    required = {"pmt_serial", RATE_TOTAL, RATE_ABOVE, RATE_BELOW}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("Missing required columns in CSV: {}".format(", ".join(missing)))

    summary_text = build_summary_text(df)
    save_summary_tables(df, output_dir)
    save_split_threshold_plot(df, output_dir, summary_text)
    save_fraction_above_plot(df, output_dir, summary_text)

    print("Saved summary plots to: {}".format(output_dir))


if __name__ == "__main__":
    main()