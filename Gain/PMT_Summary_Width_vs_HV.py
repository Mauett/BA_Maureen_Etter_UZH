#!/usr/bin/env python3
"""
plot_sigma_over_mu_vs_hv_all_pmts.py

Build one sigma/mu vs High-Voltage plot for all PMTs together.

Definition used:
    sigma/mu = sigma1 / (mu1 - munoise)

where:
    sigma1   = width of the 1 PE Gaussian
    mu1      = mean of the 1 PE Gaussian
    munoise  = pedestal mean

This uses the model-dependent fit stored in:
    model_dependent_fit.params

Input:
- scans BASE_RESULTS_DIR recursively for *_results.json

Output:
- sigma_over_mu_vs_high_voltage_all_pmts.png
- sigma_over_mu_vs_high_voltage_all_pmts.csv
"""

import re
import json
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
BASE_RESULTS_DIR = Path(__file__).resolve().parent / "Gain_Results"
BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = BASE_RESULTS_DIR
OUTPUT_PLOT = OUTPUT_DIR / "sigma_over_mu_vs_high_voltage_all_pmts.png"
OUTPUT_CSV = OUTPUT_DIR / "sigma_over_mu_vs_high_voltage_all_pmts.csv"

REQUIRE_TAG = None
#REQUIRE_TAG = "2000Hz"

# Manual lamp-voltage selection per PMT.
# Only files with the matching Lamp voltage are used for that PMT.
# Set a PMT to None or remove it from the dict if you do not want filtering for it.
TARGET_LAMP_PER_PMT = {
    "LV2480": 1.90,
    "LV2481": 1.95,
    "LV2483": 1.95,
    "LV2485": 1.95,
}

VERBOSE = True

PMT_COLORS = {
    "LV2480": "mediumseagreen",
    "LV2481": "dodgerblue",
    "LV2483": "slateblue",
    "LV2485": "purple",
}

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def extract_hv_from_name(name: str):
    s = name.replace("-", "_")
    m = re.search(
        r"U0_([0-9]+(?:\.[0-9]+)?)V",
        s,
        re.IGNORECASE,
    )
    if m:
        return int(m.group(1))
    return None

def extract_lamp_from_name(name: str):
    """
    Extract lamp voltage from strings like:
      ..._Lamp1.9Vpp_...
      ..._Lamp1.85Vpp_...
    Returns float or None.
    """
    s = name.replace("-", "_")
    m = re.search(r"Lamp([0-9.]+)Vpp", s)
    if m:
        return float(m.group(1))
    return None

def extract_pmt_serial(path: Path):
    candidates = [
        path.name,
        path.stem,
        path.parent.name,
        path.parent.parent.name if path.parent.parent else "",
    ]
    for text in candidates:
        m = re.search(r"(LV\d{4})", text)
        if m:
            return m.group(1)
    return path.parent.name


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_finite_number(x):
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def extract_sigma_over_mu(data: dict):
    """
    Returns:
        sigma_over_mu, sigma_over_mu_err, params_dict
    or
        None, None, None
    """
    model_dep = data.get("model_dependent_fit", {})
    if not isinstance(model_dep, dict):
        return None, None, None
    if not model_dep.get("fit_success", False):
        return None, None, None

    params = model_dep.get("params", {})
    sigma1 = params.get("sigma1", None)
    dmu_adc = params.get("dmu_adc", None)
    dmu_adc_err = params.get("dmu_adc_err", None)

    if not is_finite_number(sigma1) or not is_finite_number(dmu_adc):
        return None, None, None

    sigma1 = float(sigma1)
    dmu_adc = float(dmu_adc)

    if dmu_adc <= 0:
        return None, None, None

    value = sigma1 / dmu_adc

    # Optional crude uncertainty:
    # only from dmu uncertainty, because sigma1_err is not stored in the JSON
    value_err = None
    if is_finite_number(dmu_adc_err):
        dmu_adc_err = float(dmu_adc_err)
        if dmu_adc_err >= 0:
            value_err = abs(sigma1 / (dmu_adc ** 2)) * dmu_adc_err

    return value, value_err, params


# ------------------------------------------------------------------
# COLLECT DATA
# ------------------------------------------------------------------
def collect_points(base_dir: Path, require_tag=None):
    grouped = defaultdict(lambda: defaultdict(list))

    files = sorted(p for p in base_dir.rglob("*_results.json") if p.is_file())

    if VERBOSE:
        print(f"Found {len(files)} results JSON files under: {base_dir}")

    for path in files:
        if require_tag and require_tag not in path.name:
            continue

        hv = extract_hv_from_name(path.name)
        if hv is None:
            if VERBOSE:
                print(f"Skipping (no HV found): {path}")
            continue

        pmt = extract_pmt_serial(path)

        lamp_v = extract_lamp_from_name(path.name)
        target_lamp = TARGET_LAMP_PER_PMT.get(pmt, None)

        if target_lamp is not None:
            if lamp_v is None:
                if VERBOSE:
                    print(f"Skipping (no Lamp voltage found) for {pmt}: {path.name}")
                continue

            if abs(lamp_v - target_lamp) > 1e-6:
                if VERBOSE:
                    print(
                        f"Skipping wrong Lamp voltage for {pmt}: "
                        f"Lamp {lamp_v} Vpp, expected {target_lamp} Vpp in {path.name}"
                    )
                continue

        try:
            data = load_json(path)
        except Exception as e:
            print(f"Skipping unreadable JSON: {path} ({e})")
            continue

        value, value_err, params = extract_sigma_over_mu(data)

        if not is_finite_number(value):
            if VERBOSE:
                print(f"Skipping (no usable sigma/mu): {path}")
            continue

        value = float(value)

        # physics sanity cut
        if value <= 0 or value > 1.0:
            if VERBOSE:
                print(f"Skipping unphysical sigma/mu={value:.3f}: {path}")
            continue

        grouped[pmt][hv].append({
            "sigma_over_mu": value,
            "sigma_over_mu_err": float(value_err) if is_finite_number(value_err) else None,
            "sigma1": float(params["sigma1"]) if is_finite_number(params.get("sigma1")) else None,
            "mu1": float(params["mu1"]) if is_finite_number(params.get("mu1")) else None,
            "munoise": float(params["munoise"]) if is_finite_number(params.get("munoise")) else None,
            "dmu_adc": float(params["dmu_adc"]) if is_finite_number(params.get("dmu_adc")) else None,
            "file": str(path),
        })

    per_pmt = defaultdict(list)

    for pmt, hv_dict in grouped.items():
        for hv, rows in hv_dict.items():
            values = [r["sigma_over_mu"] for r in rows]
            mean_value = float(np.mean(values))

            if len(values) > 1:
                spread_err = float(np.std(values, ddof=1))
            else:
                spread_err = None

            per_pmt[pmt].append({
                "hv": hv,
                "sigma_over_mu": mean_value,
                "sigma_over_mu_err": spread_err,
                "n_points": len(rows),
                "files": [r["file"] for r in rows],
            })

        per_pmt[pmt] = sorted(per_pmt[pmt], key=lambda d: d["hv"])

    return per_pmt


# ------------------------------------------------------------------
# SAVE CSV
# ------------------------------------------------------------------
def save_csv(per_pmt: dict, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "PMT", "HV_V", "SigmaOverMu_mean", "SigmaOverMu_spread",
            "N_points", "Results_JSON_files"
        ])

        for pmt in sorted(per_pmt):
            for row in per_pmt[pmt]:
                writer.writerow([
                    pmt,
                    row["hv"],
                    row["sigma_over_mu"],
                    row["sigma_over_mu_err"] if row["sigma_over_mu_err"] is not None else "",
                    row["n_points"],
                    " ; ".join(row["files"]),
                ])

    if VERBOSE:
        print(f"Saved CSV: {out_csv}")


# ------------------------------------------------------------------
# PLOT
# ------------------------------------------------------------------
def plot_sigma_over_mu(per_pmt: dict, out_plot: Path):
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted_any = False

    for pmt in sorted(per_pmt):
        rows = per_pmt[pmt]
        if not rows:
            continue

        hv = np.array([r["hv"] for r in rows], dtype=float)
        y = np.array([r["sigma_over_mu"] for r in rows], dtype=float)

        yerr_list = [r["sigma_over_mu_err"] for r in rows]
        has_any_err = any(v is not None for v in yerr_list)

        if has_any_err:
            yerr = np.array([v if v is not None else np.nan for v in yerr_list], dtype=float)
            color = PMT_COLORS.get(pmt, None)
            ax.plot(
                hv,
                y,
                marker="o",
                linewidth=1.5,
                label=pmt,
                color=color
            )
        else:
            color = PMT_COLORS.get(pmt, None)
            ax.plot(
                hv,
                y,
                marker="o",
                linewidth=1.5,
                label=pmt,
                color=color
            )

        plotted_any = True

    if not plotted_any:
        raise RuntimeError("No valid sigma/mu points found to plot.")

    ax.set_xlabel("High Voltage [V]")
    ax.set_ylabel(r"$\sigma_{\mathrm{SPE}} / \mu_{\mathrm{SPE}}$")
    ax.set_title("Relative SPE width vs High Voltage")
    ax.set_ylim(0.25, 0.65)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, fontsize=9)

    fig.subplots_adjust(right=0.78)
    fig.savefig(out_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if VERBOSE:
        print(f"Saved plot: {out_plot}")


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
if __name__ == "__main__":
    per_pmt = collect_points(BASE_RESULTS_DIR, require_tag=REQUIRE_TAG)

    if not per_pmt:
        raise RuntimeError("No PMT data found. Check BASE_RESULTS_DIR / REQUIRE_TAG / JSON content.")

    print("\nCollected PMTs:")
    for pmt in sorted(per_pmt):
        print(f"  {pmt}: {len(per_pmt[pmt])} point(s)")

    save_csv(per_pmt, OUTPUT_CSV)
    plot_sigma_over_mu(per_pmt, OUTPUT_PLOT)

    print("\n✅ Done.")
