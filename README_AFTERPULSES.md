# PMT afterpulse analysis

This four-script workflow analyzes full PMT waveform datasets, plots the results, adds uncertainties, and checks early secondary peaks that may be ringing.

## Scripts

| Script | Purpose |
|---|---|
| `Run_Afterpulse_Analysis.py` | Full analysis; creates `afterpulse_summary.csv` and diagnostic plots. |
| `AP_Summary.py` | Recreates PMT comparison plots and tables from the CSV. |
| `AP_Error.py` | Adds gain, run-to-run, frame, and optional cut uncertainties. |
| `AP_Close_to_main.py` | Inspects secondary peaks 30–60 samples after the main pulse. |

## Requirements and setup

Requires Python 3, `numpy`, `pandas`, `matplotlib`, `uproot`, `scipy`, `tqdm`, and `pmt_analysis`.

Expected input layout:

```text
BASE_PATH/data_YYYYMMDD/APLV.../APLV..._Module_0_*.root
```

### Naming convention

Each run directory must:

- start with `APLV`;
- contain the PMT serial as `LV` plus four digits, for example `LV2483`;
- contain the high voltage as three or four digits followed by `V`, for example `800V` or `-800V`;
- contain the lamp voltage in a recognized form, such as `Lamp1.95Vpp`, `Lamp_1.95_Vpp`, or `Lamp1.95_Vpp`.

Example:

```text
data_20260306/
└── APLV2483_Lamp1.95Vpp_800V/
    ├── APLV2483_Lamp1.95Vpp_800V_Module_0_000.root
    └── APLV2483_Lamp1.95Vpp_800V_Module_0_001.root
```

The directory name is the `run_name`. Its ROOT files must repeat that name exactly, followed by `_Module_0_` and any file suffix:

```text
<run_name>/<run_name>_Module_0_*.root
```

The scripts extract `LV2483`, `1.95 Vpp`, and `800 V` from the name. These values are used to select the requested lamp dataset and find the matching gain result. Avoid extra voltage-like text, missing units, or inconsistent directory/file prefixes. A name that cannot be parsed is skipped or fails gain matching.

Before running, configure:

- `BASE_PATH`, `PMT_FOLDERS`, and `BASE_SAVE_DIR`;
- `TARGET_LAMP_PER_PMT` for afterpulse runs;
- `GAIN_RESULTS_DIRS` and `TARGET_GAIN_LAMP_PER_PMT` for calibration;
- `SUMMARY_CSV` in the post-processing script.

All scripts that consume `afterpulse_summary.csv` must point to the main analysis output directory.

## Analysis

The main script selects `APLV...` runs at the configured lamp voltage and matches gain-result JSON files by PMT, HV, and gain lamp. Model-dependent gain is preferred; model-independent gain is the fallback. Runs without a valid gain are skipped.

All waveforms are analyzed in chunks; only a reduced subset is plotted. The default peak settings are 50 ADC height and 25 samples separation. Early candidates in the 10–60 sample region must satisfy the additional 100 ADC requirement. Waveforms with 3500 samples are cropped to 1500.

The primary result is

\[
R_{\mathrm{AP}}/\mathrm{PE}=
\frac{N_{\mathrm{AP,separable}}}{\sum A_{P0}\,[\mathrm{PE}]},
\]

split into afterpulse areas below and above 2 PE. Per-run count uncertainties use

\[
\sigma_R=\frac{\sqrt{N}}{\sum A_{P0}\,[\mathrm{PE}]}.
\]

The CSV records run metadata, gain provenance, frame length, waveform and candidate counts, main-pulse normalization, rates, threshold splits, and statistical uncertainties.

## Running

```bash
python AP_Close_to_main.py
python Run_Afterpulse_Analysis.py
python AP_Summary.py /path/to/afterpulse_summary.csv
python AP_Error.py
```

Recommended order: verify gains, inspect early peaks, run the full analysis, check skipped runs and CSV contents, regenerate summary plots, then calculate uncertainties.

## Outputs

- `afterpulse_summary.csv`: authoritative per-run results
- PMT threshold, run-scatter, large-pulse-fraction, and frame-length plots
- PMT summary CSVs generated from the main table
- `afterpulse_summary_with_uncertainties.csv`
- per-rate PMT uncertainty summaries and uncertainty-budget plots
- early-secondary waveform plots and candidate/count CSVs

## Uncertainties

The post-processing script adds:

- run-to-run standard deviation and SEM;
- propagated Poisson uncertainty;
- gain systematic, using `sigma_R,G = |R| sigma_G/G`;
- frame-length systematic;
- optional cut systematic from `CUT_VARIATION_SUMMARY_FILES`.

The gain term covers normalization but not migration across the 2 PE threshold. An empty cut-variation list makes that component zero; it does not demonstrate that the cut uncertainty is negligible.

## Early-peak diagnostic

The independent diagnostic baseline-subtracts and sign-inverts waveforms, finds peaks with SciPy, and plots candidates satisfying `30 <= delay < 60` samples. It does not use `AfterPulses.find_ap()`, allowing the early-time exclusion to be checked independently.

Only four candidates per PMT are saved by default, so its counts are diagnostic rather than an unbiased early-pulse rate.

## Caveats

- Directory and file naming controls discovery, run selection, and gain matching; review skipped-run messages.
- Short or cropped frames cannot contain late afterpulses.
- PMT means weight runs equally, not by exposure or inverse variance.
- Zero candidates produce a zero error in the simple Poisson approximation; use proper intervals for low-count limits.
- A single run gives zero run-to-run spread in the script, not proof of zero uncertainty.
- Archive the scripts, configuration, gain JSONs, logs, CSVs, and software versions with reported results.
