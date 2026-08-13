# Full-data PMT afterpulse analysis

Four scripts measure PMT afterpulsing from full waveform datasets, recreate summary plots, add an uncertainty budget, and inspect secondary peaks close to the main pulse.

The main result is the separable afterpulse rate per photoelectron:

\[
R_{\mathrm{AP}}/\mathrm{PE}=
\frac{N_{\mathrm{AP,separable}}}{\sum A_{P0}\,[\mathrm{PE}]}.
\]

Rates are also split into afterpulse areas below and above 2 PE by default.

## Scripts

| Script | Purpose |
|---|---|
| `AP_Run_anaysis_better_plots.py` | Runs the full waveform analysis, writes `afterpulse_summary.csv`, and creates run and PMT plots. |
| `AP_make_summary_plots_from_csv.py` | Recreates comparison plots and tables from the summary CSV without rerunning waveform analysis. |
| `AP_postprocess_uncertainties.py` | Adds gain propagation and PMT-level statistical/systematic uncertainty estimates. |
| `AP_plot_early_secondary_pulses_side_info_style.py` | Inspects peaks 30–60 samples after the main pulse to test the ringing/recovery cut. |

If the local filenames differ, the roles and execution order are unchanged.

## Requirements

- Python 3
- `numpy`, `pandas`, `matplotlib`, `uproot`
- `scipy` and `tqdm` for the early-peak diagnostic
- `pmt_analysis`

The installed `pmt_analysis` must support the adjustable early-window arguments passed to `AfterPulses`.

## Input layout and configuration

The scripts expect dated data directories containing `APLV...` run folders:

```text
BASE_PATH/
└── data_YYYYMMDD/
    └── APLV.../
        └── APLV..._Module_0_*.root
```

Run names must contain a PMT serial such as `LV2483`, an HV such as `800V`, and a lamp value such as `Lamp1.95Vpp`.

Before running, check:

- `BASE_PATH` and `PMT_FOLDERS`;
- `BASE_SAVE_DIR` in every script;
- `TARGET_LAMP_PER_PMT` for afterpulse-run selection;
- `GAIN_RESULTS_DIRS` and `TARGET_GAIN_LAMP_PER_PMT` for gain lookup;
- `SUMMARY_CSV` or `DEFAULT_SUMMARY_CSV` in the post-processing scripts.

The analysis, plotting, and uncertainty scripts use independent path settings. Make sure they all point to the same `afterpulse_summary.csv`.

## Gain lookup and run selection

The main analysis searches `GAIN_RESULTS_DIRS` recursively for `*_results.json`. It matches each run by PMT, HV, and the configured gain lamp. It prefers:

1. model-dependent gain;
2. a successful fit;
3. the closest lamp match;
4. the newest result.

If no valid gain exists, the run is skipped. The selected value, uncertainty, source, and JSON path are stored in the summary CSV.

Runs are selected from `APLV...` folders whose lamp matches `TARGET_LAMP_PER_PMT`. ROOT branch/channel 0 is read through `ADCRawData`. Waveforms with 3500 samples are cropped to the first 1500; other lengths are retained and recorded.

## Full-data analysis

For every accepted run, the main script:

1. loads waveforms and ADC metadata;
2. selects the matching gain;
3. processes all waveforms in chunks;
4. finds peaks and constrains the main pulse;
5. calculates afterpulse properties, multiplicities, and rates;
6. combines counts and main-pulse normalization across chunks;
7. makes plots from a smaller waveform subset;
8. adds one row to `afterpulse_summary.csv`.

Key defaults:

| Setting | Default |
|---|---:|
| Area split | 2 PE |
| Peak height | 50 ADC |
| Peak distance | 25 samples |
| Conditional early region | 10–60 samples |
| Minimum peak height in that region | 100 ADC |
| Analysis chunk | 20,000 waveforms |
| Plotting subset | 10,000 waveforms |

The full dataset determines the statistics; chunking only limits memory use. Extra zoom plots may clip the main pulse to make smaller afterpulses visible, but this does not affect the calculation.

### Main CSV fields

- Run and calibration: `pmt_serial`, `run_name`, `hv_v`, `lamp_v`, `gain`, `gain_unc`, `gain_source`, `gain_result_file`
- Exposure: `n_samples`, `n_main_pulses`, `frame_type`, `n_samples_per_waveform`
- Normalization: `mean_p0_area_pe`, `sum_p0_area_pe`
- Counts: `n_ap`, `n_ap_separable`, and above/below-threshold counts
- Rates: `ap_rate_per_pe`, `ap_rate_per_pe_separable`, `ap_rate_per_pe_above_thr`, `ap_rate_per_pe_below_thr`
- Counting uncertainties: corresponding `_unc` columns

Here `n_samples` means analyzed waveforms/triggers, not ADC time bins.

Per-run counting errors use

\[
\sigma_{N/D}=\frac{\sqrt N}{D},
\]

with the normalization treated as fixed.

## Summary plots from CSV

```bash
python AP_make_summary_plots_from_csv.py /path/to/afterpulse_summary.csv \
  --output-dir /path/to/plots
```

This creates PMT threshold comparisons, run scatter, large-afterpulse fractions, frame-length comparisons, and PMT summary tables. It reports run means, standard deviations, SEMs, and propagated per-run errors on the mean.

For the fraction above 2 PE, the preferred calculation is

\[
f=\frac{N_{\ge 2\mathrm{PE}}}{N_{\mathrm{separable}}},
\qquad
\sigma_f=\sqrt{\frac{f(1-f)}{N_{\mathrm{separable}}}}.
\]

This script only replots existing results. Changes to gains, cuts, or waveform processing require rerunning the main analysis.

## Uncertainty post-processing

Set `SUMMARY_CSV`, then run:

```bash
python AP_postprocess_uncertainties.py
```

Gain propagation uses

\[
\sigma_{R,G}=|R|\frac{\sigma_G}{G}.
\]

The PMT summaries contain run-to-run standard deviation and SEM, propagated Poisson uncertainty, correlated gain systematic, frame-size systematic, and optional cut-variation systematic. Independent components are combined in quadrature in several clearly named columns.

To include cut systematics, add labeled nominal and alternative-cut CSVs to `CUT_VARIATION_SUMMARY_FILES`. An empty list makes this contribution zero; it does not establish that the effect is negligible.

Main outputs:

- `afterpulse_summary_with_uncertainties.csv`
- `<rate>_pmt_uncertainty_summary.csv`
- `PMT_comparison_split_threshold_with_uncertainties.png`
- `PMT_uncertainty_budget_separable.png`

The gain term covers PE normalization but not migration across the 2 PE boundary.

## Early-secondary-peak diagnostic

This script independently checks whether peaks 30–60 samples after the main pulse resemble ringing or recovery structure. It does not call `AfterPulses.find_ap()`.

It baseline-subtracts and sign-inverts each waveform, applies a noise prefilter, calls `scipy.signal.find_peaks()`, treats the first peak as the main pulse, and plots later peaks satisfying `30 <= delay < 60` samples.

Outputs include zoomed waveform PNGs, per-run candidate CSVs, `early_secondary_candidates_all_runs.csv`, and `early_secondary_candidate_counts.csv`.

By default, only four candidates are saved per PMT and later runs may be skipped after that limit. Therefore, the default counts are diagnostic—not an unbiased early-pulse rate.
