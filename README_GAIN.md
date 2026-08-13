# PMT gain analysis

This five-script workflow extracts PMT gain from LED ON/OFF spectra, runs the analysis in batches, plots gain versus high voltage, studies SPE resolution, and creates a plain charge-spectrum figure.

## Scripts

| Script | Purpose |
|---|---|
| `Gain_analysis_all.py` | Analyzes one LED ON/OFF pair using model-independent gain extraction and a model-dependent multi-PE fit. |
| `Run_gain_analysis.py` | Finds matching SPE files and runs the gain analysis for all configured PMTs, folders, and lamp voltages. |
| `PMT_Summary_Gain_vs_HV.py` | Reads result JSONs, combines duplicate PMT/HV measurements, fits `G(V) = A V^k`, and writes comparison plots/CSVs. |
| `PMT_Summary_Width_vs_HV.py` | Plots SPE resolution `sigma1 / (mu1 - munoise)` versus HV from successful model-dependent fits. |
| `One_Spectrum.py` | Produces a simple LED ON/OFF integrated-charge spectrum without gain extraction or fit overlays. |

## Requirements and setup

Requires Python 3, `numpy`, `matplotlib`, `scipy`, and `pmt_analysis`.

Before running, configure:

- data paths and `PMT_FOLDERS` in the runner and spectrum-only script;
- the runner’s `ANALYSIS` path so it points to `Gain_analysis_all.py`;
- baseline, integration-window, and channel settings;
- `TARGET_LAMP_PER_PMT` consistently across batch and summary scripts;
- `BASE_OUTPUT_DIR`, which defaults to `Gain_Results` beside the analysis script.

## SPE naming convention

Input files must start with `SPE` and contain these parseable tokens:

- PMT serial: `LV` followed by four digits, for example `LV2480`;
- high voltage: three or four digits followed by `V`, optionally negative, for example `800V` or `-800V`;
- lamp voltage: `Lamp<number>Vpp`, with optional underscores, for example `Lamp1.90Vpp`, `Lamp1.90_Vpp`, or `Lamp_1.90_Vpp`.

Example LED ON file:

```text
SPELV2480_basev1.00U0-800V_Lamp1.90Vpp_Loff_0.50V_Lwid_20ns_Lfreq700Hz_20260126_0_Module_0_0.root
```

Example LED OFF file:

```text
SPELV2480_basev1.00U0-800V_Lamp0.50Vpp_Lwid_20ns_Lfreq700Hz_20260126_0_Module_0_0.root
```

The scripts extract `LV2480`, `-800 V`, and the value immediately following `Lamp`. A file is treated as OFF when that lamp value is 0.50 Vpp; all other selected lamp values are ON. Keep the PMT/HV/lamp tokens unambiguous and ensure result filenames retain them, since the HV and resolution plotters parse metadata from the generated JSON filenames.

The runner selects one ON lamp per PMT from `TARGET_LAMP_PER_PMT`. Within each data folder, it chooses one 0.50 Vpp OFF file per PMT and reuses it for all selected ON files. The OFF file is not required to have the same HV as the ON file. Use `REQUIRE_TAG` when selection must also include a substring such as `700Hz` or `-800V`.

## Single-pair gain analysis

Example:

```bash
python Gain_analysis_all.py \
  -pon /path/to/LED_ON.root \
  -poff /path/to/LED_OFF.root \
  -bbl 0 -bbu 100 -bpl 100 -c 0
```

Important options:

- `-bbl`, `-bbu`: baseline window bounds;
- `-bpl`, `-bpu`: signal integration window bounds;
- `-c`: ADC channel;
- `-tr`: enable or disable area-outlier trimming.

The script integrates each waveform with `FixedWindow`, then performs:

1. **Model-independent analysis:** LED OFF corrects the pedestal fraction, occupancy is inferred, and the mean signal area is converted to gain.
2. **Model-dependent analysis:** the OFF pedestal is fitted, then the ON spectrum is fitted with a pedestal Gaussian/EMG tail plus Poisson-constrained 1–6 PE Gaussian components.

Outputs are written under `Gain_Results/<PMT>/`:

- `<ON_basename>_entries_vs_area.png`: spectrum, fitted components, and residual panel;
- `<ON_basename>_results.json`: model-independent estimates and model-dependent fit parameters;
- `<ON_basename>_hist_entries.json`: histogram edges and ON/OFF counts.

## Batch execution

```bash
python Run_gain_analysis.py
```

The runner scans the configured folders for `SPE*` files, filters ON files by each PMT’s target lamp, selects one OFF file per PMT/folder, and invokes the single-pair analysis. It continues after missing files or failed analyses, so review all warnings and return codes.

## Gain versus high voltage

```bash
python PMT_Summary_Gain_vs_HV.py
```

The script scans:

```text
Gain_Results/LV*/*_results.json
```

Set `GAIN_SOURCE` to `model_dependent` or `model_independent`. Results are filtered by PMT, lamp, and optional HV list. Duplicate PMT/HV values are inverse-variance averaged when all uncertainties are valid; otherwise their median is used and the spread estimates the uncertainty.

When enabled, the fit uses

\[
G(V)=A V^k
\]

through a linear fit in log-log space. Outputs include the gain/HV plot, the selected gain-point CSV, and a power-law-fit CSV in `Gain_Results`.

## SPE resolution versus high voltage

```bash
python PMT_Summary_Width_vs_HV.py
```

The plotted resolution is

\[
\frac{\sigma}{\mu}=
\frac{\sigma_1}{\mu_1-\mu_{\mathrm{noise}}}
=\frac{\sigma_1}{d\mu_{\mathrm{ADC}}}.
\]

Only successful model-dependent fits with finite parameters and `0 < sigma/mu <= 1` are used. The stored uncertainty is approximate because it propagates only `dmu_adc_err`; no `sigma1` uncertainty is available in the JSON.

Before running, fix the inconsistent path variable in this script: replace `BASE_RESULTS_DIR` with `BASE_OUTPUT_DIR` (or define `BASE_RESULTS_DIR = BASE_OUTPUT_DIR`). It otherwise raises `NameError` before reading results.

Outputs are `sigma_over_mu_vs_high_voltage_all_pmts.png` and `.csv`.

## Spectrum-only plot

```bash
python One_Spectrum.py
```

Configure `TARGET_PMT`, `TARGET_HV`, `TARGET_LAMP_VPP`, `SELECTED_FOLDER`, and optionally `SELECTED_ON_FILE`. The script selects one ON file and one 0.50 Vpp OFF file, integrates both with the same windows, and saves a plain overlaid histogram. It performs no gain calculation or fit.

## Recommended order

1. Verify naming, integration windows, channel, and ON/OFF pairing.
2. Run the spectrum-only script for a visual sanity check.
3. Test `Gain_analysis_all.py` on one pair.
4. Run the batch runner and inspect failed/skipped files.
5. Check the generated fit figures and JSON metadata.
6. Produce gain-versus-HV and σ/μ-versus-HV summaries.

## Caveats

- Filename parsing controls ON/OFF classification, PMT/HV grouping, lamp filtering, and downstream plots.
- HV matching generally uses absolute value, so `800V` and `-800V` may be grouped together.
- The runner reuses one OFF file per PMT within a folder and does not require ON/OFF HV matching.
- Model-dependent results should be used only when `fit_success` is true and the fit/residual plot is credible.
- Power-law fitting requires positive gains and voltages and enough distinct HV points.
- Keep lamp selections consistent between batch analysis and summary plotting.
- Archive scripts, configuration, result JSONs, plots, logs, and the `pmt_analysis` version with reported results.


