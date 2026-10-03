# Jubilee OpenOil Process Attribution

Reproducibility repository for the manuscript:

**Process Attribution in Offshore Oil Spill Transport and Weathering at the Jubilee Field, Ghana: A Stepwise OpenOil Configuration Study**

Authors: Cornelius Dorsogilaa, Shaibu Mohammed, Yaw Akyampon Boakye-Ansah, and Noel S. Dapilee.

## Status

This repository contains the submission-matched reproducibility package for the paper, including the analysis code, derived data, validation diagnostics, software-environment records, and figure-source tables used for the final manuscript and Supporting Information.

## Study design

The study is a controlled OpenDrift/OpenOil process-attribution experiment at the Jubilee Terminal FPSO location (4.60000° N, 2.89000° W). A 6 h continuous surface release within a 500 m radius is tracked for 48 h under common environmental forcing.

The four nested configurations are:

| Configuration | Process package |
|---|---|
| C0 | currents + horizontal diffusion |
| C1 | C0 + direct wind drift + Stokes drift |
| C2 | C1 + reversible vertical exchange |
| C3 | C2 + evaporation + emulsification + natural dispersion |

Production calculations use `N = 1000`, a 30 min transport timestep, hourly output, `Kh = 100 m² s⁻¹`, reference seed `20220901`, and Bonny Light (NOAA ADIOS AD01440) as a disclosed West African light-crude analogue. The primary stochastic inference uses 20 seeds (`20220901`–`20220920`).

## Important provenance notes

- Installed OpenDrift distribution: **1.14.12**.
- Bundled internal/module identifier written to production NetCDF: **1.14.11**.
- `Kh = 100 m² s⁻¹` is authoritative from the executed run configuration and filename/manifest. A queried NetCDF configuration attribute can be absent (`None`) and should not be interpreted as `Kh = 0` or as an unknown production setting.
- Production particle count is **N = 1000**. The transport-timestep screening campaign was run separately for **C3 at N = 2000** using 5, 10, 15, and 30 min timesteps and three seeds.
- C0–C3 use a common seed policy and common initialization, but activating additional stochastic processes changes subsequent random-number consumption. Ensemble inference is therefore used instead of assuming process-by-process matched random variates.
- The mechanism reconstruction contains an explicit elementwise Kelvin-to-Celsius guard because the archived production temperature field contains a mixed-unit final record. The post-release Ohnesorge ratio is approximately **465.8×**.
- OpenOil's NOAA evaporation routine stops further evaporation once the youngest active surface element exceeds 24 h of age. For the 6 h release, cumulative evaporation therefore plateaus after approximately 30 h.
- Bonny Light is an analogue used for executable OpenOil weathering, not an exact Jubilee crude assay.


## Verified source basis

The manuscript source verification is documented on three points that do not change any model result:

- The primary Jubilee crude property reference is the official **Tullow 2019 Jubilee assay** (37.41 °API, density 837.3 kg m⁻³ at 15 °C, kinematic viscosity 4.6 cSt at 40 °C, pour point 3 °C). Appenteng et al. (2013) is retained as an independent characterization showing that reported bulk properties vary with the sampled crude composite.
- The direct source for the Jubilee stochastic OILMAP precedent is the **2009 Jubilee Field Phase 1 EIS**, which documents 500 independent simulations with different start times for each spill scenario.
- Carvalho et al. (2025) is used only for the narrower sensitivity claim that API-gravity changes had limited influence on advective displacement but materially affected evaporation in MEDSLIK-II.

A concise verification record is provided in `docs/source_verification.md`.

## Repository layout

```text
scripts/
  production/       authoritative production runner
  robustness/       ensemble, Kh, particle-count, and depth analyses
  mechanism/        C2/C3 weathering-to-entrainment reconstruction
  oil_sensitivity/  ADIOS property extraction and three-oil sensitivity
  validation/       RNG, forcing-resolution, software-provenance, and well-mixed checks

derived_data/       source CSVs behind stable manuscript diagnostics
metadata/           production settings and seed inventory
docs/               reproducibility and archive notes
```

## External forcing data

ERA5 and Copernicus Marine GLORYS12V1 forcing files are **not redistributed in this GitHub repository**. They remain subject to the access and licensing conditions of their respective providers. The reproducibility archive records the forcing window, variables, native sampling, and project configuration needed to retrieve and use the same products.

Production forcing window: **2020-03-01 00:00 UTC to 2020-03-03 00:00 UTC**.

## Core analyses

### Production configurations

```bash
python scripts/production/25_run_production_scenarios.py
```

### 20-seed particle-count robustness

```bash
python scripts/robustness/70_run_particle_count_ensemble.py
python scripts/robustness/71_analyze_particle_count_convergence.py
```

### Horizontal-diffusivity sensitivity

The final sensitivity figure uses the 20-seed paired C1→C2 results at `Kh = 10, 100, 1000 m² s⁻¹`.

```bash
python scripts/robustness/57_plot_Kh_sensitivity_pub.py
```

### Robust vertical-depth statistics

```bash
python scripts/robustness/76_analyze_vertical_depth_percentiles.py
```

### Corrected weathering-to-entrainment mechanism

```bash
python scripts/mechanism/62_reconstruct_weathering_entrainment_mechanism.py
```

The corrected reconstruction gives approximately:

- C3/C2 post-release kinematic-viscosity ratio: **769.4×**
- C3/C2 post-release Ohnesorge ratio: **465.8×**
- C3/C2 reconstructed entrainment-rate ratio: **0.0393**, i.e. **96.1% lower** in C3

### Three-oil sensitivity

The paper compares Qua Iboe (AD01483), Bonny Light (AD01440), and Cabinda Blend (AD01442) as discrete oil-record contrasts. They are **not** calibrated uncertainty bounds for Jubilee crude.

## Derived data

The `derived_data/` directory contains the frozen source tables used for the submitted manuscript diagnostics. Compact trajectory and per-element sources that replace redistribution of selected raw NetCDFs are stored in `archive_sources/`. A machine-readable figure-to-source map is stored in `metadata/figure_source_inventory.csv`, with a human-readable inventory in `docs/figure_source_inventory.md`. The final figure-source package is complete; the SHA256 manifest is regenerated only after all release files are frozen.

## Environment

Three environment records are supplied:

- `environment.yml` is the portable recreation file with the principal runtime dependencies pinned.
- `environment_exact.yml` is the final executed `jubilee-openoil-g2` Conda environment snapshot, including Python 3.11.16 and OpenDrift 1.14.12. The machine-specific Conda `prefix:` line was removed before publication because it points only to the original local installation path.
- `requirements_lock.txt` lists the pinned pip-installed packages extracted from the exact environment for users who need a pip-oriented package record.

For the closest recreation of the executed environment, use `environment_exact.yml`; `environment.yml` is the more portable starting point.

## Citation

A `CITATION.cff` file is included. Zenodo provides a version-specific DOI for each archived release.

## License

Code in this repository is released under the MIT License. Derived tables created by the authors may be reused with attribution; third-party environmental data and software remain under their original licenses.


## Raw-output archive policy

The public release is intentionally source-table based rather than a bulk dump of OpenOil NetCDF outputs. The repository archives the run configuration, authoritative analysis code, per-seed/diagnostic tables, and compact trajectory/per-element exports sufficient to redraw the submitted figures. Large raw OpenOil NetCDF outputs are not planned for the public Zenodo release. A full model rerun requires the documented ERA5 and GLORYS forcing products, which remain under their providers' access and licensing terms.
