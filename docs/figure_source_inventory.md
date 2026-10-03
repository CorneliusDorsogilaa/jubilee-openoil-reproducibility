# Figure source inventory

This record maps each submitted main-manuscript and Supporting Information figure to the final archived source data and the analysis or plotting code retained for provenance.

The machine-readable version is `metadata/figure_source_inventory.csv`.

## Final archive state

The submission figure-source package is complete. Large OpenOil NetCDF outputs are intentionally not redistributed. The archive instead preserves the authoritative analysis code, submitted-figure source tables, validation diagnostic tables, and compact trajectory/per-element exports needed to verify the plotted results.

For Figures 1, 3, 4, and 6, the original figure builders are retained as provenance scripts that recompute source tables from the local raw NetCDF outputs. Because those raw NetCDFs are not part of the public archive, the committed CSV/GZIP source tables are the frozen submission-matched figure sources. Figure 2 is directly rebuilt from archived CSVs. Figure 5 is directly rebuilt from the archived mechanism table.

## Main manuscript

| Figure | Final archived source | Retained code | Archive status |
|---|---|---|---|
| Fig. 1 | `archive_sources/Fig1_reference_plot_source.csv.gz`; `derived_data/Fig1_reference_annotation_metrics.csv` | `scripts/figures/build_Fig1_reference_production_map.py`; `scripts/archive/90_export_compact_archive_sources.py` | COMPLETE |
| Fig. 2 | `derived_data/Figure2_ensemble_absolute_metrics.csv`; `derived_data/Figure2_paired_A90_changes.csv` | `scripts/figures/build_Fig2_ensemble_process_attribution.py` | COMPLETE |
| Fig. 3 | `derived_data/Fig3_vertical_timeseries.csv`; `derived_data/Fig3_vertical_depth_ensemble.csv`; `diagnostics/validation/vertical_depth_per_seed.csv`; `diagnostics/validation/vertical_depth_summary.csv` | `scripts/figures/build_Fig3_vertical_exchange.py`; `scripts/robustness/76_analyze_vertical_depth_percentiles.py` | COMPLETE |
| Fig. 4 | `derived_data/Fig4_C3_fate_timeseries.csv` | `scripts/figures/build_Fig4_C3_fate_evolution.py` | COMPLETE |
| Fig. 5 | `derived_data/Fig5_weathering_entrainment_mechanism.csv`; `diagnostics/validation/weathering_entrainment_mechanism_timeseries.csv` | `scripts/figures/build_Fig5_weathering_entrainment_mechanism.py`; `scripts/mechanism/62_reconstruct_weathering_entrainment_mechanism.py` | COMPLETE |
| Fig. 6 | `derived_data/Fig6_timestep_screen.csv`; `derived_data/Fig6_particle_count_ensemble.csv`; `diagnostics/validation/particle_count_convergence_per_seed.csv`; `diagnostics/validation/particle_count_convergence_summary.csv`; `diagnostics/validation/particle_count_convergence_paired.csv` | `scripts/figures/build_Fig6_numerical_robustness.py`; `scripts/robustness/70_run_particle_count_ensemble.py`; `scripts/robustness/71_analyze_particle_count_convergence.py` | COMPLETE |

## Supporting Information

| Figure | Final archived source | Retained code | Archive status |
|---|---|---|---|
| Fig. S1 | `diagnostics/validation/vertical_mixing_summary.csv`; `diagnostics/validation/vertical_mixing_histograms.csv`; `diagnostics/validation/vertical_mixing_diffusivity_profiles.csv` | `scripts/validation/86_check_vertical_mixing_and_wind_threshold.py` | COMPLETE |
| Fig. S2 | `diagnostics/validation/wind_threshold_occupancy.csv`; `diagnostics/validation/wind_occupancy_lag_correlations.csv`; `derived_data/Fig3_vertical_timeseries.csv` | `scripts/validation/86_check_vertical_mixing_and_wind_threshold.py` | COMPLETE |
| Fig. S3 | `derived_data/FigS3_oil_sensitivity_summary.csv`; `derived_data/FigS3_oil_property_summary.csv`; `derived_data/FigS3_reference_variability.csv`; `diagnostics/validation/three_oil_properties.csv`; `diagnostics/validation/three_oil_metrics.csv`; `archive_sources/FigS3_per_element_surface_time.csv.gz`; `archive_sources/FigS3_reference_offsets.csv` | `scripts/oil_sensitivity/82_analyze_three_oil_sensitivity.py`; `scripts/archive/90_export_compact_archive_sources.py` | COMPLETE |

## Raw-output archive policy

The final public archive does **not** redistribute the large raw OpenOil NetCDF outputs or the ERA5/GLORYS forcing files. The repository instead contains:

1. exact production configuration and environment records,
2. authoritative analysis and figure-generation code,
3. frozen submission-matched figure-source tables,
4. per-seed and validation diagnostic tables,
5. compact trajectory and per-element exports for figures that otherwise depend directly on raw NetCDFs.

A full model rerun requires the documented ERA5 and GLORYS products under their providers' access and licensing conditions.

## Freeze order

After any final repository edit, regenerate `metadata/file_manifest_sha256.csv`. Only then create and tag `v1.0.0-paper-submission`, archive that release in Zenodo, and insert the resulting DOI into the manuscript and Supporting Information.
