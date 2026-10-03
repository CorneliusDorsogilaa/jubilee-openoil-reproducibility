#!/usr/bin/env python3
"""Verify the reproducibility release package and optionally regenerate SHA256 manifest.

Usage
-----
python scripts/archive/91_verify_release_package.py
python scripts/archive/91_verify_release_package.py --write-manifest

The manifest intentionally excludes itself and Git metadata so it is stable and
non-self-referential.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "metadata" / "file_manifest_sha256.csv"

REQUIRED = [
    "CITATION.cff",
    "LICENSE",
    "README.md",
    "environment.yml",
    "environment_exact.yml",
    "requirements_lock.txt",
    "docs/data_dictionary.md",
    "docs/figure_source_inventory.md",
    "docs/reproducibility_notes.md",
    "docs/source_verification.md",
    "metadata/figure_source_inventory.csv",
    "metadata/production_configuration.csv",
    "metadata/seed_inventory.csv",
    "archive_sources/Fig1_reference_plot_source.csv.gz",
    "archive_sources/FigS3_per_element_surface_time.csv.gz",
    "archive_sources/FigS3_reference_offsets.csv",
    "derived_data/Fig1_reference_annotation_metrics.csv",
    "derived_data/Figure2_ensemble_absolute_metrics.csv",
    "derived_data/Figure2_paired_A90_changes.csv",
    "derived_data/Fig3_vertical_timeseries.csv",
    "derived_data/Fig3_vertical_depth_ensemble.csv",
    "derived_data/Fig4_C3_fate_timeseries.csv",
    "derived_data/Fig5_weathering_entrainment_mechanism.csv",
    "derived_data/Fig6_timestep_screen.csv",
    "derived_data/Fig6_particle_count_ensemble.csv",
    "derived_data/FigS3_oil_property_summary.csv",
    "derived_data/FigS3_oil_sensitivity_summary.csv",
    "derived_data/FigS3_reference_variability.csv",
    "diagnostics/validation/particle_count_convergence_per_seed.csv",
    "diagnostics/validation/particle_count_convergence_summary.csv",
    "diagnostics/validation/particle_count_convergence_paired.csv",
    "diagnostics/validation/weathering_entrainment_mechanism_timeseries.csv",
    "diagnostics/validation/vertical_depth_per_seed.csv",
    "diagnostics/validation/vertical_depth_summary.csv",
    "diagnostics/validation/three_oil_properties.csv",
    "diagnostics/validation/three_oil_metrics.csv",
    "diagnostics/validation/vertical_mixing_summary.csv",
    "diagnostics/validation/vertical_mixing_histograms.csv",
    "diagnostics/validation/vertical_mixing_diffusivity_profiles.csv",
    "diagnostics/validation/wind_threshold_occupancy.csv",
    "diagnostics/validation/wind_occupancy_lag_correlations.csv",
    "scripts/archive/90_export_compact_archive_sources.py",
    "scripts/archive/91_verify_release_package.py",
    "scripts/figures/build_Fig1_reference_production_map.py",
    "scripts/figures/build_Fig2_ensemble_process_attribution.py",
    "scripts/figures/build_Fig3_vertical_exchange.py",
    "scripts/figures/build_Fig4_C3_fate_evolution.py",
    "scripts/figures/build_Fig5_weathering_entrainment_mechanism.py",
    "scripts/figures/build_Fig6_numerical_robustness.py",
]

EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache"}
EXCLUDED_RELATIVE = {"metadata/file_manifest_sha256.csv"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def repository_files() -> list[Path]:
    out = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel in EXCLUDED_RELATIVE:
            continue
        if any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts):
            continue
        out.append(path)
    return sorted(out, key=lambda p: p.relative_to(ROOT).as_posix())


def verify_required() -> None:
    missing = [p for p in REQUIRED if not (ROOT / p).is_file()]
    if missing:
        raise SystemExit("Missing required release files:\n  " + "\n  ".join(missing))


def verify_citation() -> None:
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    if "version: 1.0.0" not in text:
        raise SystemExit("CITATION.cff is not frozen at version 1.0.0")


def verify_figure_inventory() -> None:
    path = ROOT / "metadata" / "figure_source_inventory.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    incomplete = [r["figure"] for r in rows if r.get("status") != "COMPLETE"]
    if incomplete:
        raise SystemExit("Incomplete figure-source inventory: " + ", ".join(incomplete))


def verify_no_raw_netcdf() -> None:
    nc = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*.nc") if ".git" not in p.parts]
    if nc:
        raise SystemExit(
            "Raw NetCDF files are present in the release checkout but the public "
            "archive policy excludes them:\n  " + "\n  ".join(nc)
        )


def write_manifest() -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{sha256(path)}  ./{path.relative_to(ROOT).as_posix()}"
        for path in repository_files()
    ]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {MANIFEST.relative_to(ROOT)} with {len(lines)} entries")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Regenerate metadata/file_manifest_sha256.csv after verification.",
    )
    args = parser.parse_args()

    verify_required()
    verify_citation()
    verify_figure_inventory()
    verify_no_raw_netcdf()

    print("Release package checks: PASS")
    print(f"Required files checked: {len(REQUIRED)}")

    if args.write_manifest:
        write_manifest()


if __name__ == "__main__":
    main()
