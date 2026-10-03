#!/usr/bin/env python3
"""
Script 82: Final three-oil OIL SENSITIVITY sensitivity analysis.

Purpose
-------
Combine the completed lighter, baseline, and heavier oil cases:

    AD01483 | QUA IBOE, SHELL OIL      (lighter)
    AD01440 | BONNY LIGHT, SHELL OIL    (baseline)
    AD01442 | CABINDA BLEND, SHELL OIL  (heavier)

for both:
    C2_Exchange
    C3_Full_fate

The script:
1. Uses scripts/25_run_production_scenarios.py -> extract_metrics() so the
   trajectory diagnostics exactly match the production analysis.
2. Extracts oil properties from the installed OpenOil/ADIOS library.
3. Summarizes C2/C3 transport and surface-exposure diagnostics.
4. Extracts final oil-state diagnostics where present in the NetCDF files.
5. Produces a compact property table and a two-sided sensitivity table.
6. Generates an OIL SENSITIVITY diagnostic figure. This is NOT a manuscript Figure 1.

Outputs
-------
diagnostics/validation/three_oil_properties.csv
diagnostics/validation/three_oil_metrics.csv
diagnostics/validation/three_oil_summary.txt
diagnostics/validation/three_oil_summary.json

figures/three_oil_sensitivity.png
figures/three_oil_sensitivity.pdf
figures/three_oil_sensitivity.svg
"""

from __future__ import annotations

import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from opendrift.models.openoil import OpenOil

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "diagnostics" / "validation"
FIGDIR = ROOT / "figures"
DIAG.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

SCRIPT25 = ROOT / "scripts" / "25_run_production_scenarios.py"

OILS = {
    "Qua Iboe": {
        "id": "AD01483",
        "role": "lighter",
        "C2": ROOT / "outputs" / "oil_sensitivity_lighter" / "C2_Exchange_QuaIboe_AD01483.nc",
        "C3": ROOT / "outputs" / "oil_sensitivity_lighter" / "C3_Full_fate_QuaIboe_AD01483.nc",
    },
    "Bonny Light": {
        "id": "AD01440",
        "role": "baseline",
        "C2": ROOT / "outputs" / "production" / "C2_Exchange_N1000_dt1800s_Kh100_seed20220901_20260917T023449Z.nc",
        "C3": ROOT / "outputs" / "production" / "C3_Full_fate_N1000_dt1800s_Kh100_seed20220901_20260917T023449Z.nc",
    },
    "Cabinda Blend": {
        "id": "AD01442",
        "role": "heavier",
        "C2": ROOT / "outputs" / "oil_sensitivity" / "C2_Exchange_Cabinda_AD01442.nc",
        "C3": ROOT / "outputs" / "oil_sensitivity" / "C3_Full_fate_Cabinda_AD01442.nc",
    },
}


def load_script25():
    spec = importlib.util.spec_from_file_location("prod_m10", SCRIPT25)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SCRIPT25}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def existing_or_glob(path: Path, scenario: str, oil_id: str):
    if path.exists():
        return path

    # Fallbacks for production timestamps / naming differences.
    if oil_id == "AD01440":
        patt = f"{scenario}_N1000_dt1800s_Kh100_seed20220901_*.nc"
        found = sorted(
            (ROOT / "outputs" / "production").glob(patt),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if found:
            return found[0]

    raise FileNotFoundError(f"Could not locate required NetCDF: {path}")


def oil_properties(oil_id: str):
    model = OpenOil(loglevel=50)
    model.set_oiltype_by_id(oil_id)
    ot = model.oiltype
    g = getattr(ot, "gnome_oil", {}) or {}

    def f(x):
        try:
            return float(x)
        except Exception:
            return np.nan

    try:
        rho15 = float(ot.density_at_temp(15.0, unit="C"))
    except Exception:
        rho15 = np.nan

    try:
        nu378 = float(ot.kvis_at_temp(37.8, unit="C")) * 1e6
    except Exception:
        nu378 = np.nan

    try:
        sigma = float(ot.oil_water_surface_tension())
    except Exception:
        sigma = np.nan

    try:
        ymax = float(ot.emulsion_water_fraction_max)
    except Exception:
        ymax = np.nan

    return {
        "oil_id": oil_id,
        "oil_name": getattr(ot, "name", None),
        "api": f(g.get("api")),
        "density_15C_kg_m3": rho15,
        "kvis_37p8C_cSt": nu378,
        "pour_point_K": f(g.get("pour_point")),
        "oil_water_surface_tension_N_m": sigma,
        "emulsion_water_fraction_max": ymax,
    }


def orient_final(da: xr.DataArray):
    arr = np.asarray(da.values, dtype=float)
    dims = list(da.dims)

    if "trajectory" in dims and "time" in dims:
        arr = np.moveaxis(
            arr,
            [dims.index("trajectory"), dims.index("time")],
            [0, 1],
        )
        return arr[:, -1]

    if "time" in dims:
        arr = np.moveaxis(arr, dims.index("time"), -1)
        return np.asarray(arr[..., -1]).reshape(-1)

    return arr.reshape(-1)


def mean_final(ds: xr.Dataset, candidates):
    for name in candidates:
        if name in ds:
            vals = orient_final(ds[name])
            vals = vals[np.isfinite(vals)]
            if vals.size:
                return float(np.mean(vals))
    return np.nan


def sum_final(ds: xr.Dataset, candidates):
    for name in candidates:
        if name in ds:
            vals = orient_final(ds[name])
            vals = vals[np.isfinite(vals)]
            if vals.size:
                return float(np.sum(vals))
    return np.nan


def state_metrics(path: Path):
    with xr.open_dataset(path) as ds:
        return {
            "final_mean_water_fraction": mean_final(
                ds, ["water_fraction", "emulsion_water_fraction"]
            ),
            "final_mean_viscosity_m2_s": mean_final(
                ds, ["viscosity"]
            ),
            "final_mass_oil": sum_final(
                ds, ["mass_oil"]
            ),
            "final_mass_evaporated": sum_final(
                ds, ["mass_evaporated"]
            ),
            "final_mass_dispersed": sum_final(
                ds, ["mass_dispersed"]
            ),
            "final_mass_biodegraded": sum_final(
                ds, ["mass_biodegraded"]
            ),
            "opendrift_version_attr": str(ds.attrs.get("opendrift_version", "")),
        }


def metric_value(m, key):
    if key == "centroid_displacement_km":
        return float(m["centroid"]["displacement_km"])
    if key == "A90_km2":
        return float(m["central_90_footprint_area_km2"])
    if key == "surface_exposure_fraction":
        return float(m["surface_exposure_fraction_6h_to_48h"])
    if key == "shoreline_contact_probability":
        return float(m["shoreline_contact_probability"])
    raise KeyError(key)


def main():
    prod = load_script25()

    # Resolve files.
    resolved = {}
    for oil_name, info in OILS.items():
        resolved[oil_name] = {
            "C2": existing_or_glob(info["C2"], "C2_Exchange", info["id"]),
            "C3": existing_or_glob(info["C3"], "C3_Full_fate", info["id"]),
        }

    # Oil-property table.
    props = []
    for oil_name, info in OILS.items():
        p = oil_properties(info["id"])
        props.append({
            "role": info["role"],
            "display_name": oil_name,
            **p,
        })

    prop_df = pd.DataFrame(props)
    prop_csv = DIAG / "three_oil_properties.csv"
    prop_df.to_csv(prop_csv, index=False)

    # Scenario metrics.
    records = []
    for oil_name, info in OILS.items():
        for scenario in ("C2", "C3"):
            path = resolved[oil_name][scenario]
            print(f"Analyzing {oil_name} | {scenario} | {path.name}", flush=True)

            m = prod.extract_metrics(path)
            s = state_metrics(path)

            records.append({
                "role": info["role"],
                "oil": oil_name,
                "oil_id": info["id"],
                "scenario": scenario,
                "file": str(path.relative_to(ROOT)),
                "centroid_displacement_km": metric_value(m, "centroid_displacement_km"),
                "A90_km2": metric_value(m, "A90_km2"),
                "surface_exposure_fraction": metric_value(m, "surface_exposure_fraction"),
                "shoreline_contact_probability": metric_value(m, "shoreline_contact_probability"),
                **s,
            })

    df = pd.DataFrame(records)

    # C3 - C2 within-oil changes.
    paired = []
    for oil_name in OILS:
        c2 = df[(df["oil"] == oil_name) & (df["scenario"] == "C2")].iloc[0]
        c3 = df[(df["oil"] == oil_name) & (df["scenario"] == "C3")].iloc[0]

        paired.append({
            "oil": oil_name,
            "oil_id": c2["oil_id"],
            "role": c2["role"],
            "delta_centroid_displacement_km_C3_minus_C2":
                c3["centroid_displacement_km"] - c2["centroid_displacement_km"],
            "delta_A90_km2_C3_minus_C2":
                c3["A90_km2"] - c2["A90_km2"],
            "relative_delta_A90_percent_C3_vs_C2":
                100.0 * (c3["A90_km2"] - c2["A90_km2"]) / c2["A90_km2"],
            "delta_surface_exposure_percentage_points_C3_minus_C2":
                100.0 * (
                    c3["surface_exposure_fraction"] - c2["surface_exposure_fraction"]
                ),
            "delta_shoreline_probability_C3_minus_C2":
                c3["shoreline_contact_probability"] - c2["shoreline_contact_probability"],
            "C2_surface_exposure_fraction": c2["surface_exposure_fraction"],
            "C3_surface_exposure_fraction": c3["surface_exposure_fraction"],
            "C3_final_mean_water_fraction": c3["final_mean_water_fraction"],
            "C3_final_mean_viscosity_m2_s": c3["final_mean_viscosity_m2_s"],
        })

    paired_df = pd.DataFrame(paired)

    # Merge key properties into paired summary.
    paired_df = paired_df.merge(
        prop_df[
            [
                "display_name",
                "api",
                "density_15C_kg_m3",
                "kvis_37p8C_cSt",
                "oil_water_surface_tension_N_m",
                "emulsion_water_fraction_max",
            ]
        ],
        left_on="oil",
        right_on="display_name",
        how="left",
    ).drop(columns=["display_name"])

    metrics_csv = DIAG / "three_oil_metrics.csv"
    paired_df.to_csv(metrics_csv, index=False)

    bundle = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "version_provenance_note": (
            "Installed distribution metadata reports OpenDrift 1.14.12, while "
            "the installed source internal __version__ string is 1.14.11 and "
            "that internal value is written to NetCDF opendrift_version metadata."
        ),
        "oil_properties": prop_df.replace({np.nan: None}).to_dict(orient="records"),
        "scenario_records": df.replace({np.nan: None}).to_dict(orient="records"),
        "paired_C3_minus_C2": paired_df.replace({np.nan: None}).to_dict(orient="records"),
    }

    json_path = DIAG / "three_oil_summary.json"
    json_path.write_text(
        json.dumps(bundle, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    lines = [
        "OIL SENSITIVITY — FINAL THREE-OIL SENSITIVITY",
        "=" * 104,
        f"Generated UTC: {bundle['generated_utc']}",
        "",
        "OIL PROPERTY BRACKET",
        "-" * 104,
    ]

    for _, r in prop_df.iterrows():
        lines.append(
            f"{r['role']:>8} | {r['display_name']} | {r['oil_id']} | "
            f"API={r['api']:.3f} | rho15={r['density_15C_kg_m3']:.3f} kg/m3 | "
            f"nu37.8={r['kvis_37p8C_cSt']:.4f} cSt | "
            f"sigma={r['oil_water_surface_tension_N_m']:.6f} N/m | "
            f"Ymax={r['emulsion_water_fraction_max']}"
        )

    lines.extend([
        "",
        "C3 - C2 SENSITIVITY",
        "-" * 104,
    ])

    for _, r in paired_df.iterrows():
        lines.extend([
            f"{r['role']:>8} | {r['oil']} ({r['oil_id']})",
            f"  C2 surface exposure = {r['C2_surface_exposure_fraction']:.6f}",
            f"  C3 surface exposure = {r['C3_surface_exposure_fraction']:.6f}",
            f"  Delta surface exposure = "
            f"{r['delta_surface_exposure_percentage_points_C3_minus_C2']:.3f} percentage points",
            f"  Delta A90 = {r['delta_A90_km2_C3_minus_C2']:.3f} km2 "
            f"({r['relative_delta_A90_percent_C3_vs_C2']:.3f}%)",
            f"  Delta centroid displacement = "
            f"{r['delta_centroid_displacement_km_C3_minus_C2']:.3f} km",
            f"  Delta shoreline probability = "
            f"{r['delta_shoreline_probability_C3_minus_C2']:.6f}",
            f"  C3 final mean water fraction = "
            f"{r['C3_final_mean_water_fraction']:.6f}",
            f"  C3 final mean viscosity = "
            f"{r['C3_final_mean_viscosity_m2_s']:.6g} m2/s",
            "",
        ])

    lines.extend([
        "INTERPRETATION BOUNDARY",
        "-" * 104,
        "Qua Iboe and Cabinda Blend are lower- and higher-viscosity sensitivity",
        "contrasts around the Bonny Light baseline. They are not alternate claims",
        "of the exact Jubilee assay. The three-oil comparison tests directional",
        "sensitivity to the selected ADIOS oil-property parameterization.",
    ])

    txt_path = DIAG / "three_oil_summary.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Diagnostic figure.
    order = ["Qua Iboe", "Bonny Light", "Cabinda Blend"]
    x = np.arange(len(order))

    c2_vals = [
        float(df[(df["oil"] == o) & (df["scenario"] == "C2")]["surface_exposure_fraction"].iloc[0])
        for o in order
    ]
    c3_vals = [
        float(df[(df["oil"] == o) & (df["scenario"] == "C3")]["surface_exposure_fraction"].iloc[0])
        for o in order
    ]
    c3_water = [
        float(df[(df["oil"] == o) & (df["scenario"] == "C3")]["final_mean_water_fraction"].iloc[0])
        for o in order
    ]
    viscosity = [
        float(prop_df[prop_df["display_name"] == o]["kvis_37p8C_cSt"].iloc[0])
        for o in order
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))

    width = 0.34
    axes[0].bar(x - width / 2, c2_vals, width, label="C2")
    axes[0].bar(x + width / 2, c3_vals, width, label="C3")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(order, rotation=15)
    axes[0].set_ylabel("Surface exposure fraction")
    axes[0].set_title("(a) Surface exposure")
    axes[0].legend(frameon=False)
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].plot(x, viscosity, marker="o")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(order, rotation=15)
    axes[1].set_ylabel(r"Kinematic viscosity at 37.8 °C (cSt)")
    axes[1].set_title("(b) Baseline oil viscosity")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(x, c3_water, marker="o")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(order, rotation=15)
    axes[2].set_ylabel("C3 final mean water fraction")
    axes[2].set_title("(c) Emulsification state")
    axes[2].grid(True, alpha=0.25)

    plt.tight_layout()

    png = FIGDIR / "three_oil_sensitivity.png"
    pdf = FIGDIR / "three_oil_sensitivity.pdf"
    svg = FIGDIR / "three_oil_sensitivity.svg"

    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(svg, bbox_inches="tight")

    print("\n".join(lines))
    print("=" * 104)
    print(f"Saved: {prop_csv.relative_to(ROOT)}")
    print(f"Saved: {metrics_csv.relative_to(ROOT)}")
    print(f"Saved: {txt_path.relative_to(ROOT)}")
    print(f"Saved: {json_path.relative_to(ROOT)}")
    print(f"Saved: {png.relative_to(ROOT)}")
    print(f"Saved: {pdf.relative_to(ROOT)}")
    print(f"Saved: {svg.relative_to(ROOT)}")

    plt.show()


if __name__ == "__main__":
    main()
