#!/usr/bin/env python3
"""Export compact, redistribution-safe source tables for the final paper figures.

This script does not rerun OpenDrift. It reads already completed reference NetCDFs
from the Jubilee_OpenOil_G2 project and exports only the variables needed to
redraw (i) main-manuscript Fig. 1 and (ii) Supporting Fig. S3 panels (c)-(d).

Outputs
-------
archive_sources/Fig1_reference_plot_source.csv.gz
archive_sources/FigS3_per_element_surface_time.csv.gz
archive_sources/FigS3_reference_offsets.csv

The exports are compact derivatives of the user's own model outputs. ERA5 and
GLORYS forcing fields are not copied.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "archive_sources"
OUT.mkdir(parents=True, exist_ok=True)

SUMMARY = ROOT / "diagnostics" / "production" / "production_summary_latest.json"
RELEASE_END_H = 6.0
SURFACE_Z_TOL = -1e-6

SCENARIO_ORDER = ["C0_Hydro", "C1_Surface", "C2_Exchange", "C3_Full_fate"]
SCENARIO_SHORT = {
    "C0_Hydro": "C0",
    "C1_Surface": "C1",
    "C2_Exchange": "C2",
    "C3_Full_fate": "C3",
}


def orient_trajectory_time(da: xr.DataArray) -> np.ndarray:
    arr = np.asarray(da.values)
    dims = list(da.dims)
    if arr.ndim != 2:
        raise ValueError(f"{da.name}: expected 2-D trajectory/time data, got {arr.shape}")
    if "trajectory" not in dims or "time" not in dims:
        raise ValueError(f"{da.name}: missing trajectory/time dimensions: {dims}")
    return np.moveaxis(arr, [dims.index("trajectory"), dims.index("time")], [0, 1])


def elapsed_hours(ds: xr.Dataset) -> np.ndarray:
    if "time" not in ds:
        raise KeyError("NetCDF has no time coordinate")
    t = pd.to_datetime(np.asarray(ds["time"].values))
    return ((t - t[0]) / np.timedelta64(1, "h")).astype(float)


def resolve_from_summary() -> dict[str, Path]:
    if not SUMMARY.exists():
        raise FileNotFoundError(f"Missing production summary: {SUMMARY}")
    s = json.loads(SUMMARY.read_text())
    out: dict[str, Path] = {}
    for key in SCENARIO_ORDER:
        rec = s["runs"][key]
        p = ROOT / rec["output_netcdf"]
        if not p.exists():
            raise FileNotFoundError(p)
        out[SCENARIO_SHORT[key]] = p
    return out


def resolve_oil_file(oil: str, scenario: str) -> Path:
    if oil == "Qua Iboe":
        base = ROOT / "outputs" / "oil_sensitivity_lighter"
        exact = base / (
            "C2_Exchange_QuaIboe_AD01483.nc" if scenario == "C2"
            else "C3_Full_fate_QuaIboe_AD01483.nc"
        )
        if exact.exists():
            return exact
    elif oil == "Bonny Light":
        prod = resolve_from_summary()
        return prod[scenario]
    raise FileNotFoundError(f"Could not resolve {oil} {scenario} NetCDF")


def export_fig1() -> Path:
    files = resolve_from_summary()
    parts = []
    for scenario, path in files.items():
        print(f"Fig1 export: {scenario} <- {path.relative_to(ROOT)}")
        with xr.open_dataset(path) as ds:
            lon = orient_trajectory_time(ds["lon"]).astype(float)
            lat = orient_trajectory_time(ds["lat"]).astype(float)
            h = elapsed_hours(ds)
            ntraj, nt = lon.shape
            tr = np.repeat(np.arange(ntraj, dtype=int), nt)
            ti = np.tile(np.arange(nt, dtype=int), ntraj)
            hh = np.tile(h, ntraj)
            parts.append(pd.DataFrame({
                "configuration": scenario,
                "trajectory": tr,
                "time_index": ti,
                "elapsed_h": hh,
                "lon_deg_east": lon.reshape(-1),
                "lat_deg_north": lat.reshape(-1),
            }))
    df = pd.concat(parts, ignore_index=True)
    out = OUT / "Fig1_reference_plot_source.csv.gz"
    df.to_csv(out, index=False, compression="gzip")
    return out


def per_element_surface_time(path: Path) -> pd.DataFrame:
    with xr.open_dataset(path) as ds:
        z = orient_trajectory_time(ds["z"]).astype(float)
        lon = orient_trajectory_time(ds["lon"]).astype(float)
        lat = orient_trajectory_time(ds["lat"]).astype(float)
        h = elapsed_hours(ds)

        mask_t = h >= RELEASE_END_H - 1e-12
        zpost = z[:, mask_t]
        finite = np.isfinite(zpost)
        surf = finite & (zpost >= SURFACE_Z_TOL)
        denom = finite.sum(axis=1)
        frac = np.divide(
            surf.sum(axis=1), denom,
            out=np.full(z.shape[0], np.nan, dtype=float),
            where=denom > 0,
        )
        return pd.DataFrame({
            "trajectory": np.arange(z.shape[0], dtype=int),
            "post_release_surface_time_fraction": frac,
            "final_lon_deg_east": lon[:, -1],
            "final_lat_deg_north": lat[:, -1],
        })


def centroid(df: pd.DataFrame) -> tuple[float, float]:
    good = np.isfinite(df["final_lon_deg_east"]) & np.isfinite(df["final_lat_deg_north"])
    if not good.any():
        return np.nan, np.nan
    return (
        float(df.loc[good, "final_lon_deg_east"].mean()),
        float(df.loc[good, "final_lat_deg_north"].mean()),
    )


def export_figs3() -> tuple[Path, Path]:
    rows = []
    offsets = []
    for oil in ("Qua Iboe", "Bonny Light"):
        for scenario in ("C2", "C3"):
            path = resolve_oil_file(oil, scenario)
            print(f"FigS3 export: {oil} {scenario} <- {path.relative_to(ROOT)}")
            d = per_element_surface_time(path)
            d.insert(0, "scenario", scenario)
            d.insert(0, "oil", oil)
            rows.append(d)

            for label, selector in (
                ("mostly_submerged_lt_0p50", d["post_release_surface_time_fraction"] < 0.50),
                ("surface_locked_ge_0p95", d["post_release_surface_time_fraction"] >= 0.95),
                ("all_elements", np.ones(len(d), dtype=bool)),
            ):
                sub = d.loc[selector]
                clon, clat = centroid(sub)
                offsets.append({
                    "oil": oil,
                    "scenario": scenario,
                    "group": label,
                    "n_elements": int(len(sub)),
                    "centroid_lon_deg_east": clon,
                    "centroid_lat_deg_north": clat,
                })

    all_df = pd.concat(rows, ignore_index=True)
    out1 = OUT / "FigS3_per_element_surface_time.csv.gz"
    all_df.to_csv(out1, index=False, compression="gzip")

    off = pd.DataFrame(offsets)
    out2 = OUT / "FigS3_reference_offsets.csv"
    off.to_csv(out2, index=False)
    return out1, out2


def main() -> None:
    p1 = export_fig1()
    p2, p3 = export_figs3()
    print("\nCreated compact archive sources:")
    for p in (p1, p2, p3):
        print(f"  {p.relative_to(ROOT)}  ({p.stat().st_size/1024:.1f} KiB)")


if __name__ == "__main__":
    main()
