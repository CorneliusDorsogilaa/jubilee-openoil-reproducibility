#!/usr/bin/env python3
"""
Script 73: METOCEAN metocean characterization of the production window.

Purpose
-------
Characterize the exact 48 h forcing window used in the production simulations:
    2020-03-01 00:00 UTC to 2020-03-03 00:00 UTC
at the Jubilee release location:
    4.60000 N, 2.89000 W

requested mean and range of:
    wind speed
    significant wave height, Hs
    peak wave period, Tp
    surface current speed
    mixed-layer depth
    sea-surface temperature

The script also reports:
    fraction of hourly wind samples below 5 m/s
    fraction at or above 5 m/s
because the wave-breaking parameterization has a 5 m/s wind threshold.

Data are sampled at the nearest available forcing grid cell to the release
location from the exact forcing NetCDF files referenced by the frozen project
configuration.

Outputs
-------
diagnostics/validation/metocean_timeseries.csv
diagnostics/validation/metocean_summary.txt
diagnostics/validation/metocean_summary.json
figures/metocean_window.png
figures/metocean_window.pdf
figures/metocean_window.svg
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "diagnostics" / "validation"
FIGDIR = ROOT / "figures"
DIAG.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

SCRIPT25 = ROOT / "scripts" / "25_run_production_scenarios.py"

RELEASE_LAT = 4.60000
RELEASE_LON = -2.89000
START = np.datetime64("2020-03-01T00:00:00")
END = np.datetime64("2020-03-03T00:00:00")


def load_script25():
    spec = importlib.util.spec_from_file_location(
        "prod_for_m6", SCRIPT25
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SCRIPT25}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_coord(ds, candidates):
    for c in candidates:
        if c in ds.coords or c in ds.variables:
            return c
    raise KeyError(f"Could not find any coordinate among {candidates}")


def find_var(ds, candidates):
    for c in candidates:
        if c in ds.variables:
            return c
    return None


def normalize_lon_value(lon_array, lon_value):
    vals = np.asarray(lon_array)
    if np.nanmax(vals) > 180 and lon_value < 0:
        return lon_value % 360
    return lon_value


def nearest_site(ds):
    lat_name = find_coord(ds, ["latitude", "lat", "nav_lat"])
    lon_name = find_coord(ds, ["longitude", "lon", "nav_lon"])

    lat = ds[lat_name]
    lon = ds[lon_name]

    target_lon = normalize_lon_value(lon.values, RELEASE_LON)

    if lat.ndim == 1 and lon.ndim == 1:
        return ds.sel(
            {
                lat_name: RELEASE_LAT,
                lon_name: target_lon,
            },
            method="nearest",
        )

    # Curvilinear fallback.
    latv = np.asarray(lat.values, dtype=float)
    lonv = np.asarray(lon.values, dtype=float)
    lon_target = target_lon
    dist2 = (latv - RELEASE_LAT) ** 2 + (lonv - lon_target) ** 2
    idx = np.unravel_index(np.nanargmin(dist2), dist2.shape)
    dims = lat.dims
    indexers = {d: i for d, i in zip(dims, idx)}
    return ds.isel(indexers)


def time_subset(ds):
    tname = find_coord(ds, ["time", "valid_time"])
    return ds.sel({tname: slice(START, END)}), tname


def surface_if_needed(da):
    depth_candidates = [
        "depth", "deptht", "depthu", "depthv",
        "lev", "level", "z"
    ]
    for d in depth_candidates:
        if d in da.dims:
            return da.isel({d: 0})
    return da


def to_1d_time(da, tname):
    da = surface_if_needed(da)

    # Remove singleton dimensions other than time.
    for d in list(da.dims):
        if d != tname and da.sizes.get(d, 1) == 1:
            da = da.isel({d: 0})

    # If anything remains other than time, take first index as a last-resort
    # nearest-site/surface extraction fallback.
    for d in list(da.dims):
        if d != tname:
            da = da.isel({d: 0})

    vals = np.asarray(da.values, dtype=float).reshape(-1)
    times = np.asarray(da[tname].values)
    return times, vals


def maybe_celsius(values):
    vals = np.asarray(values, dtype=float)
    med = np.nanmedian(vals)
    if med > 100:
        return vals - 273.15
    return vals


def stats(values):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "sd": float(np.std(vals, ddof=1)) if vals.size > 1 else None,
    }


def sample_at_site(path, var_candidates):
    with xr.open_dataset(path) as ds:
        ds, tname = time_subset(ds)
        site = nearest_site(ds)

        out = {}
        for label, candidates in var_candidates.items():
            v = find_var(site, candidates)
            if v is None:
                out[label] = None
            else:
                times, vals = to_1d_time(site[v], tname)
                out[label] = {
                    "variable": v,
                    "times": times,
                    "values": vals,
                    "attrs": dict(site[v].attrs),
                }
        return out


def main():
    prod = load_script25()
    config = prod.load_config(prod.DEFAULT_CONFIG)

    # Use the same mapping logic as production to find forcing filenames.
    helper04 = prod.load_script_module(
        "reader_check_m6",
        ROOT / "scripts" / "04_opendrift_reader_check.py",
    )
    mappings = helper04.STANDARD_NAME_MAPPINGS

    paths = {
        section: ROOT / "data" / "raw" / config["forcing"][section]["filename"]
        for section in mappings
    }

    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing forcing files: " + ", ".join(missing))

    print("Forcing files:")
    for k, p in paths.items():
        print(f"  {k}: {p.relative_to(ROOT)}")

    # Identify sections flexibly from actual available variables.
    sampled_sections = {}

    for section, path in paths.items():
        print(f"Sampling {section}: {path.name}", flush=True)
        sampled_sections[section] = sample_at_site(
            path,
            {
                "u_current": [
                    "uo", "eastward_sea_water_velocity",
                    "x_sea_water_velocity"
                ],
                "v_current": [
                    "vo", "northward_sea_water_velocity",
                    "y_sea_water_velocity"
                ],
                "sst": [
                    "thetao", "sea_water_temperature",
                    "sst", "sea_surface_temperature"
                ],
                "mld": [
                    "mlotst",
                    "ocean_mixed_layer_thickness_defined_by_sigma_theta",
                    "mixed_layer_thickness"
                ],
                "u10": [
                    "u10", "x_wind", "eastward_wind",
                    "eastward_wind_at_10m"
                ],
                "v10": [
                    "v10", "y_wind", "northward_wind",
                    "northward_wind_at_10m"
                ],
                "hs": [
                    "swh", "sea_surface_wave_significant_height",
                    "significant_height_of_combined_wind_waves_and_swell"
                ],
                "tp": [
                    "pp1d",
                    "sea_surface_wave_period_at_variance_spectral_density_maximum",
                    "peak_wave_period"
                ],
            },
        )

    def first_present(label):
        for section, data in sampled_sections.items():
            item = data.get(label)
            if item is not None:
                return section, item
        return None, None

    ocean_u_section, ocean_u = first_present("u_current")
    ocean_v_section, ocean_v = first_present("v_current")
    sst_section, sst = first_present("sst")
    mld_section, mld = first_present("mld")
    wind_u_section, wind_u = first_present("u10")
    wind_v_section, wind_v = first_present("v10")
    hs_section, hs = first_present("hs")
    tp_section, tp = first_present("tp")

    required = {
        "u_current": ocean_u,
        "v_current": ocean_v,
        "sst": sst,
        "mld": mld,
        "u10": wind_u,
        "v10": wind_v,
        "hs": hs,
        "tp": tp,
    }
    absent = [k for k, v in required.items() if v is None]
    if absent:
        raise RuntimeError(
            "Could not locate required forcing variables: " + ", ".join(absent)
        )

    # Build independent source-resolution series.
    ocean_time = pd.to_datetime(ocean_u["times"])
    current_speed = np.hypot(ocean_u["values"], ocean_v["values"])
    sst_vals = maybe_celsius(sst["values"])
    mld_vals = np.asarray(mld["values"], dtype=float)

    wind_time = pd.to_datetime(wind_u["times"])
    wind_speed = np.hypot(wind_u["values"], wind_v["values"])

    wave_time = pd.to_datetime(hs["times"])
    hs_vals = np.asarray(hs["values"], dtype=float)
    tp_vals = np.asarray(tp["values"], dtype=float)

    # Save a long-format timeseries table, preserving native temporal resolution.
    rows = []

    for t, v in zip(ocean_time, current_speed):
        rows.append({"source": "ocean", "time_utc": t, "metric": "current_speed_m_s", "value": v})
    for t, v in zip(pd.to_datetime(sst["times"]), sst_vals):
        rows.append({"source": "ocean", "time_utc": t, "metric": "sst_degC", "value": v})
    for t, v in zip(pd.to_datetime(mld["times"]), mld_vals):
        rows.append({"source": "ocean", "time_utc": t, "metric": "mixed_layer_depth_m", "value": v})
    for t, v in zip(wind_time, wind_speed):
        rows.append({"source": "atmosphere", "time_utc": t, "metric": "wind_speed_m_s", "value": v})
    for t, v in zip(wave_time, hs_vals):
        rows.append({"source": "waves", "time_utc": t, "metric": "Hs_m", "value": v})
    for t, v in zip(pd.to_datetime(tp["times"]), tp_vals):
        rows.append({"source": "waves", "time_utc": t, "metric": "Tp_s", "value": v})

    ts_df = pd.DataFrame(rows)
    ts_csv = DIAG / "metocean_timeseries.csv"
    ts_df.to_csv(ts_csv, index=False)

    summary = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "production_window": {
            "start_utc": str(START),
            "end_utc": str(END),
            "season": "boreal dry season / Gulf of Guinea late dry-season window",
        },
        "release_location": {
            "latitude": RELEASE_LAT,
            "longitude": RELEASE_LON,
        },
        "source_sections": {
            "surface_current": ocean_u_section,
            "sst": sst_section,
            "mixed_layer_depth": mld_section,
            "wind": wind_u_section,
            "waves": hs_section,
        },
        "variables": {
            "surface_current_speed_m_s": stats(current_speed),
            "wind_speed_m_s": stats(wind_speed),
            "significant_wave_height_m": stats(hs_vals),
            "peak_wave_period_s": stats(tp_vals),
            "mixed_layer_depth_m": stats(mld_vals),
            "sea_surface_temperature_degC": stats(sst_vals),
        },
        "wind_threshold_5_m_s": {
            "samples": int(np.isfinite(wind_speed).sum()),
            "fraction_below_5": float(np.mean(wind_speed < 5.0)),
            "fraction_at_or_above_5": float(np.mean(wind_speed >= 5.0)),
            "hours_below_5_if_hourly": float(np.sum(wind_speed < 5.0)),
            "hours_at_or_above_5_if_hourly": float(np.sum(wind_speed >= 5.0)),
        },
        "forcing_files": {
            k: str(p.relative_to(ROOT)) for k, p in paths.items()
        },
        "raw_variable_names": {
            "u_current": ocean_u["variable"],
            "v_current": ocean_v["variable"],
            "sst": sst["variable"],
            "mld": mld["variable"],
            "u10": wind_u["variable"],
            "v10": wind_v["variable"],
            "hs": hs["variable"],
            "tp": tp["variable"],
        },
    }

    json_path = DIAG / "metocean_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    lines = [
        "METOCEAN — PRODUCTION-WINDOW METOCEAN CHARACTERIZATION",
        "=" * 92,
        f"Window: {START} to {END}",
        f"Release location: {RELEASE_LAT:.5f} N, {RELEASE_LON:.5f} E",
        "",
        "NEAREST-SITE FORCING STATISTICS",
        "-" * 92,
    ]

    labels = [
        ("surface_current_speed_m_s", "Surface current speed", "m/s"),
        ("wind_speed_m_s", "Wind speed", "m/s"),
        ("significant_wave_height_m", "Significant wave height, Hs", "m"),
        ("peak_wave_period_s", "Peak wave period, Tp", "s"),
        ("mixed_layer_depth_m", "Mixed-layer depth", "m"),
        ("sea_surface_temperature_degC", "Sea-surface temperature", "degC"),
    ]

    for key, label, unit in labels:
        s = summary["variables"][key]
        lines.append(
            f"{label}: mean={s['mean']:.4g} {unit}, "
            f"range={s['min']:.4g} to {s['max']:.4g} {unit}, "
            f"n={s['n']}"
        )

    w = summary["wind_threshold_5_m_s"]
    lines.extend([
        "",
        "5 m/s WIND THRESHOLD",
        "-" * 92,
        f"Fraction below 5 m/s: {w['fraction_below_5']:.4f}",
        f"Fraction at/above 5 m/s: {w['fraction_at_or_above_5']:.4f}",
        f"Samples below threshold: {int(w['hours_below_5_if_hourly'])}",
        f"Samples at/above threshold: {int(w['hours_at_or_above_5_if_hourly'])}",
        "",
        "SOURCE FILES",
        "-" * 92,
    ])

    for k, p in summary["forcing_files"].items():
        lines.append(f"{k}: {p}")

    txt_path = DIAG / "metocean_summary.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Figure
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 10))
    axes = axes.ravel()

    plot_data = [
        (wind_time, wind_speed, "Wind speed (m/s)", "(a) 10 m wind speed"),
        (wave_time, hs_vals, r"$H_s$ (m)", "(b) Significant wave height"),
        (pd.to_datetime(tp["times"]), tp_vals, r"$T_p$ (s)", "(c) Peak wave period"),
        (ocean_time, current_speed, "Current speed (m/s)", "(d) Surface current speed"),
        (pd.to_datetime(mld["times"]), mld_vals, "Mixed-layer depth (m)", "(e) Mixed-layer depth"),
        (pd.to_datetime(sst["times"]), sst_vals, "SST (°C)", "(f) Sea-surface temperature"),
    ]

    for ax, (t, y, ylabel, title) in zip(axes, plot_data):
        ax.plot(t, y, marker="o", markersize=3, linewidth=1.2)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="x", rotation=25)

    axes[0].axhline(5.0, linestyle="--", linewidth=1.0)

    plt.tight_layout()

    png = FIGDIR / "metocean_window.png"
    pdf = FIGDIR / "metocean_window.pdf"
    svg = FIGDIR / "metocean_window.svg"

    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(svg, bbox_inches="tight")

    print("")
    print("\n".join(lines))
    print("=" * 92)
    print(f"Saved: {ts_csv.relative_to(ROOT)}")
    print(f"Saved: {txt_path.relative_to(ROOT)}")
    print(f"Saved: {json_path.relative_to(ROOT)}")
    print(f"Saved: {png.relative_to(ROOT)}")
    print(f"Saved: {pdf.relative_to(ROOT)}")
    print(f"Saved: {svg.relative_to(ROOT)}")

    plt.show()


if __name__ == "__main__":
    main()
