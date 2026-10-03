#!/usr/bin/env python3
"""
Script 75: Temporal-resolution validation check with corrected interpolation.

Purpose
-------
Quantify the forcing-resolution mismatch behind the C0 -> C1 attribution.

The production forcing uses:
    GLORYS12 surface currents: daily
    ERA5 winds: hourly
    ERA5 waves: hourly

This script documents the native sampling and the hourly interpolation available
to OpenDrift. It does not claim that interpolation recovers omitted tidal,
inertial, or other sub-daily current variability.

Outputs
-------
diagnostics/validation/temporal_resolution_check.csv
diagnostics/validation/temporal_resolution_check.txt
diagnostics/validation/temporal_resolution_check.json
figures/temporal_resolution_diagnostic.png
figures/temporal_resolution_diagnostic.pdf
figures/temporal_resolution_diagnostic.svg
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

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "diagnostics" / "validation"
FIGDIR = ROOT / "figures"
DIAG.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

SCRIPT25 = ROOT / "scripts" / "25_run_production_scenarios.py"
RELEASE_LAT = 4.60000
RELEASE_LON = -2.89000

START = pd.Timestamp("2020-03-01T00:00:00")
END = pd.Timestamp("2020-03-03T00:00:00")


def load_script25():
    spec = importlib.util.spec_from_file_location("prod_m7", SCRIPT25)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SCRIPT25}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_coord(ds, candidates):
    for c in candidates:
        if c in ds.coords or c in ds.variables:
            return c
    raise KeyError(candidates)


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
            {lat_name: RELEASE_LAT, lon_name: target_lon},
            method="nearest",
        )

    latv = np.asarray(lat.values, dtype=float)
    lonv = np.asarray(lon.values, dtype=float)
    dist2 = (latv - RELEASE_LAT) ** 2 + (lonv - target_lon) ** 2
    idx = np.unravel_index(np.nanargmin(dist2), dist2.shape)
    return ds.isel({d: i for d, i in zip(lat.dims, idx)})


def subset_time(ds):
    tname = find_coord(ds, ["time", "valid_time"])
    return ds.sel({tname: slice(START.to_datetime64(), END.to_datetime64())}), tname


def surface_series(site, varname, tname):
    da = site[varname]
    for d in ["depth", "deptht", "depthu", "depthv", "lev", "level", "z"]:
        if d in da.dims:
            da = da.isel({d: 0})
    for d in list(da.dims):
        if d != tname:
            da = da.isel({d: 0})
    return (
        pd.to_datetime(np.asarray(da[tname].values)),
        np.asarray(da.values, dtype=float).reshape(-1),
    )


def spacing_hours(times):
    t = pd.DatetimeIndex(times)
    if len(t) < 2:
        return []
    return [float(x / np.timedelta64(1, "h")) for x in np.diff(t.values)]


def vector_direction_deg(u, v):
    return np.degrees(np.arctan2(v, u))


def circular_delta_deg(a, b):
    return ((b - a + 180.0) % 360.0) - 180.0


def get_production_c0_c1_separation():
    latest = ROOT / "diagnostics" / "production" / "production_summary_latest.json"
    if not latest.exists():
        return None
    try:
        d = json.loads(latest.read_text())
        return float(
            d["paired_attribution"]["C0_to_C1"]["centroid_endpoint_separation_km"]
        )
    except Exception:
        return None


def main():
    prod = load_script25()
    config = prod.load_config(prod.DEFAULT_CONFIG)
    helper04 = prod.load_script_module(
        "reader_check_m7",
        ROOT / "scripts" / "04_opendrift_reader_check.py",
    )
    mappings = helper04.STANDARD_NAME_MAPPINGS

    paths = {
        section: ROOT / "data" / "raw" / config["forcing"][section]["filename"]
        for section in mappings
    }

    current = None
    wind = None
    wave = None

    for section, path in paths.items():
        with xr.open_dataset(path) as ds:
            ds, tname = subset_time(ds)
            site = nearest_site(ds)

            ucur = find_var(site, ["uo", "eastward_sea_water_velocity", "x_sea_water_velocity"])
            vcur = find_var(site, ["vo", "northward_sea_water_velocity", "y_sea_water_velocity"])
            u10 = find_var(site, ["u10", "x_wind", "eastward_wind", "eastward_wind_at_10m"])
            v10 = find_var(site, ["v10", "y_wind", "northward_wind", "northward_wind_at_10m"])
            hs = find_var(site, [
                "swh",
                "sea_surface_wave_significant_height",
                "significant_height_of_combined_wind_waves_and_swell",
            ])

            if ucur and vcur and current is None:
                tc, u = surface_series(site, ucur, tname)
                _, v = surface_series(site, vcur, tname)
                current = {
                    "section": section, "file": path, "time": tc, "u": u, "v": v,
                    "u_var": ucur, "v_var": vcur,
                }

            if u10 and v10 and wind is None:
                tw, u = surface_series(site, u10, tname)
                _, v = surface_series(site, v10, tname)
                wind = {
                    "section": section, "file": path, "time": tw, "u": u, "v": v,
                    "u_var": u10, "v_var": v10,
                }

            if hs and wave is None:
                th, h = surface_series(site, hs, tname)
                wave = {
                    "section": section, "file": path, "time": th, "hs": h,
                    "hs_var": hs,
                }

    if current is None or wind is None or wave is None:
        raise RuntimeError("Could not locate current, wind, and wave forcing.")

    cur_speed = np.hypot(current["u"], current["v"])
    cur_dir = vector_direction_deg(current["u"], current["v"])
    wind_speed = np.hypot(wind["u"], wind["v"])

    cur_spacing = spacing_hours(current["time"])
    wind_spacing = spacing_hours(wind["time"])
    wave_spacing = spacing_hours(wave["time"])

    hourly = pd.date_range(START, END, freq="1h")

    # Robust interpolation coordinate: elapsed hours from the production start.
    # This avoids datetime integer-resolution mismatches (ns/us) across pandas
    # and NumPy versions.
    current_h = np.asarray(
        (pd.DatetimeIndex(current["time"]) - START) / pd.Timedelta(hours=1),
        dtype=float,
    )
    hourly_h = np.asarray(
        (hourly - START) / pd.Timedelta(hours=1),
        dtype=float,
    )

    u_hourly = np.interp(hourly_h, current_h, current["u"])
    v_hourly = np.interp(hourly_h, current_h, current["v"])
    speed_hourly = np.hypot(u_hourly, v_hourly)

    # Self-check: interpolation must reproduce every native current vector
    # exactly at its own timestamp.
    u_native_check = np.interp(current_h, current_h, current["u"])
    v_native_check = np.interp(current_h, current_h, current["v"])
    if not (
        np.allclose(u_native_check, current["u"], rtol=0, atol=1e-12)
        and np.allclose(v_native_check, current["v"], rtol=0, atol=1e-12)
    ):
        raise RuntimeError("Current interpolation self-check failed.")

    # NumPy 2.x uses np.trapezoid (np.trapz was removed).
    dt_s = 3600.0
    east_m = float(np.trapezoid(u_hourly, dx=dt_s))
    north_m = float(np.trapezoid(v_hourly, dx=dt_s))
    net_current_displacement_km = math.hypot(east_m, north_m) / 1000.0
    current_pathlength_km = float(np.trapezoid(speed_hourly, dx=dt_s) / 1000.0)

    native_changes = []
    for i in range(len(cur_speed) - 1):
        native_changes.append({
            "from_utc": str(current["time"][i]),
            "to_utc": str(current["time"][i + 1]),
            "speed_change_m_s": float(cur_speed[i + 1] - cur_speed[i]),
            "direction_change_deg": float(
                circular_delta_deg(cur_dir[i], cur_dir[i + 1])
            ),
        })

    c0_c1_sep = get_production_c0_c1_separation()

    rows = []
    for t, u, v, s in zip(hourly, u_hourly, v_hourly, speed_hourly):
        rows.append({
            "time_utc": t,
            "u_current_interpolated_m_s": u,
            "v_current_interpolated_m_s": v,
            "current_speed_interpolated_m_s": s,
        })

    pd.DataFrame(rows).to_csv(
        DIAG / "temporal_resolution_check.csv", index=False
    )

    summary = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window": {"start": str(START), "end": str(END)},
        "native_sampling": {
            "current_samples": int(len(current["time"])),
            "current_spacing_hours": cur_spacing,
            "wind_samples": int(len(wind["time"])),
            "wind_spacing_hours": wind_spacing,
            "wave_samples": int(len(wave["time"])),
            "wave_spacing_hours": wave_spacing,
        },
        "native_surface_current": {
            "speed_m_s": [float(x) for x in cur_speed],
            "direction_deg_math": [float(x) for x in cur_dir],
            "changes": native_changes,
        },
        "interpolation_validation": {
            "method": "linear interpolation in elapsed hours from production start",
            "native_vector_self_check_passed": True,
        },
        "hourly_interpolated_current": {
            "samples": int(len(hourly)),
            "mean_speed_m_s": float(np.mean(speed_hourly)),
            "min_speed_m_s": float(np.min(speed_hourly)),
            "max_speed_m_s": float(np.max(speed_hourly)),
            "48h_net_advective_displacement_km": net_current_displacement_km,
            "48h_advective_pathlength_km": current_pathlength_km,
        },
        "wind": {
            "samples": int(len(wind["time"])),
            "mean_speed_m_s": float(np.mean(wind_speed)),
            "min_speed_m_s": float(np.min(wind_speed)),
            "max_speed_m_s": float(np.max(wind_speed)),
        },
        "production_C0_to_C1_centroid_endpoint_separation_km": c0_c1_sep,
        "interpretive_limit": (
            "The production current file contains only daily native snapshots. "
            "Hourly interpolation cannot recover sub-daily variance, tides, inertial "
            "motions, or other high-frequency current variability; therefore the "
            "existing forcing cannot demonstrate that daily currents capture most "
            "of the true advective variance at the site."
        ),
        "forcing_files": {
            "current": str(current["file"].relative_to(ROOT)),
            "wind": str(wind["file"].relative_to(ROOT)),
            "waves": str(wave["file"].relative_to(ROOT)),
        },
    }

    json_path = DIAG / "temporal_resolution_check.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "TEMPORAL RESOLUTION — TEMPORAL-RESOLUTION CHECK",
        "=" * 94,
        f"Window: {START} to {END}",
        "",
        "NATIVE FORCING TEMPORAL RESOLUTION",
        "-" * 94,
        f"Currents: {len(current['time'])} samples | spacings={cur_spacing} h",
        f"Winds:    {len(wind['time'])} samples | first spacings={wind_spacing[:5]} h",
        f"Waves:    {len(wave['time'])} samples | first spacings={wave_spacing[:5]} h",
        "",
        "NATIVE SURFACE CURRENT SNAPSHOTS",
        "-" * 94,
    ]

    for t, u, v, s, d in zip(
        current["time"], current["u"], current["v"], cur_speed, cur_dir
    ):
        lines.append(
            f"{t} | u={u:.5f} m/s | v={v:.5f} m/s | "
            f"speed={s:.5f} m/s | direction={d:.2f} deg"
        )

    lines.extend(["", "DAILY-SNAPSHOT CHANGES", "-" * 94])

    for c in native_changes:
        lines.append(
            f"{c['from_utc']} -> {c['to_utc']} | "
            f"speed change={c['speed_change_m_s']:.5f} m/s | "
            f"direction change={c['direction_change_deg']:.2f} deg"
        )

    lines.extend([
        "",
        "HOURLY INTERPOLATION OF THE DAILY CURRENT FIELD",
        "-" * 94,
        f"Mean interpolated speed: {np.mean(speed_hourly):.5f} m/s",
        f"Range: {np.min(speed_hourly):.5f} to {np.max(speed_hourly):.5f} m/s",
        f"48 h net advective displacement from interpolated current: "
        f"{net_current_displacement_km:.3f} km",
        f"48 h advective path length from interpolated current: "
        f"{current_pathlength_km:.3f} km",
    ])

    if c0_c1_sep is not None:
        lines.append(
            f"Production C0 -> C1 centroid endpoint separation: "
            f"{c0_c1_sep:.3f} km"
        )

    lines.extend([
        "",
        "WHAT THIS CHECK CAN AND CANNOT ESTABLISH",
        "-" * 94,
        "The mismatch is real: the current forcing has 3 native snapshots over 48 h,",
        "whereas wind and wave forcing have 49 hourly samples.",
        "Linear interpolation provides an hourly value to the model but does not add",
        "new current variability between daily snapshots.",
        "Therefore these production data cannot quantify omitted tidal, inertial, or",
        "other sub-daily current variance and cannot demonstrate that the daily field",
        "captures most of the true advective variance at Jubilee.",
        "This is a forcing-resolution limitation on the C0 -> C1 attribution.",
    ])

    txt_path = DIAG / "temporal_resolution_check.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.2))

    axes[0].plot(
        hourly, speed_hourly, linewidth=1.5,
        label="Hourly interpolation of daily GLORYS current"
    )
    axes[0].plot(
        current["time"], cur_speed, "o",
        label="Native GLORYS snapshots"
    )
    axes[0].set_ylabel("Surface current speed (m/s)")
    axes[0].set_title("(a) Native vs interpolated surface-current forcing")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(wind["time"], wind_speed, linewidth=1.4, label="ERA5 wind speed")
    axes[1].plot(wave["time"], wave["hs"], linewidth=1.4, label="ERA5 $H_s$")
    axes[1].set_ylabel("Wind speed (m/s) / $H_s$ (m)")
    axes[1].set_title("(b) Hourly wind and wave forcing")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)

    plt.tight_layout()

    png = FIGDIR / "temporal_resolution_diagnostic_v2.png"
    pdf = FIGDIR / "temporal_resolution_diagnostic_v2.pdf"
    svg = FIGDIR / "temporal_resolution_diagnostic_v2.svg"

    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(svg, bbox_inches="tight")

    print("\n".join(lines))
    print("=" * 94)
    print(f"Saved: {txt_path.relative_to(ROOT)}")
    print(f"Saved: {json_path.relative_to(ROOT)}")
    print(f"Saved: {png.relative_to(ROOT)}")
    print(f"Saved: {pdf.relative_to(ROOT)}")
    print(f"Saved: {svg.relative_to(ROOT)}")

    plt.show()


if __name__ == "__main__":
    main()
