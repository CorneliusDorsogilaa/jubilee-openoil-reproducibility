#!/usr/bin/env python3
"""Production OpenOil scenario runner after corrected Gate G4.

Authoritative production numerics
---------------------------------
- horizontal diffusivity Kh = 100 m2/s
- calculation timestep = 30 min (1800 s)
- output timestep = 60 min (3600 s)
- particles = 1000
- master seed = 20220901
- release radius = 500 m
- continuous release = 6 h
- trajectory horizon = 48 h
- oil analogue = Bonny Light, AD01440

Production scenario matrix
--------------------------
C0_Hydro
C1_Surface
C2_Exchange
C3_Full_fate

Each scenario is executed in a fresh Python process to avoid accumulated
native reader state.  All scenarios use the same seed and identical forcing,
release, oil, and numerical settings so that the scenario ladder remains a
controlled attribution experiment.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jubilee_openoil.config import DEFAULT_CONFIG, config_digest, load_config, validate_config

SCENARIOS = ["C0_Hydro", "C1_Surface", "C2_Exchange", "C3_Full_fate"]

MASTER_SEED = 20220901
RELEASE_LON = -2.89000
RELEASE_LAT = 4.60000
RELEASE_RADIUS_M = 500.0
N_ELEMENTS = 1000

START_UTC = datetime(2020, 3, 1, 0, 0)
RELEASE_END_UTC = START_UTC + timedelta(hours=6)
RUN_END_UTC = START_UTC + timedelta(hours=48)

CALCULATION_STEP_SECONDS = 1800
OUTPUT_STEP_SECONDS = 3600
HORIZONTAL_DIFFUSIVITY_M2_S = 100.0
RELEASE_RATE_M3_PER_HOUR = 1.0

OIL_ADIOS_ID = "AD01440"
OIL_NAME = "BONNY LIGHT, SHELL OIL"

SURFACE_Z_TOL_M = 1.0e-6
EARTH_RADIUS_M = 6_371_008.8


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def set_frozen_oil(model) -> dict[str, str]:
    attempts: list[str] = []

    if hasattr(model, "set_oiltype_by_id"):
        try:
            model.set_oiltype_by_id(OIL_ADIOS_ID)
            return {"method": "set_oiltype_by_id", "value": OIL_ADIOS_ID}
        except Exception as exc:
            attempts.append(
                f"set_oiltype_by_id({OIL_ADIOS_ID!r}): {type(exc).__name__}: {exc}"
            )

    if hasattr(model, "set_oiltype_by_name"):
        try:
            model.set_oiltype_by_name(OIL_NAME)
            return {"method": "set_oiltype_by_name", "value": OIL_NAME}
        except Exception as exc:
            attempts.append(
                f"set_oiltype_by_name({OIL_NAME!r}): {type(exc).__name__}: {exc}"
            )

    raise RuntimeError(
        "Could not set the frozen analogue oil. Attempts: " + " | ".join(attempts)
    )


def apply_scenario(model, scenario: dict[str, Any], scenario_keys: dict[str, str]) -> None:
    missing = sorted(set(scenario) - set(scenario_keys))
    if missing:
        raise KeyError(f"Scenario contains unmapped switches: {missing}")

    for semantic_name, value in scenario.items():
        model.set_config(scenario_keys[semantic_name], value)


def apply_production_numerics(model, config: dict[str, Any]) -> None:
    numerics = config.get("numerics", {})

    frozen_kh = float(numerics["horizontal_diffusivity_m2_per_s"])
    if abs(frozen_kh - HORIZONTAL_DIFFUSIVITY_M2_S) > 1e-12:
        raise RuntimeError(
            f"Frozen config Kh={frozen_kh} differs from corrected G4 Kh="
            f"{HORIZONTAL_DIFFUSIVITY_M2_S}"
        )

    model.set_config(
        "environment:constant:horizontal_diffusivity",
        HORIZONTAL_DIFFUSIVITY_M2_S,
    )
    applied = model.get_config("environment:constant:horizontal_diffusivity")
    if applied is None or abs(float(applied) - HORIZONTAL_DIFFUSIVITY_M2_S) > 1e-12:
        raise RuntimeError(f"Horizontal diffusivity was not applied: {applied}")

    model.set_config("drift:vertical_mixing_at_surface", False)

    optional_settings = {
        "vertical_mixing_diffusivity_model": "vertical_mixing:diffusivitymodel",
        "vertical_mixing_timestep_seconds": "vertical_mixing:timestep",
        "vertical_mixing_background_diffusivity_m2_per_s":
            "vertical_mixing:background_diffusivity",
    }
    for source_key, od_key in optional_settings.items():
        if source_key in numerics:
            value = numerics[source_key]
            if source_key == "vertical_mixing_timestep_seconds":
                value = float(value)
            model.set_config(od_key, value)


def orient_trajectory_time(da: xr.DataArray) -> np.ndarray:
    arr = np.asarray(da.values)
    dims = list(da.dims)

    if arr.ndim != 2:
        raise ValueError(f"{da.name} expected 2-D trajectory/time data, got {arr.shape}")

    if dims == ["trajectory", "time"]:
        return arr

    if "trajectory" in dims and "time" in dims:
        return np.moveaxis(
            arr,
            [dims.index("trajectory"), dims.index("time")],
            [0, 1],
        )

    raise ValueError(f"{da.name} lacks trajectory/time dimensions: {dims}")


def lonlat_to_local_xy_m(lon, lat):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    dlon = np.radians(lon - RELEASE_LON)
    dlat = np.radians(lat - RELEASE_LAT)

    x = EARTH_RADIUS_M * math.cos(math.radians(RELEASE_LAT)) * dlon
    y = EARTH_RADIUS_M * dlat
    return x, y


def local_xy_to_lonlat(x_m: float, y_m: float) -> tuple[float, float]:
    lon = RELEASE_LON + math.degrees(
        x_m / (EARTH_RADIUS_M * math.cos(math.radians(RELEASE_LAT)))
    )
    lat = RELEASE_LAT + math.degrees(y_m / EARTH_RADIUS_M)
    return lon, lat


def convex_hull(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    pts = pts[np.all(np.isfinite(pts), axis=1)]

    if len(pts) <= 1:
        return pts

    pts = np.unique(pts, axis=0)
    if len(pts) <= 2:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (
            (a[0] - o[0]) * (b[1] - o[1])
            - (a[1] - o[1]) * (b[0] - o[0])
        )

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))

    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def polygon_area_m2(poly: np.ndarray) -> float:
    poly = np.asarray(poly, dtype=float)
    if len(poly) < 3:
        return 0.0

    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(
        np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    )


def central_90_footprint_area_km2(lon_final, lat_final) -> tuple[float, int]:
    valid = np.isfinite(lon_final) & np.isfinite(lat_final)
    lon = np.asarray(lon_final)[valid]
    lat = np.asarray(lat_final)[valid]

    if len(lon) < 3:
        return 0.0, len(lon)

    x, y = lonlat_to_local_xy_m(lon, lat)
    cx = float(np.mean(x))
    cy = float(np.mean(y))

    r = np.hypot(x - cx, y - cy)
    cutoff = float(np.quantile(r, 0.90))
    keep = r <= cutoff

    pts = np.column_stack([x[keep], y[keep]])
    hull = convex_hull(pts)

    return polygon_area_m2(hull) / 1.0e6, int(np.sum(keep))


def status_mapping(da: xr.DataArray) -> dict[int, str]:
    values = da.attrs.get("flag_values", [])
    meanings = da.attrs.get("flag_meanings", "")

    names = meanings.split() if isinstance(meanings, str) else list(meanings)
    vals = [int(v) for v in np.asarray(values).ravel()] if len(np.asarray(values).ravel()) else []

    return {
        vals[i]: str(names[i])
        for i in range(min(len(vals), len(names)))
    }


def shoreline_contact_by_trajectory(ds: xr.Dataset) -> np.ndarray:
    ntraj = int(ds.sizes.get("trajectory", 0))
    contact = np.zeros(ntraj, dtype=bool)

    if "status" in ds:
        status = orient_trajectory_time(ds["status"]).astype(float)
        mapping = status_mapping(ds["status"])

        stranded_codes = [
            code for code, label in mapping.items()
            if "strand" in label.lower()
        ]
        for code in stranded_codes:
            contact |= np.any(status == code, axis=1)

    if "land_binary_mask" in ds:
        land = orient_trajectory_time(ds["land_binary_mask"]).astype(float)
        contact |= np.any(np.isfinite(land) & (land >= 0.5), axis=1)

    return contact


def sum_final(ds: xr.Dataset, variable: str) -> float | None:
    if variable not in ds:
        return None

    da = ds[variable]
    arr = np.asarray(da.values, dtype=float)

    if "time" in da.dims:
        arr = np.take(arr, -1, axis=da.dims.index("time"))

    return float(np.nansum(arr))


def mean_final(ds: xr.Dataset, variable: str) -> float | None:
    if variable not in ds:
        return None

    da = ds[variable]
    arr = np.asarray(da.values, dtype=float)

    if "time" in da.dims:
        arr = np.take(arr, -1, axis=da.dims.index("time"))

    return float(np.nanmean(arr))


def extract_metrics(nc_path: Path) -> dict[str, Any]:
    ds = xr.open_dataset(nc_path)

    try:
        lon = orient_trajectory_time(ds["lon"]).astype(float)
        lat = orient_trajectory_time(ds["lat"]).astype(float)

        lon_final = lon[:, -1]
        lat_final = lat[:, -1]

        valid_final = np.isfinite(lon_final) & np.isfinite(lat_final)
        x_final, y_final = lonlat_to_local_xy_m(
            lon_final[valid_final],
            lat_final[valid_final],
        )

        centroid_x_m = float(np.mean(x_final))
        centroid_y_m = float(np.mean(y_final))
        centroid_disp_km = math.hypot(centroid_x_m, centroid_y_m) / 1000.0
        centroid_lon, centroid_lat = local_xy_to_lonlat(
            centroid_x_m,
            centroid_y_m,
        )

        area90, count90 = central_90_footprint_area_km2(
            lon_final,
            lat_final,
        )

        z = orient_trajectory_time(ds["z"]).astype(float)
        times = np.asarray(ds["time"].values)
        post_release = times >= np.datetime64(RELEASE_END_UTC)

        z_post = z[:, post_release]
        valid_z = np.isfinite(z_post)
        surface = valid_z & (z_post >= -SURFACE_Z_TOL_M)
        surface_exposure = float(surface.sum() / valid_z.sum())

        z_final = z[:, -1]
        contact = shoreline_contact_by_trajectory(ds)

        mass_oil = sum_final(ds, "mass_oil")
        mass_evap = sum_final(ds, "mass_evaporated")
        mass_disp = sum_final(ds, "mass_dispersed")
        mass_biodeg = sum_final(ds, "mass_biodegraded")

        known_mass = [
            v for v in [mass_oil, mass_evap, mass_disp, mass_biodeg]
            if v is not None
        ]

        return {
            "final_valid_particles": int(valid_final.sum()),
            "centroid": {
                "east_km": centroid_x_m / 1000.0,
                "north_km": centroid_y_m / 1000.0,
                "displacement_km": centroid_disp_km,
                "longitude_deg_east": centroid_lon,
                "latitude_deg_north": centroid_lat,
            },
            "central_90_footprint_area_km2": float(area90),
            "central_90_particle_count": int(count90),
            "surface_exposure_fraction_6h_to_48h": surface_exposure,
            "shoreline_contact_probability": float(np.mean(contact)),
            "shoreline_contact_particles": int(contact.sum()),
            "vertical": {
                "overall_min_z_m": float(np.nanmin(z)),
                "final_min_z_m": float(np.nanmin(z_final)),
                "final_mean_z_m": float(np.nanmean(z_final)),
                "final_subsurface_particles": int(np.sum(z_final < 0)),
            },
            "fate": {
                "mass_oil_48h": mass_oil,
                "mass_evaporated_48h": mass_evap,
                "mass_dispersed_48h": mass_disp,
                "mass_biodegraded_48h": mass_biodeg,
                "known_mass_total_48h": (
                    float(sum(known_mass)) if known_mass else None
                ),
                "mean_fraction_evaporated_48h": mean_final(
                    ds, "fraction_evaporated"
                ),
                "mean_water_fraction_48h": mean_final(
                    ds, "water_fraction"
                ),
            },
        }
    finally:
        ds.close()


def is_complete_trajectory(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        ds = xr.open_dataset(path)
        try:
            if int(ds.sizes.get("trajectory", -1)) != N_ELEMENTS:
                return False

            times = np.asarray(ds["time"].values)
            if times.size < 49:
                return False

            if np.datetime64(times[-1]) < np.datetime64("2020-03-03T00:00:00"):
                return False

            kh = ds.attrs.get(
                "config_environment:constant:horizontal_diffusivity"
            )
            if kh is None or abs(float(kh) - HORIZONTAL_DIFFUSIVITY_M2_S) > 1e-12:
                return False

            return True
        finally:
            ds.close()
    except Exception:
        return False


def worker_run(scenario_name: str, out_nc: Path) -> int:
    config = load_config(DEFAULT_CONFIG)
    config_errors, _ = validate_config(config)
    if config_errors:
        raise RuntimeError(
            "Frozen configuration failed validation: "
            + "; ".join(config_errors)
        )

    if scenario_name not in SCENARIOS:
        raise KeyError(scenario_name)

    helper04 = load_script_module(
        "g2_reader_check_production",
        ROOT / "scripts" / "04_opendrift_reader_check.py",
    )
    helper05 = load_script_module(
        "g2_scenario_check_production",
        ROOT / "scripts" / "05_verify_scenario_switches.py",
    )

    mappings = helper04.STANDARD_NAME_MAPPINGS
    scenario_keys = helper05.SCENARIO_KEYS
    scenario = config["scenarios"][scenario_name]

    from opendrift.models.openoil import OpenOil
    from opendrift.readers import reader_netCDF_CF_generic

    paths = {
        section: ROOT / "data" / "raw" / config["forcing"][section]["filename"]
        for section in mappings
    }

    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing forcing file(s): " + ", ".join(missing))

    np.random.seed(MASTER_SEED)
    random.seed(MASTER_SEED)

    if out_nc.exists() and not is_complete_trajectory(out_nc):
        out_nc.unlink()

    model = OpenOil(seed=MASTER_SEED, loglevel=20)

    readers = {}
    for section, path in paths.items():
        readers[section] = reader_netCDF_CF_generic.Reader(
            str(path),
            standard_name_mapping=mappings[section],
        )
    model.add_reader(list(readers.values()))

    if not bool(model.get_config("general:use_auto_landmask")):
        raise RuntimeError("OpenDrift automatic landmask is not enabled.")

    apply_scenario(model, scenario, scenario_keys)
    apply_production_numerics(model, config)
    oil_selection = set_frozen_oil(model)

    model.seed_elements(
        lon=RELEASE_LON,
        lat=RELEASE_LAT,
        radius=RELEASE_RADIUS_M,
        number=N_ELEMENTS,
        time=[START_UTC, RELEASE_END_UTC],
        z=0,
        m3_per_hour=RELEASE_RATE_M3_PER_HOUR,
    )

    print(
        f"PRODUCTION START | {scenario_name} | "
        f"N={N_ELEMENTS} | dt={CALCULATION_STEP_SECONDS/60:g} min | "
        f"Kh={HORIZONTAL_DIFFUSIVITY_M2_S:g} m2/s | "
        f"seed={MASTER_SEED} | oil={oil_selection}",
        flush=True,
    )

    model.run(
        duration=timedelta(hours=48),
        time_step=CALCULATION_STEP_SECONDS,
        time_step_output=OUTPUT_STEP_SECONDS,
        outfile=str(out_nc),
    )

    if not is_complete_trajectory(out_nc):
        raise RuntimeError(
            f"Production worker produced incomplete trajectory: {out_nc}"
        )

    print(
        f"PRODUCTION PASS | {scenario_name} | {out_nc}",
        flush=True,
    )
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--scenario")
    parser.add_argument("--outfile")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.worker:
        if not args.scenario or not args.outfile:
            raise SystemExit("--worker requires --scenario and --outfile")
        return worker_run(
            args.scenario,
            Path(args.outfile).resolve(),
        )

    config = load_config(DEFAULT_CONFIG)
    config_errors, config_warnings = validate_config(config)
    if config_errors:
        raise RuntimeError(
            "Frozen configuration failed validation: "
            + "; ".join(config_errors)
        )

    try:
        installed = version("opendrift")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "OpenDrift is not installed in the active Python environment."
        ) from exc

    expected = str(config["software"]["opendrift"])
    if installed != expected:
        raise RuntimeError(
            f"OpenDrift version mismatch: installed={installed}, expected={expected}"
        )

    run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    out_dir = ROOT / "outputs" / "production"
    diag_dir = ROOT / "diagnostics" / "production"
    out_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

    summary_path = diag_dir / f"production_summary_{run_tag}.json"
    latest_path = diag_dir / "production_summary_latest.json"

    summary: dict[str, Any] = {
        "generated_utc": run_tag,
        "scope": "Final production scenario matrix after corrected Gate G4",
        "config_sha256": config_digest(DEFAULT_CONFIG),
        "opendrift_version": installed,
        "authoritative_numerics": {
            "horizontal_diffusivity_m2_per_s": HORIZONTAL_DIFFUSIVITY_M2_S,
            "calculation_step_seconds": CALCULATION_STEP_SECONDS,
            "calculation_step_minutes": CALCULATION_STEP_SECONDS / 60,
            "output_step_seconds": OUTPUT_STEP_SECONDS,
            "particle_count": N_ELEMENTS,
            "master_seed": MASTER_SEED,
        },
        "release": {
            "longitude_deg_east": RELEASE_LON,
            "latitude_deg_north": RELEASE_LAT,
            "radius_m": RELEASE_RADIUS_M,
            "start_utc": START_UTC.isoformat(),
            "release_end_utc": RELEASE_END_UTC.isoformat(),
            "run_end_utc": RUN_END_UTC.isoformat(),
            "rate_m3_per_hour": RELEASE_RATE_M3_PER_HOUR,
        },
        "oil": {
            "adios_id": OIL_ADIOS_ID,
            "name": OIL_NAME,
        },
        "scenario_order": SCENARIOS,
        "scenario_settings": {
            name: config["scenarios"][name]
            for name in SCENARIOS
        },
        "warnings": list(config_warnings),
        "runs": {},
        "paired_attribution": {},
        "overall_status": "RUNNING",
    }
    write_json(summary_path, summary)

    for scenario_name in SCENARIOS:
        existing = sorted(
            out_dir.glob(
                f"{scenario_name}_N{N_ELEMENTS}_dt{CALCULATION_STEP_SECONDS}s_"
                f"Kh{int(HORIZONTAL_DIFFUSIVITY_M2_S)}_seed{MASTER_SEED}_*.nc"
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        complete = next(
            (p for p in existing if is_complete_trajectory(p)),
            None,
        )

        if complete is not None:
            out_nc = complete
            source = "reused_complete_production_output"
            runtime_seconds = None
            print(
                f"REUSE | {scenario_name} | {out_nc.name}",
                flush=True,
            )
        else:
            out_nc = out_dir / (
                f"{scenario_name}_N{N_ELEMENTS}_dt{CALCULATION_STEP_SECONDS}s_"
                f"Kh{int(HORIZONTAL_DIFFUSIVITY_M2_S)}_seed{MASTER_SEED}_"
                f"{run_tag}.nc"
            )
            source = "fresh_isolated_worker"

            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--scenario", scenario_name,
                "--outfile", str(out_nc),
            ]

            print(f"LAUNCH | {scenario_name}", flush=True)
            t0 = time.perf_counter()
            completed = subprocess.run(cmd)
            runtime_seconds = time.perf_counter() - t0

            if completed.returncode != 0:
                raise RuntimeError(
                    f"Production worker failed for {scenario_name}: "
                    f"returncode={completed.returncode}"
                )

        metrics = extract_metrics(out_nc)

        summary["runs"][scenario_name] = {
            "status": "PASS",
            "source": source,
            "runtime_seconds": runtime_seconds,
            "output_netcdf": str(out_nc.relative_to(ROOT)),
            "metrics": metrics,
        }
        write_json(summary_path, summary)

        print(
            f"{scenario_name} | "
            f"centroid={metrics['centroid']['displacement_km']:.3f} km | "
            f"area90={metrics['central_90_footprint_area_km2']:.3f} km2 | "
            f"surface={metrics['surface_exposure_fraction_6h_to_48h']:.4f} | "
            f"shore={metrics['shoreline_contact_probability']:.4f}",
            flush=True,
        )

    def paired(left: str, right: str) -> dict[str, Any]:
        a = summary["runs"][left]["metrics"]
        b = summary["runs"][right]["metrics"]

        dx = b["centroid"]["east_km"] - a["centroid"]["east_km"]
        dy = b["centroid"]["north_km"] - a["centroid"]["north_km"]

        return {
            "from": left,
            "to": right,
            "centroid_endpoint_separation_km": math.hypot(dx, dy),
            "footprint_area_change_km2": (
                b["central_90_footprint_area_km2"]
                - a["central_90_footprint_area_km2"]
            ),
            "surface_exposure_change": (
                b["surface_exposure_fraction_6h_to_48h"]
                - a["surface_exposure_fraction_6h_to_48h"]
            ),
            "shoreline_probability_change": (
                b["shoreline_contact_probability"]
                - a["shoreline_contact_probability"]
            ),
        }

    summary["paired_attribution"] = {
        "C0_to_C1": paired("C0_Hydro", "C1_Surface"),
        "C1_to_C2": paired("C1_Surface", "C2_Exchange"),
        "C2_to_C3": paired("C2_Exchange", "C3_Full_fate"),
    }

    summary["overall_status"] = "PASS"
    summary["interpretation"] = (
        "All four production scenarios completed under the authoritative corrected "
        "Gate G4 numerics. C0-C3 constitute the production attribution ladder; "
        "C3_Full_fate is the main full-fate production case."
    )

    write_json(summary_path, summary)
    write_json(latest_path, summary)

    print("\n" + "=" * 84)
    print("PRODUCTION SCENARIO MATRIX: PASS")
    print(f"Summary: {summary_path}")
    print(f"Latest : {latest_path}")

    print("\nPAIRED ATTRIBUTION")
    for key, value in summary["paired_attribution"].items():
        print(
            f"{key}: "
            f"endpoint={value['centroid_endpoint_separation_km']:.3f} km | "
            f"dA90={value['footprint_area_change_km2']:.3f} km2 | "
            f"dSurface={value['surface_exposure_change']:.5f} | "
            f"dPshore={value['shoreline_probability_change']:.5f}"
        )

    print("\nSTOP HERE. Review production matrix before final analysis/figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
