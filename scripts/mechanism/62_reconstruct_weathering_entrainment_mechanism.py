#!/usr/bin/env python3
"""
Script 62: MECHANISM mechanism reconstruction and figure.

Purpose
-------
Demonstrate the mechanism behind the higher surface exposure in C3 relative
to C2 using the authoritative production NetCDFs and the exact OpenDrift
1.14.12 Li et al. (2017) entrainment formulation.

Mechanistic chain examined
--------------------------
water fraction -> oil viscosity -> Ohnesorge number -> wave entrainment rate
-> surface/subsurface occupancy

Notes
-----
- Uses output states already written by the production runs.
- Does NOT rerun OpenOil.
- Entrainment rate is reconstructed with the exact installed OpenDrift
  oil_wave_entrainment_rate_li2017() function.
- The Ohnesorge number is reconstructed from the same variables and formula
  used by that function.
- "Observed transitions" are transitions between hourly archived outputs and
  therefore are a lower bound on true sub-hourly wave-entrainment events.

Outputs
-------
diagnostics/validation/weathering_entrainment_mechanism_timeseries.csv
diagnostics/validation/weathering_entrainment_mechanism_summary.txt
diagnostics/validation/weathering_entrainment_mechanism_summary.json
C2_C3_weathering_entrainment_mechanism.png
C2_C3_weathering_entrainment_mechanism.pdf
C2_C3_weathering_entrainment_mechanism.svg
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from opendrift.models.physics_methods import (
    oil_wave_entrainment_rate_li2017,
    PhysicsMethods,
)

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "diagnostics" / "validation"
OUTDIR.mkdir(parents=True, exist_ok=True)

PATTERNS = {
    "C2": "C2_Exchange_N1000_dt1800s_Kh100_seed20220901_*.nc",
    "C3": "C3_Full_fate_N1000_dt1800s_Kh100_seed20220901_*.nc",
}

INTERFACIAL_TENSION_N_M = 0.030109424
SURFACE_TOL_M = 1.0e-6
G = 9.81


def latest_file(pattern: str) -> Path:
    files = sorted(
        (ROOT / "outputs" / "production").glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(pattern)
    return files[0]


def orient(da: xr.DataArray) -> np.ndarray:
    arr = np.asarray(da.values, dtype=float)
    dims = list(da.dims)

    if arr.ndim == 1:
        return arr

    if "trajectory" in dims and "time" in dims:
        return np.moveaxis(
            arr,
            [dims.index("trajectory"), dims.index("time")],
            [0, 1],
        )

    raise ValueError(f"Cannot orient {da.name}: dims={dims}, shape={arr.shape}")


def nanmean_axis0(arr):
    with np.errstate(invalid="ignore"):
        return np.nanmean(arr, axis=0)


def choose_wave_period(ds: xr.Dataset) -> np.ndarray:
    tm_name = (
        "sea_surface_wave_mean_period_from_variance_spectral_density_"
        "second_frequency_moment"
    )
    tp_name = "sea_surface_wave_period_at_variance_spectral_density_maximum"

    wind = np.hypot(orient(ds["x_wind"]), orient(ds["y_wind"]))

    if tm_name in ds:
        tm = orient(ds[tm_name])
        T = tm.copy() if np.nanmax(tm) > 0 else None
    else:
        T = None

    if T is None and tp_name in ds:
        tp = orient(ds[tp_name])
        if np.nanmax(tp) > 0:
            T = tp.copy()

    if T is None:
        omega = 5.0 * np.ones_like(wind)
        mask = wind > 0
        omega[mask] = 0.877 * 9.81 / (1.17 * wind[mask])
        T = 2 * np.pi / omega

    positive = T > 0
    if np.any(~positive) and np.any(positive):
        T[~positive] = np.nanmean(T[positive])

    return T


def seawater_density_from_output(ds: xr.Dataset) -> np.ndarray:
    temp = orient(ds["sea_water_temperature"])
    sal = orient(ds["sea_water_salinity"])

    # Archived output contains Celsius for most hours but Kelvin at the final
    # 48 h record. Convert elementwise so mixed-unit arrays are handled safely.
    temp_c = np.where(np.isfinite(temp) & (temp > 100.0), temp - 273.15, temp)

    return PhysicsMethods.sea_water_density(T=temp_c, S=sal)


def process(label: str, path: Path):
    with xr.open_dataset(path) as ds:
        times = np.asarray(ds["time"].values)
        t0 = times[0]
        hours = ((times - t0) / np.timedelta64(1, "h")).astype(float)

        water_fraction = orient(ds["water_fraction"])
        viscosity = orient(ds["viscosity"])
        oil_density = orient(ds["density"])
        z = orient(ds["z"])
        hs = orient(ds["sea_surface_wave_significant_height"])
        wind = np.hypot(orient(ds["x_wind"]), orient(ds["y_wind"]))
        period = choose_wave_period(ds)
        rho_w = seawater_density_from_output(ds)

        with np.errstate(divide="ignore", invalid="ignore"):
            breaking = 0.032 * (wind - 5.0) / period
        breaking = np.where(np.isfinite(breaking), breaking, 0.0)
        breaking[breaking < 0] = 0.0

        dynamic_viscosity = viscosity * oil_density

        q = oil_wave_entrainment_rate_li2017(
            dynamic_viscosity=dynamic_viscosity,
            oil_density=oil_density,
            interfacial_tension=INTERFACIAL_TENSION_N_M,
            significant_wave_height=hs,
            wave_breaking_fraction=breaking,
            sea_water_density=rho_w,
        )

        delta_rho = rho_w - oil_density
        with np.errstate(divide="ignore", invalid="ignore"):
            d_o = 4.0 * np.sqrt(
                INTERFACIAL_TENSION_N_M / (delta_rho * G)
            )
            oh = dynamic_viscosity / np.sqrt(
                oil_density * INTERFACIAL_TENSION_N_M * d_o
            )

        vertical_mixing_dt_s = float(
            ds.attrs.get("config_vertical_mixing:timestep", 60.0)
        )
        with np.errstate(over="ignore", invalid="ignore"):
            p_ent = 1.0 - np.exp(-q * vertical_mixing_dt_s)

        valid_z = np.isfinite(z)
        surface = valid_z & (z >= -SURFACE_TOL_M)
        surface_fraction = np.sum(surface, axis=0) / np.sum(valid_z, axis=0)

        transitions = np.zeros(z.shape[1], dtype=float)
        eligible_surface = np.zeros(z.shape[1], dtype=float)
        for j in range(1, z.shape[1]):
            prev_valid = np.isfinite(z[:, j - 1])
            now_valid = np.isfinite(z[:, j])
            prev_surface = prev_valid & (z[:, j - 1] >= -SURFACE_TOL_M)
            now_subsurface = now_valid & (z[:, j] < -SURFACE_TOL_M)

            transitions[j] = np.sum(prev_surface & now_subsurface)
            eligible_surface[j] = np.sum(prev_surface)

        transition_fraction = np.divide(
            transitions,
            eligible_surface,
            out=np.full_like(transitions, np.nan),
            where=eligible_surface > 0,
        )

        result = pd.DataFrame({
            "scenario": label,
            "hour": hours,
            "mean_water_fraction": nanmean_axis0(water_fraction),
            "mean_kinematic_viscosity_cSt": nanmean_axis0(viscosity) * 1e6,
            "mean_dynamic_viscosity_Pa_s": nanmean_axis0(dynamic_viscosity),
            "mean_oil_density_kg_m3": nanmean_axis0(oil_density),
            "mean_ohnesorge": nanmean_axis0(oh),
            "mean_entrainment_rate_s-1": nanmean_axis0(q),
            "mean_entrainment_probability_60s": nanmean_axis0(p_ent),
            "mean_wave_breaking_fraction": nanmean_axis0(breaking),
            "mean_significant_wave_height_m": nanmean_axis0(hs),
            "mean_wind_speed_m_s": nanmean_axis0(wind),
            "surface_fraction": surface_fraction,
            "archived_surface_to_subsurface_count": transitions,
            "archived_transition_fraction": transition_fraction,
        })

        raw = {
            "vertical_mixing_timestep_s": vertical_mixing_dt_s,
            "n_trajectories": int(z.shape[0]),
            "n_times": int(z.shape[1]),
            "time_start": str(times[0]),
            "time_end": str(times[-1]),
        }

        return result, raw


def safe_ratio(a, b):
    if b == 0 or not np.isfinite(b):
        return None
    return float(a / b)


def main():
    frames = []
    meta = {}

    for label, pattern in PATTERNS.items():
        path = latest_file(pattern)
        print(f"Processing {label}: {path.name}", flush=True)
        frame, info = process(label, path)
        frames.append(frame)
        meta[label] = {
            "file": str(path.relative_to(ROOT)),
            **info,
        }

    df = pd.concat(frames, ignore_index=True)

    csv_path = OUTDIR / "weathering_entrainment_mechanism_timeseries.csv"
    df.to_csv(csv_path, index=False)

    post = df[df["hour"] >= 6].copy()
    summary = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "interfacial_tension_N_m": INTERFACIAL_TENSION_N_M,
        "runs": meta,
        "post_release_6_to_48h": {},
    }

    metrics = [
        "mean_water_fraction",
        "mean_kinematic_viscosity_cSt",
        "mean_dynamic_viscosity_Pa_s",
        "mean_ohnesorge",
        "mean_entrainment_rate_s-1",
        "mean_entrainment_probability_60s",
        "surface_fraction",
        "archived_surface_to_subsurface_count",
        "archived_transition_fraction",
    ]

    for label in ["C2", "C3"]:
        d = post[post["scenario"] == label]
        summary["post_release_6_to_48h"][label] = {
            m: float(np.nanmean(d[m].to_numpy(dtype=float)))
            for m in metrics
        }

    c2 = summary["post_release_6_to_48h"]["C2"]
    c3 = summary["post_release_6_to_48h"]["C3"]

    summary["C3_vs_C2"] = {
        "kinematic_viscosity_ratio":
            safe_ratio(c3["mean_kinematic_viscosity_cSt"],
                       c2["mean_kinematic_viscosity_cSt"]),
        "dynamic_viscosity_ratio":
            safe_ratio(c3["mean_dynamic_viscosity_Pa_s"],
                       c2["mean_dynamic_viscosity_Pa_s"]),
        "ohnesorge_ratio":
            safe_ratio(c3["mean_ohnesorge"], c2["mean_ohnesorge"]),
        "entrainment_rate_ratio":
            safe_ratio(c3["mean_entrainment_rate_s-1"],
                       c2["mean_entrainment_rate_s-1"]),
        "entrainment_probability_ratio":
            safe_ratio(c3["mean_entrainment_probability_60s"],
                       c2["mean_entrainment_probability_60s"]),
        "surface_fraction_difference":
            float(c3["surface_fraction"] - c2["surface_fraction"]),
        "archived_transition_count_ratio":
            safe_ratio(c3["archived_surface_to_subsurface_count"],
                       c2["archived_surface_to_subsurface_count"]),
    }

    json_path = OUTDIR / "weathering_entrainment_mechanism_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "MECHANISM — C2/C3 MECHANISM RECONSTRUCTION",
        "=" * 78,
        f"Generated UTC: {summary['generated_utc']}",
        f"Interfacial tension used: {INTERFACIAL_TENSION_N_M:.9f} N/m",
        "",
        "POST-RELEASE MEANS, 6-48 h",
        "-" * 78,
    ]

    for label in ["C2", "C3"]:
        s = summary["post_release_6_to_48h"][label]
        lines.extend([
            f"{label}:",
            f"  water fraction                 = {s['mean_water_fraction']:.6g}",
            f"  kinematic viscosity            = {s['mean_kinematic_viscosity_cSt']:.6g} cSt",
            f"  dynamic viscosity              = {s['mean_dynamic_viscosity_Pa_s']:.6g} Pa s",
            f"  Ohnesorge number               = {s['mean_ohnesorge']:.6g}",
            f"  entrainment rate               = {s['mean_entrainment_rate_s-1']:.6g} s^-1",
            f"  entrainment probability / 60 s = {s['mean_entrainment_probability_60s']:.6g}",
            f"  surface fraction               = {s['surface_fraction']:.6g}",
            f"  archived S->subsurface count   = {s['archived_surface_to_subsurface_count']:.6g} per hourly output",
            f"  archived transition fraction   = {s['archived_transition_fraction']:.6g}",
            "",
        ])

    lines.append("C3 / C2 OR C3 - C2")
    lines.append("-" * 78)
    for k, v in summary["C3_vs_C2"].items():
        lines.append(f"{k}: {v}")

    lines.extend([
        "",
        "Important:",
        "- The entrainment rate is reconstructed using the installed OpenDrift",
        "  Li et al. (2017) implementation and archived model states.",
        "- Hourly surface-to-subsurface transitions are a lower bound on true",
        "  entrainment events because the internal mixing timestep is shorter.",
    ])

    txt_path = OUTDIR / "weathering_entrainment_mechanism_summary.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(3, 2, figsize=(10.5, 10.0))
    axes = axes.ravel()

    plotting = [
        ("mean_water_fraction", "Water fraction", "(a) Emulsification state"),
        ("mean_kinematic_viscosity_cSt", "Kinematic viscosity (cSt)",
         "(b) Oil-emulsion viscosity"),
        ("mean_ohnesorge", "Ohnesorge number", "(c) Viscous resistance"),
        ("mean_entrainment_rate_s-1", r"Entrainment rate $Q$ (s$^{-1}$)",
         "(d) Li et al. entrainment rate"),
        ("surface_fraction", "Surface fraction",
         "(e) Surface occupancy"),
        ("archived_transition_fraction",
         "Hourly surface-to-subsurface transition fraction",
         "(f) Archived vertical transitions"),
    ]

    for ax, (column, ylabel, title) in zip(axes, plotting):
        for label in ["C2", "C3"]:
            d = df[df["scenario"] == label]
            ax.plot(d["hour"], d[column], label=label, linewidth=1.6)

        ax.axvline(6, linestyle="--", linewidth=1.0)
        ax.set_xlabel("Time since release start (h)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

    axes[0].legend(frameon=False)

    fig.text(
        0.5, 0.005,
        "Dashed vertical line marks the end of the 6 h release. "
        "C2: exchange only; C3: full fate/weathering.",
        ha="center", fontsize=9,
    )

    plt.tight_layout(rect=[0, 0.025, 1, 1])

    png = ROOT / "C2_C3_weathering_entrainment_mechanism.png"
    pdf = ROOT / "C2_C3_weathering_entrainment_mechanism.pdf"
    svg = ROOT / "C2_C3_weathering_entrainment_mechanism.svg"

    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(svg, bbox_inches="tight")

    print("")
    print("\n".join(lines))
    print("=" * 78)
    print(f"Saved: {csv_path.relative_to(ROOT)}")
    print(f"Saved: {txt_path.relative_to(ROOT)}")
    print(f"Saved: {json_path.relative_to(ROOT)}")
    print(f"Saved: {png.name}")
    print(f"Saved: {pdf.name}")
    print(f"Saved: {svg.name}")

    plt.show()


if __name__ == "__main__":
    main()
