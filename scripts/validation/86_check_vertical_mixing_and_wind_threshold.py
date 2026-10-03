#!/usr/bin/env python3
"""
86_check_VERTICAL MIXING_vertical_mixing_and_wind_threshold.py

check for two outstanding issues:

VERTICAL MIXING-A. Verify the sign and implementation details of the OpenDrift vertical
      diffusivity-gradient correction used by the installed OpenDrift package.

VERTICAL MIXING-B. Run an isolated well-mixed-condition diagnostic using the same numerical
      update form as OpenDrift, comparing the installed/source sign with the
      opposite sign under:
        (i) a smooth K(z) profile, and
        (ii) the production Sundby1983 profile with the study MLD/background K.

VERTICAL MIXING-C. Extract the 49 hourly ERA5 10 m wind values at the Jubilee release point,
      calculate the implemented 5 m/s breaking-wave threshold diagnostic, and
      align them with the Fig. 3 C2/C3 occupancy time series.

This script does NOT modify any production NetCDF file.

Outputs
-------
diagnostics/validation/vertical_mixing_source_check.txt
diagnostics/validation/vertical_mixing_summary.csv
diagnostics/validation/vertical_mixing_histograms.csv
diagnostics/validation/wind_threshold_occupancy.csv
diagnostics/validation/wind_threshold_summary.txt

figures/vertical_mixing_diagnostic.pdf/png/svg
figures/wind_threshold_occupancy.pdf/png/svg

Interpretation discipline
-------------------------
* The well-mixed test isolates the turbulent random-walk update. It intentionally
  excludes buoyant rise, wave entrainment, weathering, advection, and stranding.
* The wind/occupancy comparison is descriptive. Temporal alignment or correlation
  does not prove causation.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
import re
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from opendrift.models.oceandrift import OceanDrift
from opendrift.models.physics_methods import verticaldiffusivity_Sundby1983


# =============================================================================
# PATHS / CONSTANTS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "scripts" else SCRIPT_DIR

DIAG = ROOT / "diagnostics" / "validation"
FIGDIR = ROOT / "figures"
DIAG.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

RELEASE_LAT = 4.60000
RELEASE_LON = -2.89000
START = pd.Timestamp("2020-03-01T00:00:00")
END = pd.Timestamp("2020-03-03T00:00:00")

# Production vertical-mixing settings used in the manuscript.
MLD_M = 10.53
WIND_MEAN_M_S = 5.667
BACKGROUND_K_M2_S = 1.2e-5
DT_MIX_S = 60.0

# Well-mixed numerical experiment.
N_PARTICLES = 40000
TEST_HOURS = 6.0
TEST_COLUMN_DEPTH_M = 20.0
N_HIST_BINS = 40
RNG_SEED = 20220901

FIG3_CSV = ROOT / "figures" / "Fig3_vertical_timeseries.csv"


# =============================================================================
# SOURCE CHECK
# =============================================================================

def opendrift_version() -> str:
    try:
        return version("opendrift")
    except PackageNotFoundError:
        return "unknown"


def check_installed_source() -> dict:
    src = inspect.getsource(OceanDrift.vertical_mixing)

    lines = src.splitlines()
    key_lines = []
    for line in lines:
        if (
            "gradK =" in line
            or "dKdz =" in line
            or "self.elements.z = self.elements.z -" in line
            or "Kz =" in line
            or "should be evaluated" in line
            or "z - dKdz*dt_mix" in line
        ):
            key_lines.append(line.rstrip())

    has_grad_definition = "gradK = -np.gradient(Kprofiles, mixing_z, axis=0)" in src
    has_update = (
        "self.elements.z = self.elements.z - self.elements.moving*(" in src
        and "dKdz*dt_mix - R*np.sqrt" in src
    )
    source_mentions_current_K = (
        "Kz is evaluated in zi" in src
        and "should be evaluated" in src
    )

    # OpenDrift's local variable is:
    #   gradK = - gradient(K, z)
    # followed by:
    #   z_new = z - gradK*dt + random
    # Therefore the deterministic physical drift is + (dK/dz)*dt
    # under the manuscript convention z=0 at surface, z<0 below.
    equivalent_plus_dKdz = bool(has_grad_definition and has_update)

    result = {
        "opendrift_distribution_version": opendrift_version(),
        "source_file": inspect.getsourcefile(OceanDrift.vertical_mixing),
        "gradK_internal_definition_detected": has_grad_definition,
        "position_update_detected": has_update,
        "source_notes_K_evaluated_at_current_index": source_mentions_current_K,
        "equivalent_physical_gradient_term": (
            "+(∂K/∂z) Δt" if equivalent_plus_dKdz else "UNRESOLVED"
        ),
        "key_source_lines": key_lines,
    }

    out = DIAG / "vertical_mixing_source_check.txt"
    text = [
        "VERTICAL MIXING VERTICAL-MIXING SOURCE CHECK",
        "=" * 88,
        f"Installed OpenDrift distribution: {result['opendrift_distribution_version']}",
        f"Source file: {result['source_file']}",
        "",
        "KEY RUNTIME SOURCE LINES",
        "-" * 88,
        *key_lines,
        "",
        "SIGN INTERPRETATION",
        "-" * 88,
    ]

    if equivalent_plus_dKdz:
        text.extend([
            "The runtime source defines an internal variable as",
            "    gradK = - gradient(Kprofiles, mixing_z)",
            "and then updates particle z with",
            "    z_new = z - gradK*dt + random_term.",
            "",
            "Because the manuscript coordinate is z=0 at the surface and z<0 below,",
            "the two minus signs cancel. The deterministic gradient correction is",
            "therefore equivalent to:",
            "    +(∂K/∂z) Δt",
            "not -(∂K/∂z) Δt.",
            "",
            "Thus, if the manuscript currently writes a negative physical gradient",
            "term with ∂K/∂z defined in the usual z-positive-upward sense, that is a",
            "documentation/transcription sign error rather than the runtime source sign.",
        ])
    else:
        text.append(
            "Automatic source-pattern recognition could not resolve the sign. "
            "Inspect the key source lines manually."
        )

    text.extend([
        "",
        "OFFSET-EVALUATION NOTE",
        "-" * 88,
    ])

    if source_mentions_current_K:
        text.extend([
            "The installed OpenDrift source explicitly notes that Kz is evaluated at",
            "the current discrete depth index zi, while commenting that an offset",
            "position would be preferable. Therefore the manuscript should document",
            "the implemented OpenDrift update rather than claim that this version",
            "actually evaluates K at the offset position.",
        ])
    else:
        text.append(
            "No explicit current-index/offset comment was detected automatically."
        )

    out.write_text("\n".join(text) + "\n", encoding="utf-8")
    return result


# =============================================================================
# WELL-MIXED DIAGNOSTIC
# =============================================================================

def reflect_0_H(z: np.ndarray, H: float) -> np.ndarray:
    """
    Reflect z at the surface (0) and a flat bottom (-H).
    Repeated reflection protects against rare overshoots crossing both bounds.
    """
    z = z.copy()
    for _ in range(4):
        above = z > 0
        if np.any(above):
            z[above] = -z[above]

        below = z < -H
        if np.any(below):
            z[below] = -2.0 * H - z[below]

        if not np.any(z > 0) and not np.any(z < -H):
            break
    return z


def run_random_walk(
    initial_z: np.ndarray,
    mixing_z: np.ndarray,
    K_profile: np.ndarray,
    hours: float,
    dt_s: float,
    seed: int,
    use_source_sign: bool,
    H: float,
) -> np.ndarray:
    """
    Source-equivalent isolated OpenDrift random-walk update.

    OpenDrift runtime form:
        gradK_internal = - gradient(K, mixing_z)
        z_new = z - gradK_internal*dt + random

    Reversed-sign comparison changes ONLY the deterministic gradient term.
    """
    z = initial_z.copy()
    K_profile = np.asarray(K_profile, dtype=float).reshape(-1)
    mixing_z = np.asarray(mixing_z, dtype=float).reshape(-1)

    if len(K_profile) != len(mixing_z):
        raise ValueError("K_profile and mixing_z must have the same length.")

    depth_levels = -mixing_z
    indices = np.arange(len(mixing_z), dtype=float)

    gradK_internal = -np.gradient(K_profile, mixing_z)
    gradK_internal[np.abs(gradK_internal) < 1e-10] = 0.0

    nsteps = int(round(hours * 3600.0 / dt_s))
    rng = np.random.default_rng(seed)
    rvar = 1.0 / 3.0

    for _ in range(nsteps):
        # Mirror OpenDrift's interp1d + round behavior using a monotone grid.
        idx_float = np.interp(
            -z,
            depth_levels,
            indices,
            left=0.0,
            right=float(len(indices) - 1),
        )
        zi = np.rint(idx_float).astype(int)
        zi = np.clip(zi, 0, len(indices) - 1)

        Kz = K_profile[zi]
        dK_internal = gradK_internal[zi]

        R = 2.0 * rng.random(len(z)) - 1.0
        random_term = R * np.sqrt(Kz * abs(dt_s) * 2.0 / rvar)

        if use_source_sign:
            # Exact deterministic sign used by OpenDrift:
            z = z - dK_internal * dt_s + random_term
        else:
            # Diagnostic counterfactual: reverse the gradient term only.
            z = z + dK_internal * dt_s + random_term

        z = reflect_0_H(z, H)

    return z


def histogram_metrics(initial_z, final_z, H, nbins):
    bins = np.linspace(-H, 0.0, nbins + 1)
    h0, edges = np.histogram(initial_z, bins=bins)
    hf, _ = np.histogram(final_z, bins=bins)

    p0 = h0 / h0.sum()
    pf = hf / hf.sum()
    uniform = np.full(nbins, 1.0 / nbins)

    rmse_vs_uniform_rel = (
        np.sqrt(np.mean((pf - uniform) ** 2)) / (1.0 / nbins)
    )
    max_abs_dev_uniform_rel = (
        np.max(np.abs(pf - uniform)) / (1.0 / nbins)
    )
    total_variation_from_initial = 0.5 * np.sum(np.abs(pf - p0))

    centers = 0.5 * (edges[:-1] + edges[1:])

    summary = {
        "mean_depth_initial_m": float(np.mean(-initial_z)),
        "mean_depth_final_m": float(np.mean(-final_z)),
        "mean_depth_shift_m": float(np.mean(-final_z) - np.mean(-initial_z)),
        "hist_rmse_relative_to_uniform": float(rmse_vs_uniform_rel),
        "hist_max_relative_deviation_from_uniform": float(max_abs_dev_uniform_rel),
        "total_variation_distance_from_initial": float(total_variation_from_initial),
    }

    table = pd.DataFrame({
        "bin_center_depth_m": -centers,
        "initial_fraction": p0,
        "final_fraction": pf,
        "uniform_fraction": uniform,
    })

    return summary, table


def build_smooth_profile(H=20.0):
    depth = np.linspace(0.0, H, 201)
    mixing_z = -depth
    # Smooth positive profile: K decreases gradually with depth.
    K = 0.002 + 0.008 * (1.0 - depth / H)
    return mixing_z, K


def build_production_sundby_profile():
    # Match OpenDrift's production construction:
    # mixing_z_analytical = -arange(0, max(MLD)+2)
    depth = np.arange(0.0, math.ceil(MLD_M) + 2.0, 1.0)
    mixing_z = -depth

    wind = np.full((len(depth), 1), WIND_MEAN_M_S, dtype=float)
    depth2 = depth[:, None]
    mld = np.array([MLD_M], dtype=float)

    K = verticaldiffusivity_Sundby1983(
        wind,
        depth2,
        mixedlayerdepth=mld,
        background_diffusivity=BACKGROUND_K_M2_S,
    )
    K = np.asarray(K, dtype=float).reshape(-1)
    return mixing_z, K


def run_well_mixed_suite():
    rng = np.random.default_rng(RNG_SEED)
    initial = -rng.uniform(0.0, TEST_COLUMN_DEPTH_M, N_PARTICLES)

    profiles = {
        "smooth": build_smooth_profile(TEST_COLUMN_DEPTH_M),
        "production_sundby": build_production_sundby_profile(),
    }

    summary_rows = []
    hist_rows = []
    profile_rows = []

    for profile_name, (mixing_z, K) in profiles.items():
        source_final = run_random_walk(
            initial, mixing_z, K,
            TEST_HOURS, DT_MIX_S, RNG_SEED + 1,
            True, TEST_COLUMN_DEPTH_M,
        )
        reversed_final = run_random_walk(
            initial, mixing_z, K,
            TEST_HOURS, DT_MIX_S, RNG_SEED + 1,
            False, TEST_COLUMN_DEPTH_M,
        )

        for sign_name, final_z in [
            ("source_sign", source_final),
            ("reversed_sign", reversed_final),
        ]:
            metrics, hist = histogram_metrics(
                initial, final_z, TEST_COLUMN_DEPTH_M, N_HIST_BINS
            )
            summary_rows.append({
                "profile": profile_name,
                "gradient_sign": sign_name,
                "n_particles": N_PARTICLES,
                "duration_h": TEST_HOURS,
                "dt_s": DT_MIX_S,
                **metrics,
            })

            hist["profile"] = profile_name
            hist["gradient_sign"] = sign_name
            hist_rows.append(hist)

        profile_rows.append(pd.DataFrame({
            "profile": profile_name,
            "z_m": mixing_z,
            "depth_m": -mixing_z,
            "K_m2_s": K,
        }))

    summary = pd.DataFrame(summary_rows)
    hist = pd.concat(hist_rows, ignore_index=True)
    profiles_df = pd.concat(profile_rows, ignore_index=True)

    summary.to_csv(DIAG / "vertical_mixing_summary.csv", index=False)
    hist.to_csv(DIAG / "vertical_mixing_histograms.csv", index=False)
    profiles_df.to_csv(DIAG / "vertical_mixing_diffusivity_profiles.csv", index=False)

    # Plot: K profiles + final depth distributions.
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8))

    for col, profile_name in enumerate(["smooth", "production_sundby"]):
        axK = axes[0, col]
        p = profiles_df[profiles_df["profile"] == profile_name]
        axK.plot(p["K_m2_s"], p["depth_m"])
        axK.invert_yaxis()
        axK.set_xlabel(r"$K_z$ (m$^2$ s$^{-1}$)")
        axK.set_ylabel("Depth (m)")
        title = "Smooth profile" if profile_name == "smooth" else "Production Sundby profile"
        axK.set_title(f"({chr(97+col)}) {title}", loc="left", fontweight="bold")
        axK.grid(alpha=0.25)

        axH = axes[1, col]
        hh = hist[hist["profile"] == profile_name]
        # Initial/uniform only once
        one = hh[hh["gradient_sign"] == "source_sign"]
        axH.plot(
            one["initial_fraction"],
            one["bin_center_depth_m"],
            linestyle=":",
            label="Initial uniform",
        )
        for sign_name, label in [
            ("source_sign", "OpenDrift source sign"),
            ("reversed_sign", "Reversed gradient sign"),
        ]:
            h = hh[hh["gradient_sign"] == sign_name]
            axH.plot(
                h["final_fraction"],
                h["bin_center_depth_m"],
                label=label,
            )
        axH.invert_yaxis()
        axH.set_xlabel("Fraction of particles per depth bin")
        axH.set_ylabel("Depth (m)")
        axH.set_title(
            f"({chr(99+col)}) Well-mixed distribution after {TEST_HOURS:.0f} h",
            loc="left",
            fontweight="bold",
        )
        axH.grid(alpha=0.25)
        axH.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(FIGDIR / "vertical_mixing_diagnostic.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / "vertical_mixing_diagnostic.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGDIR / "vertical_mixing_diagnostic.svg", bbox_inches="tight")
    plt.close(fig)

    return summary


# =============================================================================
# WIND THRESHOLD / FIG. 3 OCCUPANCY CHECK
# =============================================================================

def load_python_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_script74():
    candidates = sorted((ROOT / "scripts").glob("74_check_temporal_resolution_fixed*.py"))
    if not candidates:
        raise FileNotFoundError(
            "Could not find scripts/74_check_temporal_resolution_fixed*.py"
        )
    return candidates[0]


def extract_release_point_wind_wave():
    reader_tools = load_python_module(find_script74(), "temporal_reader_tools")
    prod = reader_tools.load_script25()
    config = prod.load_config(prod.DEFAULT_CONFIG)
    helper04 = prod.load_script_module(
        "reader_check_r3",
        ROOT / "scripts" / "04_opendrift_reader_check.py",
    )
    mappings = helper04.STANDARD_NAME_MAPPINGS

    paths = {
        section: ROOT / "data" / "raw" / config["forcing"][section]["filename"]
        for section in mappings
    }

    wind = None
    wave = None

    for section, path in paths.items():
        with xr.open_dataset(path) as ds:
            ds, tname = reader_tools.subset_time(ds)
            site = reader_tools.nearest_site(ds)

            u10 = reader_tools.find_var(
                site,
                ["u10", "x_wind", "eastward_wind", "eastward_wind_at_10m"],
            )
            v10 = reader_tools.find_var(
                site,
                ["v10", "y_wind", "northward_wind", "northward_wind_at_10m"],
            )
            tp = reader_tools.find_var(
                site,
                [
                    "pp1d",
                    "sea_surface_wave_period_at_variance_spectral_density_maximum",
                    "sea_surface_wave_period_at_variance_spectral_density_maximum",
                ],
            )

            if u10 and v10 and wind is None:
                tw, u = reader_tools.surface_series(site, u10, tname)
                _, v = reader_tools.surface_series(site, v10, tname)
                wind = pd.DataFrame({
                    "time_utc": pd.to_datetime(tw),
                    "u10_m_s": u,
                    "v10_m_s": v,
                })

            if tp and wave is None:
                tt, period = reader_tools.surface_series(site, tp, tname)
                wave = pd.DataFrame({
                    "time_utc": pd.to_datetime(tt),
                    "peak_wave_period_s": period,
                })

    if wind is None:
        raise RuntimeError("Could not extract ERA5 u10/v10 at the release point.")

    wind["wind_speed_m_s"] = np.hypot(wind["u10_m_s"], wind["v10_m_s"])
    wind["elapsed_h"] = (
        (wind["time_utc"] - START) / pd.Timedelta(hours=1)
    ).astype(float)

    if wave is not None:
        wind = wind.merge(wave, on="time_utc", how="left")
    else:
        wind["peak_wave_period_s"] = np.nan

    # Implemented breaking-fraction behavior reconstructed in MECHANISM.
    with np.errstate(divide="ignore", invalid="ignore"):
        breaking = 0.032 * (wind["wind_speed_m_s"].to_numpy() - 5.0) / \
            wind["peak_wave_period_s"].to_numpy()

    breaking = np.where(np.isfinite(breaking), breaking, 0.0)
    breaking[breaking < 0] = 0.0

    wind["wind_ge_5_m_s"] = wind["wind_speed_m_s"] >= 5.0
    wind["breaking_fraction"] = breaking

    return wind


def lag_correlations(wind_occ: pd.DataFrame, occ_col: str, driver_col: str, maxlag=6):
    rows = []
    base = wind_occ[["elapsed_h", driver_col, occ_col]].dropna().copy()

    for lag in range(0, maxlag + 1):
        # driver(t) compared with occupancy(t+lag)
        shifted = base[["elapsed_h", occ_col]].copy()
        shifted["elapsed_h"] = shifted["elapsed_h"] - lag
        merged = base[["elapsed_h", driver_col]].merge(
            shifted, on="elapsed_h", how="inner"
        ).dropna()

        if len(merged) >= 3:
            corr = float(
                np.corrcoef(merged[driver_col], merged[occ_col])[0, 1]
            )
        else:
            corr = np.nan

        rows.append({
            "occupancy_column": occ_col,
            "driver": driver_col,
            "lag_h_driver_leads_occupancy": lag,
            "n": len(merged),
            "pearson_r": corr,
        })

    return rows


def wind_threshold_check():
    wind = extract_release_point_wind_wave()

    if not FIG3_CSV.exists():
        raise FileNotFoundError(
            f"Fig. 3 CSV not found: {FIG3_CSV}\n"
            "Run script 77 first."
        )

    fig3 = pd.read_csv(FIG3_CSV)

    needed = {
        "configuration",
        "time_h",
        "instantaneous_subsurface_occupancy",
    }
    missing = needed - set(fig3.columns)
    if missing:
        raise KeyError(f"Fig. 3 CSV is missing columns: {sorted(missing)}")

    c2 = fig3[
        fig3["configuration"].astype(str).str.startswith("C2")
    ][["time_h", "instantaneous_subsurface_occupancy"]].copy()
    c2.columns = ["elapsed_h", "C2_subsurface_occupancy"]

    c3 = fig3[
        fig3["configuration"].astype(str).str.startswith("C3")
    ][["time_h", "instantaneous_subsurface_occupancy"]].copy()
    c3.columns = ["elapsed_h", "C3_subsurface_occupancy"]

    out = wind.merge(c2, on="elapsed_h", how="left")
    out = out.merge(c3, on="elapsed_h", how="left")

    out.to_csv(DIAG / "wind_threshold_occupancy.csv", index=False)

    below = out["wind_speed_m_s"] < 5.0
    above = out["wind_speed_m_s"] >= 5.0

    summary_lines = [
        "VERTICAL MIXING WIND-THRESHOLD / FIG. 3 OCCUPANCY CHECK",
        "=" * 88,
        f"Hourly ERA5 samples: {len(out)}",
        f"Wind < 5 m/s: {int(below.sum())}",
        f"Wind >= 5 m/s: {int(above.sum())}",
        "",
    ]

    for col in ["C2_subsurface_occupancy", "C3_subsurface_occupancy"]:
        mean_below = float(out.loc[below, col].mean())
        mean_above = float(out.loc[above, col].mean())
        summary_lines.extend([
            col,
            f"  mean occupancy when wind <5 m/s : {mean_below:.5f}",
            f"  mean occupancy when wind >=5 m/s: {mean_above:.5f}",
            f"  difference (>=5 minus <5)       : {mean_above - mean_below:.5f}",
            "",
        ])

    corr_rows = []
    for occ_col in ["C2_subsurface_occupancy", "C3_subsurface_occupancy"]:
        corr_rows += lag_correlations(
            out, occ_col, "wind_speed_m_s", maxlag=6
        )
        corr_rows += lag_correlations(
            out, occ_col, "breaking_fraction", maxlag=6
        )

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(DIAG / "wind_occupancy_lag_correlations.csv", index=False)

    for occ_col in ["C2_subsurface_occupancy", "C3_subsurface_occupancy"]:
        sub = corr_df[corr_df["occupancy_column"] == occ_col]
        for driver in ["wind_speed_m_s", "breaking_fraction"]:
            d = sub[sub["driver"] == driver].dropna(subset=["pearson_r"])
            if len(d):
                best = d.iloc[np.argmax(np.abs(d["pearson_r"].to_numpy()))]
                summary_lines.append(
                    f"{occ_col} vs {driver}: strongest |r| over 0-6 h driver lead "
                    f"at lag {int(best['lag_h_driver_leads_occupancy'])} h, "
                    f"r={best['pearson_r']:.3f}, n={int(best['n'])}"
                )

    summary_lines.extend([
        "",
        "INTERPRETATION LIMIT",
        "-" * 88,
        "This is a descriptive temporal-alignment check. A threshold-aligned occupancy",
        "response would support the proposed interpretation, but it does not establish",
        "causality because wave height, wave period, buoyant resurfacing, and oil state",
        "also affect vertical exchange.",
    ])

    (DIAG / "wind_threshold_summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    # Plot wind threshold over occupancy.
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)

    axes[0].plot(out["elapsed_h"], out["wind_speed_m_s"], label="ERA5 10 m wind")
    axes[0].axhline(5.0, linestyle="--", linewidth=1.0, label="5 m/s threshold")
    axes[0].set_ylabel("Wind speed (m/s)")
    axes[0].set_title(
        "(a) Release-point ERA5 wind and breaking-wave threshold",
        loc="left",
        fontweight="bold",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(
        out["elapsed_h"],
        out["C2_subsurface_occupancy"],
        label="C2 Vertical exchange",
    )
    axes[1].plot(
        out["elapsed_h"],
        out["C3_subsurface_occupancy"],
        label="C3 Full fate",
    )
    axes[1].set_xlabel("Elapsed time (h)")
    axes[1].set_ylabel("Instantaneous subsurface occupancy")
    axes[1].set_title(
        "(b) Fig. 3 subsurface occupancy",
        loc="left",
        fontweight="bold",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(FIGDIR / "wind_threshold_occupancy.pdf", bbox_inches="tight")
    fig.savefig(
        FIGDIR / "wind_threshold_occupancy.png",
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(FIGDIR / "wind_threshold_occupancy.svg", bbox_inches="tight")
    plt.close(fig)

    return out, corr_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\nVERTICAL MIXING-A: checking installed OpenDrift source...")
    source_check = check_installed_source()
    print(
        "Installed OpenDrift:",
        source_check["opendrift_distribution_version"],
    )
    print(
        "Equivalent physical gradient term:",
        source_check["equivalent_physical_gradient_term"],
    )

    print("\nVERTICAL MIXING-B: running isolated well-mixed diagnostics...")
    summary = run_well_mixed_suite()
    print(summary.to_string(index=False))

    print("\nVERTICAL MIXING-C: extracting 49-hourly wind series and aligning with Fig. 3...")
    wind_occ, corr = wind_threshold_check()

    print("\nCreated:")
    for p in [
        DIAG / "vertical_mixing_source_check.txt",
        DIAG / "vertical_mixing_summary.csv",
        DIAG / "vertical_mixing_histograms.csv",
        DIAG / "vertical_mixing_diffusivity_profiles.csv",
        DIAG / "wind_threshold_occupancy.csv",
        DIAG / "wind_occupancy_lag_correlations.csv",
        DIAG / "wind_threshold_summary.txt",
        FIGDIR / "vertical_mixing_diagnostic.pdf",
        FIGDIR / "wind_threshold_occupancy.pdf",
    ]:
        print(f"  {p}")

    print("\nSTOP. Review these outputs before changing Fig. 3 or manuscript wording.")


if __name__ == "__main__":
    main()
