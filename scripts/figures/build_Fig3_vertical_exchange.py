#!/usr/bin/env python3
"""
build_Fig3_vertical_exchange.py

Build Figure 3 for the Jubilee OpenOil manuscript.

Creates:
    Fig3_vertical_timeseries.csv
    Fig3_vertical_depth_ensemble.csv
    Fig3_vertical_exchange.pdf
    Fig3_vertical_exchange.png
    Fig3_vertical_exchange.svg

Panels:
    (a) Cumulative post-release surface exposure, f_s(t)
    (b) Instantaneous subsurface occupancy
    (c) Mean depth of submerged particles
    (d) Ensemble upper-tail penetration depth (p95 and p99 with 95% CIs)

The script searches recursively from the project root for the baseline production
NetCDF files:
    C1_Surface_N1000_dt1800s_Kh100_seed20220901*.nc
    C2_Exchange_N1000_dt1800s_Kh100_seed20220901*.nc
    C3_Full_fate_N1000_dt1800s_Kh100_seed20220901*.nc

If diagnostics/validation/vertical_depth_summary.csv exists, panel (d) is
read directly from that file. Otherwise the verified vertical-depth ensemble values are
used as a fallback.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

# If the script is stored in PROJECT_ROOT/scripts/, use PROJECT_ROOT.
# Otherwise use the script folder itself.
if SCRIPT_DIR.name.lower() == "figures":
    ROOT = SCRIPT_DIR.parents[1]
elif SCRIPT_DIR.name.lower() == "scripts":
    ROOT = SCRIPT_DIR.parent
else:
    ROOT = SCRIPT_DIR

OUTDIR = ROOT / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

CSV_TS_OUT = OUTDIR / "Fig3_vertical_timeseries.csv"
CSV_ENSEMBLE_OUT = OUTDIR / "Fig3_vertical_depth_ensemble.csv"

FIG_PDF_OUT = OUTDIR / "Fig3_vertical_exchange.pdf"
FIG_PNG_OUT = OUTDIR / "Fig3_vertical_exchange.png"
FIG_SVG_OUT = OUTDIR / "Fig3_vertical_exchange.svg"

vertical-depth_SUMMARY = ROOT / "diagnostics" / "validation" / "vertical_depth_summary.csv"

RELEASE_END_H = 6.0
SURFACE_Z_TOL_M = 1.0e-6


# ---------------------------------------------------------------------
# FILE DISCOVERY
# ---------------------------------------------------------------------

def find_one(pattern: str) -> Path:
    """Find exactly one suitable file recursively under ROOT."""
    matches = sorted(ROOT.rglob(pattern))
    matches = [p for p in matches if p.is_file()]

    if not matches:
        raise FileNotFoundError(
            f"\nCould not find a file matching:\n  {pattern}\n"
            f"under project root:\n  {ROOT}\n"
        )

    # Prefer shortest path / most direct production file if duplicates exist.
    matches.sort(key=lambda p: (len(p.parts), str(p)))

    if len(matches) > 1:
        print(f"\nMultiple matches found for {pattern}. Using:")
        print(f"  {matches[0]}")
        print("Other matches:")
        for p in matches[1:]:
            print(f"  {p}")

    return matches[0]


FILE_C1 = find_one("C1_Surface_N1000_dt1800s_Kh100_seed20220901*.nc")
FILE_C2 = find_one("C2_Exchange_N1000_dt1800s_Kh100_seed20220901*.nc")
FILE_C3 = find_one("C3_Full_fate_N1000_dt1800s_Kh100_seed20220901*.nc")


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def orient_trajectory_time(da: xr.DataArray) -> np.ndarray:
    """
    Return an array shaped (trajectory, time).
    """
    arr = np.asarray(da.values, dtype=float)
    dims = list(da.dims)

    if "trajectory" not in dims or "time" not in dims:
        raise ValueError(
            f"{da.name!r} must have trajectory and time dimensions; got {dims}"
        )

    arr = np.moveaxis(
        arr,
        [dims.index("trajectory"), dims.index("time")],
        [0, 1],
    )
    return arr


def get_time_hours(ds: xr.Dataset) -> np.ndarray:
    """
    Return elapsed hours from the first model output.
    """
    if "time" not in ds:
        raise KeyError("Dataset has no 'time' coordinate.")

    t = pd.to_datetime(ds["time"].values)
    return np.asarray((t - t[0]) / np.timedelta64(1, "h"), dtype=float)


def load_z_time(path: Path):
    """
    Read z and time from one OpenDrift/OpenOil NetCDF.
    Returns time_hours and z shaped (trajectory, time).
    """
    with xr.open_dataset(path) as ds:
        if "z" not in ds:
            raise KeyError(f"'z' not found in {path}")

        z = orient_trajectory_time(ds["z"])
        time_h = get_time_hours(ds)

    if z.shape[1] != len(time_h):
        raise ValueError(
            f"Time mismatch for {path.name}: z shape={z.shape}, "
            f"time points={len(time_h)}"
        )

    return time_h, z


def compute_timeseries(time_h: np.ndarray, z: np.ndarray, configuration: str):
    """
    Calculate the three time-dependent diagnostics used in panels (a)-(c).

    Definitions
    -----------
    cumulative_surface_exposure:
        cumulative fraction of finite particle-time records at the surface from
        release completion (6 h) through the current time.

    instantaneous_subsurface_occupancy:
        fraction of finite particles below the surface at each output time.

    mean_submerged_depth_m:
        mean positive depth of particles below the surface at each output time.
    """
    finite = np.isfinite(z)
    at_surface = finite & (z >= -SURFACE_Z_TOL_M)
    submerged = finite & (z < -SURFACE_Z_TOL_M)

    nt = z.shape[1]

    cumulative_fs = np.full(nt, np.nan, dtype=float)
    subsurface_fraction = np.full(nt, np.nan, dtype=float)
    mean_submerged_depth = np.full(nt, np.nan, dtype=float)

    # Instantaneous quantities
    for j in range(nt):
        valid_j = finite[:, j]
        n_valid = int(valid_j.sum())

        if n_valid > 0:
            subsurface_fraction[j] = submerged[:, j].sum() / n_valid

        depths_j = -z[:, j]
        use = submerged[:, j] & np.isfinite(depths_j)
        if np.any(use):
            mean_submerged_depth[j] = float(np.mean(depths_j[use]))

    # Cumulative post-release surface exposure
    post = np.where(time_h >= RELEASE_END_H)[0]
    if len(post):
        j0 = int(post[0])

        for j in range(j0, nt):
            valid_window = finite[:, j0:j+1]
            surf_window = at_surface[:, j0:j+1]

            denom = int(valid_window.sum())
            numer = int((surf_window & valid_window).sum())

            if denom > 0:
                cumulative_fs[j] = numer / denom

    return pd.DataFrame({
        "configuration": configuration,
        "time_h": time_h,
        "cumulative_post_release_surface_exposure": cumulative_fs,
        "instantaneous_subsurface_occupancy": subsurface_fraction,
        "mean_submerged_depth_m": mean_submerged_depth,
    })


def load_m9_summary() -> pd.DataFrame:
    """
    Read the authoritative vertical-depth ensemble summary when available.

    Required metrics:
        trajectory_max_depth_p95_m
        trajectory_max_depth_p99_m

    Falls back to the verified manuscript-analysis values only if the vertical-depth summary
    CSV is not present.
    """
    wanted = {
        "trajectory_max_depth_p95_m": "p95",
        "trajectory_max_depth_p99_m": "p99",
    }

    if vertical-depth_SUMMARY.exists():
        df = pd.read_csv(vertical-depth_SUMMARY)

        required = {
            "scenario", "metric", "mean", "ci95_low", "ci95_high"
        }
        missing = required - set(df.columns)
        if missing:
            raise KeyError(
                f"vertical-depth summary exists but is missing columns: {sorted(missing)}"
            )

        df = df[df["metric"].isin(wanted)].copy()
        df["statistic"] = df["metric"].map(wanted)

        out = df[[
            "scenario",
            "statistic",
            "mean",
            "ci95_low",
            "ci95_high"
        ]].copy()

        out.columns = [
            "configuration",
            "statistic",
            "mean_m",
            "ci95_low_m",
            "ci95_high_m",
        ]

        out["configuration"] = out["configuration"].astype(str)

        print(f"\nPanel (d) data loaded from:\n  {vertical-depth_SUMMARY}")
        return out.sort_values(["configuration", "statistic"]).reset_index(drop=True)

    print(
        "\nWARNING: vertical-depth summary CSV not found. "
        "Using verified vertical-depth ensemble values embedded in this script."
    )

    return pd.DataFrame({
        "configuration": ["C2", "C2", "C3", "C3"],
        "statistic": ["p95", "p99", "p95", "p99"],
        "mean_m": [
            11.0310,
            11.4709,
            10.9739,
            11.6106,
        ],
        "ci95_low_m": [
            11.0142,
            11.4425,
            10.9379,
            11.5836,
        ],
        "ci95_high_m": [
            11.0477,
            11.4992,
            11.0099,
            11.6375,
        ],
    })


def add_release_window(ax):
    ax.axvspan(0, RELEASE_END_H, alpha=0.12)
    ax.axvline(RELEASE_END_H, linestyle="--", linewidth=1.0)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print("\nBaseline files")
    print("-" * 78)
    print(f"C1: {FILE_C1}")
    print(f"C2: {FILE_C2}")
    print(f"C3: {FILE_C3}")

    # Load baseline reference-realization outputs.
    t1, z1 = load_z_time(FILE_C1)
    t2, z2 = load_z_time(FILE_C2)
    t3, z3 = load_z_time(FILE_C3)

    # Time-series data for panels a-c.
    d1 = compute_timeseries(t1, z1, "C1 Surface drift")
    d2 = compute_timeseries(t2, z2, "C2 Vertical exchange")
    d3 = compute_timeseries(t3, z3, "C3 Full fate")

    ts = pd.concat([d1, d2, d3], ignore_index=True)
    ts.to_csv(CSV_TS_OUT, index=False)

    # Ensemble data for panel d.
    depth = load_m9_summary()
    depth.to_csv(CSV_ENSEMBLE_OUT, index=False)

    # -----------------------------
    # Plot
    # -----------------------------
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axa, axb, axc, axd = axes.ravel()

    # Panel a
    for df, label in [
        (d1, "C1 Surface drift"),
        (d2, "C2 Vertical exchange"),
        (d3, "C3 Full fate"),
    ]:
        axa.plot(
            df["time_h"],
            df["cumulative_post_release_surface_exposure"],
            linewidth=1.6,
            label=label,
        )

    add_release_window(axa)
    axa.set_xlim(0, 48)
    axa.set_ylim(0.70, 1.02)
    axa.set_xlabel("Elapsed time (h)")
    axa.set_ylabel(r"$f_s(t)$")
    axa.set_title(
        "(a) Cumulative post-release surface exposure",
        loc="left",
        fontweight="bold",
    )
    axa.grid(alpha=0.25)
    axa.legend(frameon=False)

    # Panel b
    for df, label in [
        (d2, "C2 Vertical exchange"),
        (d3, "C3 Full fate"),
    ]:
        axb.plot(
            df["time_h"],
            df["instantaneous_subsurface_occupancy"],
            linewidth=1.6,
            label=label,
        )

    add_release_window(axb)
    axb.set_xlim(0, 48)
    axb.set_ylim(bottom=0)
    axb.set_xlabel("Elapsed time (h)")
    axb.set_ylabel("Fraction of finite particles below surface")
    axb.set_title(
        "(b) Instantaneous subsurface occupancy",
        loc="left",
        fontweight="bold",
    )
    axb.grid(alpha=0.25)
    axb.legend(frameon=False)

    # Panel c
    for df, label in [
        (d2, "C2 Vertical exchange"),
        (d3, "C3 Full fate"),
    ]:
        axc.plot(
            df["time_h"],
            df["mean_submerged_depth_m"],
            linewidth=1.6,
            label=label,
        )

    add_release_window(axc)
    axc.set_xlim(0, 48)
    axc.set_ylim(bottom=0)
    axc.set_xlabel("Elapsed time (h)")
    axc.set_ylabel("Mean depth of submerged particles (m)")
    axc.set_title(
        "(c) Mean depth of submerged particles",
        loc="left",
        fontweight="bold",
    )
    axc.grid(alpha=0.25)
    axc.legend(frameon=False)

    # Panel d
    configurations = ["C2", "C3"]
    x = np.arange(len(configurations), dtype=float)
    offset = 0.11

    for statistic, marker, dx in [
        ("p95", "o", -offset),
        ("p99", "s", +offset),
    ]:
        sub = (
            depth[depth["statistic"] == statistic]
            .set_index("configuration")
            .loc[configurations]
            .reset_index()
        )

        means = sub["mean_m"].to_numpy(dtype=float)
        lows = sub["ci95_low_m"].to_numpy(dtype=float)
        highs = sub["ci95_high_m"].to_numpy(dtype=float)

        yerr = np.vstack([
            means - lows,
            highs - means,
        ])

        axd.errorbar(
            x + dx,
            means,
            yerr=yerr,
            fmt=marker,
            markersize=6,
            linewidth=1.5,
            capsize=4,
            label=rf"${statistic[1:]}_{{{statistic[1:]}}}$"
            if False else rf"$p_{{{statistic[1:]}}}$",
        )

        for xx, yy in zip(x + dx, means):
            axd.text(
                xx,
                yy + 0.035,
                f"{yy:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axd.set_xticks(x)
    axd.set_xticklabels(["C2 Vertical exchange", "C3 Full fate"])
    axd.set_ylabel("Maximum penetration depth (m)")
    axd.set_title(
        "(d) Ensemble upper-tail penetration depth",
        loc="left",
        fontweight="bold",
    )
    axd.grid(axis="y", alpha=0.25)
    axd.legend(frameon=False)

    # Figure-level note
    fig.text(
        0.5,
        0.012,
        "Panels (a-c): reference production realization, seed 20220901. "
        "Panel (d): 20-seed ensemble mean with 95% confidence intervals. "
        "Shading marks the 0-6 h continuous release; dashed line marks release completion.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )

    fig.tight_layout(rect=[0, 0.045, 1, 1])

    fig.savefig(FIG_PDF_OUT, bbox_inches="tight")
    fig.savefig(FIG_PNG_OUT, dpi=600, bbox_inches="tight")
    fig.savefig(FIG_SVG_OUT, bbox_inches="tight")

    print("\nCreated")
    print("-" * 78)
    print(CSV_TS_OUT)
    print(CSV_ENSEMBLE_OUT)
    print(FIG_PDF_OUT)
    print(FIG_PNG_OUT)
    print(FIG_SVG_OUT)

    plt.show()


if __name__ == "__main__":
    main()
