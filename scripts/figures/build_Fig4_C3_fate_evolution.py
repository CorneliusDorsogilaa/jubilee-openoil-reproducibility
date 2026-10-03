#!/usr/bin/env python3
"""
build_Fig4_C3_fate_evolution.py

Rebuild Figure 4 exactly as described in the main manuscript:

(a) Represented mass components
(b) Fate partition relative to release-complete mass
(c) Emulsification expressed as mean particle water fraction
(d) Post-release represented-mass closure

The script exports the complete plotting dataset to CSV before creating the figure.

New outputs
-----------
figures/Fig4_C3_fate_timeseries.csv
figures/Fig4_C3_fate_evolution.pdf
figures/Fig4_C3_fate_evolution.png
figures/Fig4_C3_fate_evolution.svg

Important interpretation
------------------------
Panel (d) is an internal represented-mass accounting consistency check.
It should not be described as independent physical validation.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# PATHS AND SETTINGS
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

# Intended location: PROJECT_ROOT/scripts/
if SCRIPT_DIR.name.lower() == "figures":
    ROOT = SCRIPT_DIR.parents[1]
elif SCRIPT_DIR.name.lower() == "scripts":
    ROOT = SCRIPT_DIR.parent
else:
    ROOT = SCRIPT_DIR

OUTDIR = ROOT / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

CSV_OUT = OUTDIR / "Fig4_C3_fate_timeseries.csv"
PDF_OUT = OUTDIR / "Fig4_C3_fate_evolution.pdf"
PNG_OUT = OUTDIR / "Fig4_C3_fate_evolution.png"
SVG_OUT = OUTDIR / "Fig4_C3_fate_evolution.svg"

RELEASE_END_H = 6.0

# Manuscript endpoint values, used only as a consistency check.
EXPECTED_48H = {
    "remaining": 0.418,
    "evaporated": 0.438,
    "dispersed": 0.144,
    "water_fraction": 0.896,
}


# ---------------------------------------------------------------------
# FILE DISCOVERY
# ---------------------------------------------------------------------

def find_c3_production_file() -> Path:
    """
    Find the reference C3 production realization.

    Preference order:
    1. outputs/production/
    2. any other project location, excluding ensemble folders
    """
    pattern = "C3_Full_fate_N1000_dt1800s_Kh100_seed20220901*.nc"

    preferred = sorted((ROOT / "outputs" / "production").glob(pattern))
    preferred = [p for p in preferred if p.is_file()]
    if preferred:
        return preferred[-1]

    matches = sorted(ROOT.rglob(pattern))
    matches = [
        p for p in matches
        if p.is_file()
        and "seed_ensemble" not in str(p)
        and "particle_convergence" not in str(p)
    ]

    if not matches:
        raise FileNotFoundError(
            "\nCould not find the reference C3 production NetCDF matching:\n"
            f"  {pattern}\nunder:\n  {ROOT}\n"
        )

    if len(matches) > 1:
        print("\nMultiple candidate C3 files found. Using:")
        print(f"  {matches[-1]}")
        print("Other candidates:")
        for p in matches[:-1]:
            print(f"  {p}")

    return matches[-1]


C3_FILE = find_c3_production_file()


# ---------------------------------------------------------------------
# DATA HELPERS
# ---------------------------------------------------------------------

def time_hours(ds: xr.Dataset) -> np.ndarray:
    """Elapsed hours from the first model output."""
    if "time" not in ds:
        raise KeyError("Dataset does not contain a 'time' coordinate.")

    t = pd.to_datetime(ds["time"].values)
    return np.asarray((t - t[0]) / np.timedelta64(1, "h"), dtype=float)


def sum_by_time(ds: xr.Dataset, variable: str, nt: int) -> np.ndarray:
    """
    Sum a trajectory-dependent variable at each model output time.
    Returns NaNs if the variable is absent.
    """
    if variable not in ds:
        return np.full(nt, np.nan, dtype=float)

    da = ds[variable]
    arr = np.asarray(da.values, dtype=float)
    dims = list(da.dims)

    if "time" not in dims:
        # Static variable. Not expected for mass reservoirs.
        return np.full(nt, float(np.nansum(arr)), dtype=float)

    time_axis = dims.index("time")
    arr = np.moveaxis(arr, time_axis, -1)

    # Sum every axis except time.
    if arr.ndim == 1:
        out = arr
    else:
        axes = tuple(range(arr.ndim - 1))
        out = np.nansum(arr, axis=axes)

    out = np.asarray(out, dtype=float).reshape(-1)

    if len(out) != nt:
        raise ValueError(
            f"{variable}: expected {nt} time values, obtained {len(out)}."
        )

    return out


def mean_by_time(ds: xr.Dataset, variable: str, nt: int) -> np.ndarray:
    """
    Mean of a trajectory-dependent variable at each model output time.
    """
    if variable not in ds:
        return np.full(nt, np.nan, dtype=float)

    da = ds[variable]
    arr = np.asarray(da.values, dtype=float)
    dims = list(da.dims)

    if "time" not in dims:
        return np.full(nt, float(np.nanmean(arr)), dtype=float)

    time_axis = dims.index("time")
    arr = np.moveaxis(arr, time_axis, -1)

    if arr.ndim == 1:
        out = arr
    else:
        axes = tuple(range(arr.ndim - 1))
        out = np.nanmean(arr, axis=axes)

    out = np.asarray(out, dtype=float).reshape(-1)

    if len(out) != nt:
        raise ValueError(
            f"{variable}: expected {nt} time values, obtained {len(out)}."
        )

    return out


def nearest_index(x: np.ndarray, target: float) -> int:
    return int(np.nanargmin(np.abs(np.asarray(x, dtype=float) - target)))


def add_release_window(ax):
    ax.axvspan(0, RELEASE_END_H, alpha=0.12)
    ax.axvline(RELEASE_END_H, linestyle="--", linewidth=1.0)


# ---------------------------------------------------------------------
# BUILD DATA TABLE
# ---------------------------------------------------------------------

def extract_fate_timeseries(path: Path) -> pd.DataFrame:
    with xr.open_dataset(path) as ds:
        th = time_hours(ds)
        nt = len(th)

        mass_oil = sum_by_time(ds, "mass_oil", nt)
        mass_evap = sum_by_time(ds, "mass_evaporated", nt)
        mass_disp = sum_by_time(ds, "mass_dispersed", nt)
        mass_bio = sum_by_time(ds, "mass_biodegraded", nt)

        # Biodegradation is not part of the reported C3 fate partition in the
        # manuscript. Include it in accounting only if it actually exists.
        has_bio = np.any(np.isfinite(mass_bio))

        total_represented = (
            np.nan_to_num(mass_oil, nan=0.0)
            + np.nan_to_num(mass_evap, nan=0.0)
            + np.nan_to_num(mass_disp, nan=0.0)
        )

        if has_bio:
            total_represented = total_represented + np.nan_to_num(
                mass_bio, nan=0.0
            )

        water_fraction = mean_by_time(ds, "water_fraction", nt)

    # Release-complete denominator at the output nearest 6 h.
    i6 = nearest_index(th, RELEASE_END_H)
    mass_release_complete = float(total_represented[i6])

    if not np.isfinite(mass_release_complete) or mass_release_complete <= 0:
        raise RuntimeError(
            f"Invalid release-complete represented mass at {th[i6]:.2f} h: "
            f"{mass_release_complete}"
        )

    frac_oil = mass_oil / mass_release_complete
    frac_evap = mass_evap / mass_release_complete
    frac_disp = mass_disp / mass_release_complete

    # Post-release accounting identity.
    closure = total_represented / mass_release_complete
    closure[th < RELEASE_END_H] = np.nan

    df = pd.DataFrame({
        "time_h": th,
        "mass_oil_model_units": mass_oil,
        "mass_evaporated_model_units": mass_evap,
        "mass_dispersed_model_units": mass_disp,
        "mass_biodegraded_model_units": mass_bio,
        "total_represented_mass_model_units": total_represented,
        "release_complete_mass_model_units": mass_release_complete,
        "fraction_remaining_oil": frac_oil,
        "fraction_evaporated": frac_evap,
        "fraction_dispersed": frac_disp,
        "mean_particle_water_fraction": water_fraction,
        "post_release_mass_closure": closure,
    })

    return df


# ---------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------

def main():
    print("\nReference C3 production file")
    print("-" * 80)
    print(C3_FILE)

    df = extract_fate_timeseries(C3_FILE)
    df.to_csv(CSV_OUT, index=False)

    final = df.iloc[-1]

    f_oil = float(final["fraction_remaining_oil"])
    f_evap = float(final["fraction_evaporated"])
    f_disp = float(final["fraction_dispersed"])
    wf = float(final["mean_particle_water_fraction"])
    closure48 = float(final["post_release_mass_closure"])

    print("\n48 h values extracted from the NetCDF")
    print("-" * 80)
    print(f"Remaining oil fraction : {f_oil:.6f} ({100*f_oil:.1f}%)")
    print(f"Evaporated fraction    : {f_evap:.6f} ({100*f_evap:.1f}%)")
    print(f"Dispersed fraction     : {f_disp:.6f} ({100*f_disp:.1f}%)")
    print(f"Mean water fraction    : {wf:.6f}")
    print(f"Mass-accounting closure: {closure48:.6f}")

    # Warn if the selected file does not reproduce the manuscript endpoint.
    checks = {
        "remaining": f_oil,
        "evaporated": f_evap,
        "dispersed": f_disp,
        "water_fraction": wf,
    }

    for key, value in checks.items():
        expected = EXPECTED_48H[key]
        if abs(value - expected) > 0.01:
            print(
                f"WARNING: {key}={value:.4f} differs from manuscript "
                f"reference {expected:.4f} by >0.01."
            )

    # Plot settings
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

    t = df["time_h"].to_numpy()

    # ---------------- Panel (a): represented mass components ----------------
    axa.plot(t, df["mass_oil_model_units"], label="Remaining oil", linewidth=1.7)
    axa.plot(t, df["mass_evaporated_model_units"], label="Evaporated", linewidth=1.7)
    axa.plot(t, df["mass_dispersed_model_units"], label="Dispersed", linewidth=1.7)

    add_release_window(axa)
    axa.set_xlim(0, 48)
    axa.set_ylim(bottom=0)
    axa.set_xlabel("Elapsed time (h)")
    axa.set_ylabel("Represented mass (model units)")
    axa.set_title(
        "(a) C3 represented mass components",
        loc="left",
        fontweight="bold",
    )
    axa.grid(alpha=0.25)
    axa.legend(frameon=False)

    # ---------------- Panel (b): normalized fate partition ----------------
    axb.plot(t, df["fraction_remaining_oil"], label="Remaining oil", linewidth=1.7)
    axb.plot(t, df["fraction_evaporated"], label="Evaporated", linewidth=1.7)
    axb.plot(t, df["fraction_dispersed"], label="Dispersed", linewidth=1.7)

    add_release_window(axb)
    axb.set_xlim(0, 48)
    axb.set_ylim(bottom=0)
    axb.set_xlabel("Elapsed time (h)")
    axb.set_ylabel("Fraction of release-complete mass")
    axb.set_title(
        "(b) Fate partition relative to release-complete mass",
        loc="left",
        fontweight="bold",
    )
    axb.grid(alpha=0.25)

    axb.text(
        0.97, 0.96,
        f"48 h: remaining {100*f_oil:.1f}%\n"
        f"evaporated {100*f_evap:.1f}%\n"
        f"dispersed {100*f_disp:.1f}%",
        transform=axb.transAxes,
        ha="right",
        va="top",
        fontsize=8.7,
    )

    # ---------------- Panel (c): emulsification ----------------
    axc.plot(
        t,
        df["mean_particle_water_fraction"],
        linewidth=1.7,
        label="Mean particle water fraction",
    )

    add_release_window(axc)
    axc.set_xlim(0, 48)
    axc.set_ylim(0, 1.0)
    axc.set_xlabel("Elapsed time (h)")
    axc.set_ylabel("Mean particle water fraction")
    axc.set_title(
        "(c) Emulsification evolution",
        loc="left",
        fontweight="bold",
    )
    axc.grid(alpha=0.25)

    axc.text(
        0.97, 0.08,
        f"48 h = {wf:.3f}",
        transform=axc.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )

    # ---------------- Panel (d): represented-mass accounting ----------------
    post = df["time_h"] >= RELEASE_END_H
    tp = df.loc[post, "time_h"].to_numpy()
    cp = df.loc[post, "post_release_mass_closure"].to_numpy(dtype=float)

    axd.plot(tp, cp, linewidth=1.7)
    axd.axhline(1.0, linestyle="--", linewidth=1.0)

    finite_cp = cp[np.isfinite(cp)]
    if finite_cp.size:
        cmin = float(np.min(finite_cp))
        cmax = float(np.max(finite_cp))
        spread = max(cmax - cmin, 2e-4)
        pad = max(0.25 * spread, 2e-4)
        axd.set_ylim(min(1.0, cmin) - pad, max(1.0, cmax) + pad)

    axd.set_xlim(RELEASE_END_H, 48)
    axd.set_xlabel("Elapsed time (h)")
    axd.set_ylabel("Total represented mass / mass at 6 h")
    axd.set_title(
        "(d) Post-release represented-mass closure",
        loc="left",
        fontweight="bold",
    )
    axd.grid(alpha=0.25)

    axd.text(
        0.97, 0.08,
        f"48 h = {closure48:.6f}",
        transform=axd.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )

    # Figure note
    fig.text(
        0.5,
        0.012,
        "Panels (a-c): shaded interval marks the 0-6 h continuous release; "
        "dashed line marks release completion. Panel (d) is an internal "
        "represented-mass accounting check.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    fig.savefig(PDF_OUT, bbox_inches="tight")
    fig.savefig(PNG_OUT, dpi=600, bbox_inches="tight")
    fig.savefig(SVG_OUT, bbox_inches="tight")

    print("\nCreated")
    print("-" * 80)
    print(CSV_OUT)
    print(PDF_OUT)
    print(PNG_OUT)
    print(SVG_OUT)

    plt.show()


if __name__ == "__main__":
    main()
