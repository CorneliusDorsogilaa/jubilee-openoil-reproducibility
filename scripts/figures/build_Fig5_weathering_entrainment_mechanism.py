#!/usr/bin/env python3
"""
build_Fig5_weathering_entrainment_mechanism.py

Rebuild the manuscript Fig. 5 mechanism figure from the authoritative
Internal validation mechanism reconstruction outputs.

Scientific chain
----------------
C3 emulsification / weathering
    -> higher oil-emulsion viscosity
    -> higher Ohnesorge number
    -> lower Li et al. wave-entrainment rate Q
    -> contributes to higher surface exposure in C3

This figure is intentionally limited to the mechanism variables. Surface
occupancy itself is already shown in Fig. 3 and is therefore not duplicated.

Inputs
------
diagnostics/validation/weathering_entrainment_mechanism_timeseries.csv

Outputs
-------
figures/Fig5_weathering_entrainment_mechanism.csv
figures/Fig5_weathering_entrainment_mechanism.pdf
figures/Fig5_weathering_entrainment_mechanism.png
figures/Fig5_weathering_entrainment_mechanism.svg
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1] if SCRIPT_DIR.name.lower() == "figures" else (SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "scripts" else SCRIPT_DIR)

INDIR = ROOT / "diagnostics" / "validation"
OUTDIR = ROOT / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

SRC_CSV = INDIR / "weathering_entrainment_mechanism_timeseries.csv"

CSV_OUT = OUTDIR / "Fig5_weathering_entrainment_mechanism.csv"
PDF_OUT = OUTDIR / "Fig5_weathering_entrainment_mechanism.pdf"
PNG_OUT = OUTDIR / "Fig5_weathering_entrainment_mechanism.png"
SVG_OUT = OUTDIR / "Fig5_weathering_entrainment_mechanism.svg"


def require(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required input not found:\n  {path}\n\n"
            "Run the mechanism reconstruction first (script 62) in the Jubilee OpenDrift environment."
        )


def post_release_mean(df: pd.DataFrame, config: str, col: str) -> float:
    d = df[(df["scenario"] == config) & (df["hour"] >= 6.0)]
    return float(np.nanmean(d[col].to_numpy(dtype=float)))


def ratio(df: pd.DataFrame, col: str) -> float:
    c2 = post_release_mean(df, "C2", col)
    c3 = post_release_mean(df, "C3", col)
    return c3 / c2


def main():
    require(SRC_CSV)

    df = pd.read_csv(SRC_CSV)

    required = [
        "scenario",
        "hour",
        "mean_water_fraction",
        "mean_kinematic_viscosity_cSt",
        "mean_ohnesorge",
        "mean_entrainment_rate_s-1",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"mechanism CSV is missing required columns: {missing}")

    # Export a clean figure-source CSV containing exactly the plotted variables.
    plot_df = df[
        [
            "scenario",
            "hour",
            "mean_water_fraction",
            "mean_kinematic_viscosity_cSt",
            "mean_ohnesorge",
            "mean_entrainment_rate_s-1",
        ]
    ].copy()

    plot_df.to_csv(CSV_OUT, index=False)

    # Recalculate headline values directly from the source CSV so the figure
    # does not depend on hard-coded ratios.
    visc_ratio = ratio(df, "mean_kinematic_viscosity_cSt")
    oh_ratio = ratio(df, "mean_ohnesorge")
    q_ratio = ratio(df, "mean_entrainment_rate_s-1")
    q_reduction = 100.0 * (1.0 - q_ratio)

    c2_visc = post_release_mean(df, "C2", "mean_kinematic_viscosity_cSt")
    c3_visc = post_release_mean(df, "C3", "mean_kinematic_viscosity_cSt")
    c2_oh = post_release_mean(df, "C2", "mean_ohnesorge")
    c3_oh = post_release_mean(df, "C3", "mean_ohnesorge")
    c2_q = post_release_mean(df, "C2", "mean_entrainment_rate_s-1")
    c3_q = post_release_mean(df, "C3", "mean_entrainment_rate_s-1")
    c3_water = post_release_mean(df, "C3", "mean_water_fraction")

    print("\nPOST-RELEASE MEANS, 6-48 h")
    print("-" * 72)
    print(f"C3 mean water fraction       : {c3_water:.6g}")
    print(f"C2 kinematic viscosity       : {c2_visc:.6g} cSt")
    print(f"C3 kinematic viscosity       : {c3_visc:.6g} cSt")
    print(f"C3/C2 viscosity ratio        : {visc_ratio:.6g} x")
    print(f"C2 Ohnesorge                 : {c2_oh:.6g}")
    print(f"C3 Ohnesorge                 : {c3_oh:.6g}")
    print(f"C3/C2 Ohnesorge ratio        : {oh_ratio:.6g} x")
    print(f"C2 entrainment rate          : {c2_q:.6g} s^-1")
    print(f"C3 entrainment rate          : {c3_q:.6g} s^-1")
    print(f"C3/C2 entrainment-rate ratio : {q_ratio:.6g}")
    print(f"Reduction in C3              : {q_reduction:.3f}%")

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()

    panels = [
        (
            "mean_water_fraction",
            "Mean particle water fraction",
            "(a) Emulsification state",
            "linear",
        ),
        (
            "mean_kinematic_viscosity_cSt",
            "Kinematic viscosity (cSt)",
            "(b) Oil-emulsion viscosity",
            "log",
        ),
        (
            "mean_ohnesorge",
            "Ohnesorge number",
            "(c) Viscous resistance to breakup",
            "log",
        ),
        (
            "mean_entrainment_rate_s-1",
            r"Entrainment rate, $Q$ (s$^{-1}$)",
            "(d) Li et al. wave-entrainment rate",
            "linear",
        ),
    ]

    for ax, (col, ylabel, title, yscale) in zip(axes, panels):
        for config, label, ls in [
            ("C2", "C2 Vertical exchange", "-"),
            ("C3", "C3 Full fate", "--"),
        ]:
            d = plot_df[plot_df["scenario"] == config]
            x = d["hour"].to_numpy(dtype=float)
            y = d[col].to_numpy(dtype=float)

            if yscale == "log":
                mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
                ax.plot(x[mask], y[mask], linestyle=ls, linewidth=1.8, label=label)
            else:
                ax.plot(x, y, linestyle=ls, linewidth=1.8, label=label)

        ax.axvspan(0, 6, alpha=0.10)
        ax.axvline(6, linestyle=":", linewidth=1.1)
        ax.set_xlim(0, 48)
        ax.set_xlabel("Elapsed time (h)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(alpha=0.25)

        if yscale == "log":
            ax.set_yscale("log")

    axes[0].legend(frameon=False)

    # Panel-specific concise annotations.
    axes[1].text(
        0.04, 0.95,
        f"6-48 h mean: C3/C2 = {visc_ratio:.0f}x",
        transform=axes[1].transAxes,
        ha="left", va="top", fontsize=9,
    )
    axes[2].text(
        0.04, 0.95,
        f"6-48 h mean: C3/C2 = {oh_ratio:.0f}x",
        transform=axes[2].transAxes,
        ha="left", va="top", fontsize=9,
    )
    axes[3].text(
        0.96, 0.95,
        f"6-48 h mean: {q_reduction:.1f}% lower in C3",
        transform=axes[3].transAxes,
        ha="right", va="top", fontsize=9,
    )

    fig.text(
        0.5, 0.012,
        "Shading marks the 0-6 h continuous release; dotted line marks release completion. "
        "C2 and C3 share the same forcing and vertical-exchange formulation; C3 additionally "
        "includes evaporation, emulsification and natural dispersion.",
        ha="center", va="bottom", fontsize=8.5,
    )

    fig.tight_layout(rect=[0, 0.045, 1, 1])

    fig.savefig(PDF_OUT, bbox_inches="tight")
    fig.savefig(PNG_OUT, dpi=600, bbox_inches="tight")
    fig.savefig(SVG_OUT, bbox_inches="tight")

    print("\nCreated")
    print("-" * 72)
    print(CSV_OUT)
    print(PDF_OUT)
    print(PNG_OUT)
    print(SVG_OUT)

    plt.show()


if __name__ == "__main__":
    main()
