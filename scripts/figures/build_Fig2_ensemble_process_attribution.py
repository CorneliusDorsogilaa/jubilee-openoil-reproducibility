#!/usr/bin/env python3
"""Rebuild manuscript Fig. 2 from archived source CSVs only."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "derived_data"
OUT = ROOT / "figures_rebuilt"
OUT.mkdir(exist_ok=True)

ABS = DATA / "Figure2_ensemble_absolute_metrics.csv"
PAIR = DATA / "Figure2_paired_A90_changes.csv"


def main():
    a = pd.read_csv(ABS)
    p = pd.read_csv(PAIR)
    labels = ["C0\nHydro", "C1\nSurface", "C2\nExchange", "C3\nFull fate"]
    x = np.arange(4)

    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.95], hspace=0.22, wspace=0.22)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    bars = ax1.bar(x, a["A90_mean_km2"])
    ax1.set_xticks(x, labels)
    ax1.set_ylabel(r"Ensemble mean $A_{90}$ (km$^2$)")
    ax1.set_title("(a) Central 90% footprint", loc="left", fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.25)
    for b, v in zip(bars, a["A90_mean_km2"]):
        ax1.text(b.get_x()+b.get_width()/2, v+12, f"{v:.1f}", ha="center")

    xp = np.arange(len(p))
    y = p["delta_A90_mean_km2"].to_numpy(float)
    lo = y - p["CI95_low_km2"].to_numpy(float)
    hi = p["CI95_high_km2"].to_numpy(float) - y
    ax2.errorbar(xp, y, yerr=np.vstack([lo, hi]), fmt="o", capsize=5)
    ax2.set_xticks(xp, p["step"])
    ax2.set_ylabel(r"Paired mean $\Delta A_{90}$ (km$^2$)")
    ax2.set_title("(b) Incremental footprint response", loc="left", fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.25)
    for xx, row in p.iterrows():
        ax2.text(xx, row.delta_A90_mean_km2+12,
                 f"{row.delta_A90_mean_km2:.1f}\n95% CI [{row.CI95_low_km2:.1f}, {row.CI95_high_km2:.1f}]",
                 ha="center", va="bottom", fontsize=9)

    bars = ax3.bar(x, a["surface_exposure_mean"])
    ax3.set_xticks(x, labels)
    ax3.set_ylim(0.72, 1.035)
    ax3.set_ylabel(r"Ensemble mean post-release surface exposure, $f_s$")
    ax3.set_title("(c) Surface residence across configurations", loc="left", fontweight="bold")
    ax3.grid(True, axis="y", alpha=0.25)
    for b, v in zip(bars, a["surface_exposure_mean"]):
        ax3.text(b.get_x()+b.get_width()/2, v+0.006, f"{v:.3f}", ha="center")
    ax3.text(0.99, 0.035,
             r"$n=20$ seeds; $N=1000$; $\Delta t=30$ min; $K_h=100\ \mathrm{m^2\ s^{-1}}$",
             transform=ax3.transAxes, ha="right", va="bottom", fontsize=9)

    for ext in ("pdf", "png", "svg"):
        kw = {"dpi": 600} if ext == "png" else {}
        fig.savefig(OUT / f"Figure2_ensemble_process_attribution.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)


if __name__ == "__main__":
    main()
