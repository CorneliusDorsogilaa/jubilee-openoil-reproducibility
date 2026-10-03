#!/usr/bin/env python3
"""
Script 57: Publication-ready horizontal-diffusivity sensitivity figure.

Purpose
-------
Generate the final 2 x 2 sensitivity figure for the C1 -> C2 comparison
at Kh = 10, 100, and 1000 m^2/s using the 20-seed paired ensemble results.

Changes relative to Script 56
-----------------------------
1. Removes in-panel n = 20 boxes.
2. Removes sign-agreement boxes from the plotting area.
3. Keeps zero reference only for the A90 panels.
4. Removes the zero line from the surface-exposure panel so the small
   Kh-dependent differences remain visually resolvable.
5. Uses mean points with 95% CI error bars and no connecting lines.
6. Writes a compact sign-agreement note below the two A90 panels.
7. Saves PNG, PDF, and SVG versions.

Outputs
-------
Kh_sensitivity_C1_to_C2_pub.png
Kh_sensitivity_C1_to_C2_pub.pdf
Kh_sensitivity_C1_to_C2_pub.svg
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 20-seed paired sensitivity results
# ---------------------------------------------------------------------
kh = np.array([10, 100, 1000], dtype=float)

panels = [
    {
        "mean": np.array([57.026, 62.008, 238.214]),
        "low":  np.array([51.384, 39.461, 110.751]),
        "high": np.array([62.668, 84.556, 365.677]),
        "ylabel": r"$\Delta A_{90}$ (km$^2$)",
        "title": r"(a) Footprint area change",
        "zero": True,
        "sign_note": "Paired sign agreement: 100%, 85%, 75%",
    },
    {
        "mean": np.array([34.848, 10.252, 6.333]),
        "low":  np.array([30.782, 6.374, 3.036]),
        "high": np.array([38.914, 14.131, 9.631]),
        "ylabel": r"Relative $\Delta A_{90}$ (%)",
        "title": r"(b) Relative footprint change",
        "zero": True,
        "sign_note": "Paired sign agreement: 100%, 85%, 75%",
    },
    {
        "mean": np.array([5.783, 5.661, 4.516]),
        "low":  np.array([5.718, 5.553, 4.173]),
        "high": np.array([5.847, 5.769, 4.859]),
        "ylabel": "Centroid endpoint separation (km)",
        "title": r"(c) Centroid endpoint separation",
        "zero": False,
    },
    {
        "mean": np.array([-18.136, -18.165, -18.524]),
        "low":  np.array([-18.288, -18.341, -18.730]),
        "high": np.array([-17.984, -17.989, -18.319]),
        "ylabel": r"$\Delta$ surface exposure (percentage points)",
        "title": r"(d) Surface exposure change",
        "zero": False,
    },
]

# ---------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2))
axes = axes.ravel()

for ax, p in zip(axes, panels):
    mean = p["mean"]
    low = p["low"]
    high = p["high"]
    yerr = np.vstack([mean - low, high - mean])

    ax.errorbar(
        kh,
        mean,
        yerr=yerr,
        fmt="o",
        capsize=4,
        elinewidth=1.3,
        markersize=6,
    )

    ax.set_xscale("log")
    ax.set_xticks([10, 100, 1000])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel(r"$K_h$ (m$^2$ s$^{-1}$)")
    ax.set_ylabel(p["ylabel"])
    ax.set_title(p["title"])
    ax.grid(True, alpha=0.25)

    if p["zero"]:
        ax.axhline(0, linestyle="--", linewidth=1.0)

    if "sign_note" in p:
        ax.text(
            0.5, 0.035,
            p["sign_note"],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

# Tight but readable y ranges for panels where zero is not the comparison target.
axes[2].set_ylim(4.1, 5.9)
axes[3].set_ylim(-18.80, -17.90)

fig.text(
    0.5, 0.005,
    "Points show paired-seed means; error bars show 95% confidence intervals; n = 20 paired realizations per $K_h$.",
    ha="center",
    va="bottom",
    fontsize=9,
)

plt.tight_layout(rect=[0, 0.035, 1, 1])

png_name = "Kh_sensitivity_C1_to_C2_pub.png"
pdf_name = "Kh_sensitivity_C1_to_C2_pub.pdf"
svg_name = "Kh_sensitivity_C1_to_C2_pub.svg"

plt.savefig(png_name, dpi=300, bbox_inches="tight")
plt.savefig(pdf_name, bbox_inches="tight")
plt.savefig(svg_name, bbox_inches="tight")

print(f"Saved: {png_name}")
print(f"Saved: {pdf_name}")
print(f"Saved: {svg_name}")

plt.show()
