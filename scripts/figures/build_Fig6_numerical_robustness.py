#!/usr/bin/env python3
"""
build_Fig6_numerical_robustness.py

Rebuild Fig. 6 without the outdated seed-CV panels or the phrase
"minimum converged particle count."

Final structure
---------------
(a) Reference-realization timestep sensitivity at Kh = 100 m2/s.
    These are the documented adjacent-step differences already reported in SI Table S3.
(b) C3 A90 versus particle count using 20-seed ensemble means and 95% t intervals.
(c) Adjacent relative change in the 20-seed ensemble mean A90.
(d) Relative deviation of ensemble-mean centroid displacement and surface exposure
    from the N=4000 ensemble means.

The particle-count panels are recomputed directly from the local C3 NetCDF ensemble
using the same A90 estimator and fs definition as the production analysis.

Outputs
-------
figures/Fig6_numerical_robustness_source.csv
figures/Fig6_numerical_robustness.pdf
figures/Fig6_numerical_robustness.png
figures/Fig6_numerical_robustness.svg
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import t as student_t


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1] if SCRIPT_DIR.name.lower() == "figures" else (SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "scripts" else SCRIPT_DIR)
OUTDIR = ROOT / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

CSV_OUT = OUTDIR / "Fig6_numerical_robustness_source.csv"
PDF_OUT = OUTDIR / "Fig6_numerical_robustness.pdf"
PNG_OUT = OUTDIR / "Fig6_numerical_robustness.png"
SVG_OUT = OUTDIR / "Fig6_numerical_robustness.svg"

RELEASE_LON = -2.89000
RELEASE_LAT = 4.60000
SURFACE_Z_TOL_M = 1.0e-6

N_VALUES = [500, 1000, 2000, 4000]
EXPECTED_SEEDS = list(range(20220901, 20220921))

# Authoritative timestep-sensitivity values from SI Table S3.
TIMESTEP = pd.DataFrame({
    "comparison": ["10 vs 5", "15 vs 10", "30 vs 15"],
    "centroid_percent": [0.05, 0.38, 0.60],
    "A90_percent": [0.33, 1.53, 1.05],
    "fs_percent": [0.23, 0.24, 0.38],
})


def orient_trajectory_time(da: xr.DataArray) -> np.ndarray:
    arr = np.asarray(da.values)
    dims = list(da.dims)

    if "trajectory" in dims and "time" in dims:
        return np.moveaxis(
            arr,
            [dims.index("trajectory"), dims.index("time")],
            [0, 1],
        )

    if arr.ndim == 2:
        # Conservative fallback for older files.
        if arr.shape[0] == da.sizes.get("trajectory", -1):
            return arr
        if arr.shape[1] == da.sizes.get("trajectory", -1):
            return arr.T

    raise ValueError(f"Cannot orient {da.name}: dims={dims}, shape={arr.shape}")


def lonlat_to_local_xy_m(lon, lat):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    dlon = np.radians(lon - RELEASE_LON)
    dlat = np.radians(lat - RELEASE_LAT)
    lat0 = math.radians(RELEASE_LAT)
    R = 6371000.0

    x = R * np.cos(lat0) * dlon
    y = R * dlat
    return x, y


def convex_hull(points):
    pts = np.unique(np.asarray(points, dtype=float), axis=0)
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


def polygon_area_m2(poly):
    poly = np.asarray(poly, dtype=float)
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(
        np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    )


def central_90_footprint_area_km2(lon_final, lat_final):
    valid = np.isfinite(lon_final) & np.isfinite(lat_final)
    lon = np.asarray(lon_final, dtype=float)[valid]
    lat = np.asarray(lat_final, dtype=float)[valid]

    if len(lon) < 3:
        return np.nan

    x, y = lonlat_to_local_xy_m(lon, lat)
    cx = float(np.mean(x))
    cy = float(np.mean(y))

    r = np.hypot(x - cx, y - cy)
    cutoff = float(np.quantile(r, 0.90))
    pts = np.column_stack([x[r <= cutoff], y[r <= cutoff]])
    hull = convex_hull(pts)

    return polygon_area_m2(hull) / 1.0e6


def parse_seed(path: Path):
    m = re.search(r"seed(\d{8})", path.name)
    return int(m.group(1)) if m else None


def candidate_score(path: Path):
    s = str(path).lower()
    score = 0
    if "particle" in s:
        score += 50
    if "convergence" in s:
        score += 50
    if "validation" in s:
        score += 20
    if "production" in s:
        score -= 10
    return score


def find_ensemble_files(N):
    """
    Robust discovery for the particle-count ensemble.

    Multiple archived run sets use different directory / filename conventions, so do not
    assume that N is encoded in the filename. Instead:

    1. scan all C3-like NetCDF files under outputs/,
    2. require one of the 20 ensemble seeds,
    3. open the file and confirm trajectory count == requested N,
    4. prefer paths containing particle/convergence/particle-count/validation terms,
    5. retain one file per seed.

    For N=1000, a seed-ensemble C3 file is scientifically equivalent
    if the dedicated particle-count path is absent, because the production settings are
    the same (C3, dt=1800 s, Kh=100 m2/s).
    """
    outputs = ROOT / "outputs"
    if not outputs.exists():
        raise FileNotFoundError(f"Outputs directory not found: {outputs}")

    candidates = []
    for p in outputs.rglob("*.nc"):
        name = p.name.lower()
        full = str(p).lower()

        # Keep only C3/full-fate files.
        if not ("c3" in name or "full_fate" in name or "full-fate" in name):
            continue

        seed = parse_seed(p)
        if seed not in EXPECTED_SEEDS:
            continue

        # Where filenames carry the settings, reject obvious mismatches.
        if "dt" in name and "dt1800" not in name and "dt1800s" not in name:
            continue
        if "kh" in name and "kh100" not in name:
            continue

        try:
            with xr.open_dataset(p) as ds:
                ntraj = int(ds.sizes.get("trajectory", -1))
        except Exception:
            continue

        if ntraj != N:
            continue

        candidates.append((seed, p))

    # Choose the strongest candidate for each seed.
    unique = {}
    for seed, p in candidates:
        if seed not in unique:
            unique[seed] = p
            continue

        current = unique[seed]

        def score(path):
            s = str(path).lower()
            val = candidate_score(path)
            # Dedicated particle-count material is preferred.
            if "particle_count" in s or "particle-count" in s:
                val += 100
            if f"n{N}" in path.name.lower():
                val += 25
            return val

        if (score(p), p.stat().st_mtime) > (
            score(current), current.stat().st_mtime
        ):
            unique[seed] = p

    files = [unique[s] for s in sorted(unique)]

    print(f"\nN={N}: found {len(files)} unique ensemble seeds")
    for p in files[:5]:
        print(f"  {p}")
    if len(files) > 5:
        print("  ...")

    if len(files) != 20:
        # Print useful diagnostics before stopping.
        print("\nCandidate C3 files with this actual trajectory count:")
        for seed, p in candidates[:50]:
            print(f"  seed={seed} | {p}")

        raise RuntimeError(
            f"Expected 20 unique C3 files for N={N}, found {len(files)}.\n"
            "The script now searches by the ACTUAL NetCDF trajectory count, not "
            "by filename. If this still fails, the local particle-count ensemble files are "
            "not all present under outputs/."
        )

    return files


def extract_metrics(path: Path):
    with xr.open_dataset(path) as ds:
        lon = orient_trajectory_time(ds["lon"]).astype(float)
        lat = orient_trajectory_time(ds["lat"]).astype(float)
        z = orient_trajectory_time(ds["z"]).astype(float)
        times = np.asarray(ds["time"].values)

        lonf = lon[:, -1]
        latf = lat[:, -1]
        validf = np.isfinite(lonf) & np.isfinite(latf)

        x, y = lonlat_to_local_xy_m(lonf[validf], latf[validf])
        centroid_km = math.hypot(float(np.mean(x)), float(np.mean(y))) / 1000.0

        A90 = central_90_footprint_area_km2(lonf, latf)

        release_end = times[0] + np.timedelta64(6, "h")
        post = times >= release_end
        zp = z[:, post]
        validz = np.isfinite(zp)
        fs = float(
            np.sum(validz & (zp >= -SURFACE_Z_TOL_M))
            / np.sum(validz)
        )

    return centroid_km, A90, fs


def mean_ci95(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    mean = float(np.mean(v))
    sd = float(np.std(v, ddof=1))
    tcrit = float(student_t.ppf(0.975, n - 1))
    half = tcrit * sd / math.sqrt(n)
    return mean, mean - half, mean + half, sd, n


def build_particle_table():
    raw = []

    for N in N_VALUES:
        files = find_ensemble_files(N)
        for p in files:
            seed = parse_seed(p)
            centroid, A90, fs = extract_metrics(p)
            raw.append({
                "analysis": "particle_count_seed",
                "N": N,
                "seed": seed,
                "centroid_km": centroid,
                "A90_km2": A90,
                "fs": fs,
                "source_file": str(p),
            })

    return pd.DataFrame(raw)


def summarize_particle(raw):
    rows = []

    for N in N_VALUES:
        d = raw[raw["N"] == N]

        row = {"N": N}
        for col, stem in [
            ("centroid_km", "centroid"),
            ("A90_km2", "A90"),
            ("fs", "fs"),
        ]:
            mean, lo, hi, sd, n = mean_ci95(d[col].to_numpy())
            row[f"{stem}_mean"] = mean
            row[f"{stem}_ci95_low"] = lo
            row[f"{stem}_ci95_high"] = hi
            row[f"{stem}_sd"] = sd
            row[f"{stem}_n"] = n

        rows.append(row)

    out = pd.DataFrame(rows)

    # Adjacent relative change in ensemble means.
    out["A90_adjacent_change_percent"] = np.nan
    for i in range(1, len(out)):
        prev = float(out.loc[i - 1, "A90_mean"])
        now = float(out.loc[i, "A90_mean"])
        out.loc[i, "A90_adjacent_change_percent"] = 100.0 * (now - prev) / prev

    # Relative deviation from the N=4000 ensemble mean.
    ref = out[out["N"] == 4000].iloc[0]
    out["centroid_vs_N4000_percent"] = (
        100.0 * (out["centroid_mean"] - ref["centroid_mean"]) / ref["centroid_mean"]
    )
    out["fs_vs_N4000_percent"] = (
        100.0 * (out["fs_mean"] - ref["fs_mean"]) / ref["fs_mean"]
    )

    return out


def main():
    raw = build_particle_table()
    summary = summarize_particle(raw)

    print("\n20-SEED PARTICLE-COUNT ENSEMBLE SUMMARY")
    print("-" * 90)
    print(summary.to_string(index=False))

    # Create one source CSV containing timestep rows, seed-level rows, and summaries.
    ts_long = TIMESTEP.melt(
        id_vars="comparison",
        var_name="metric",
        value_name="relative_difference_percent",
    )
    ts_long.insert(0, "analysis", "timestep_reference_realization")

    raw_out = raw.copy()
    summary_out = summary.copy()
    summary_out.insert(0, "analysis", "particle_count_summary")

    combined = pd.concat(
        [ts_long, raw_out, summary_out],
        ignore_index=True,
        sort=False,
    )
    combined.to_csv(CSV_OUT, index=False)

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

    # ------------------------------------------------------------------
    # (a) Timestep sensitivity
    # ------------------------------------------------------------------
    x = np.arange(len(TIMESTEP))
    width = 0.23

    metric_specs = [
        ("centroid_percent", "Centroid displacement"),
        ("A90_percent", r"$A_{90}$"),
        ("fs_percent", r"$f_s$"),
    ]

    for j, (col, label) in enumerate(metric_specs):
        axa.bar(x + (j - 1) * width, TIMESTEP[col], width=width, label=label)

    axa.axhline(5.0, linestyle="--", linewidth=1.1, label="5% criterion")
    axa.set_xticks(x)
    axa.set_xticklabels(TIMESTEP["comparison"])
    axa.set_ylabel("Relative difference (%)")
    axa.set_xlabel("Adjacent transport timestep comparison (min)")
    axa.set_title(
        "(a) Reference-realization timestep sensitivity",
        loc="left",
        fontweight="bold",
    )
    axa.set_ylim(0, 5.5)
    axa.grid(axis="y", alpha=0.25)
    axa.legend(frameon=False, ncol=2)

    # ------------------------------------------------------------------
    # (b) Ensemble A90
    # ------------------------------------------------------------------
    Ns = summary["N"].to_numpy(dtype=float)
    means = summary["A90_mean"].to_numpy(dtype=float)
    lower = means - summary["A90_ci95_low"].to_numpy(dtype=float)
    upper = summary["A90_ci95_high"].to_numpy(dtype=float) - means

    axb.errorbar(
        Ns, means, yerr=np.vstack([lower, upper]),
        marker="o", capsize=4, linewidth=1.5,
    )
    axb.set_xscale("log", base=2)
    axb.set_xticks(N_VALUES)
    axb.set_xticklabels([str(n) for n in N_VALUES])
    axb.set_xlabel("Particle count, N")
    axb.set_ylabel(r"Ensemble mean $A_{90}$ (km$^2$)")
    axb.set_title(
        "(b) Particle-count ensemble footprint",
        loc="left",
        fontweight="bold",
    )
    axb.grid(alpha=0.25)

    for N, m in zip(Ns, means):
        axb.annotate(
            f"{m:.0f}",
            (N, m),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
        )

    # ------------------------------------------------------------------
    # (c) Adjacent A90 mean changes
    # ------------------------------------------------------------------
    comps = ["500→1000", "1000→2000", "2000→4000"]
    changes = summary.loc[summary["N"] != 500, "A90_adjacent_change_percent"].to_numpy()

    axc.bar(np.arange(3), changes)
    axc.axhline(5.0, linestyle="--", linewidth=1.1, label="5% criterion")
    axc.set_xticks(np.arange(3))
    axc.set_xticklabels(comps)
    axc.set_ylabel(r"Change in ensemble mean $A_{90}$ (%)")
    axc.set_title(
        "(c) Adjacent particle-count sensitivity",
        loc="left",
        fontweight="bold",
    )
    axc.grid(axis="y", alpha=0.25)
    axc.legend(frameon=False)

    for i, val in enumerate(changes):
        axc.text(i, val + 0.12, f"{val:.2f}%", ha="center", va="bottom", fontsize=9)

    # ------------------------------------------------------------------
    # (d) Other ensemble-mean diagnostics vs N=4000
    # ------------------------------------------------------------------
    axd.plot(
        Ns,
        summary["centroid_vs_N4000_percent"],
        marker="o",
        label="Centroid displacement",
    )
    axd.plot(
        Ns,
        summary["fs_vs_N4000_percent"],
        marker="s",
        label=r"Surface exposure, $f_s$",
    )
    axd.axhline(0.0, linewidth=1.0)
    axd.axhline(5.0, linestyle="--", linewidth=0.9)
    axd.axhline(-5.0, linestyle="--", linewidth=0.9)
    axd.set_xscale("log", base=2)
    axd.set_xticks(N_VALUES)
    axd.set_xticklabels([str(n) for n in N_VALUES])
    axd.set_xlabel("Particle count, N")
    axd.set_ylabel("Deviation from N = 4000 ensemble mean (%)")
    axd.set_title(
        "(d) Centroid and surface-exposure stability",
        loc="left",
        fontweight="bold",
    )
    axd.grid(alpha=0.25)
    axd.legend(frameon=False)

    fig.text(
        0.5,
        0.012,
        "Panel (a) shows the documented reference-realization timestep sensitivity. "
        "Panels (b-d) use 20 paired seeds for each C3 particle count; error bars in (b) "
        "are 95% Student's t intervals. N = 1000 is the adopted production resolution, "
        "not a claimed minimum converged particle count.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )

    fig.tight_layout(rect=[0, 0.055, 1, 1])

    fig.savefig(PDF_OUT, bbox_inches="tight")
    fig.savefig(PNG_OUT, dpi=600, bbox_inches="tight")
    fig.savefig(SVG_OUT, bbox_inches="tight")

    print("\nCreated")
    print("-" * 90)
    print(CSV_OUT)
    print(PDF_OUT)
    print(PNG_OUT)
    print(SVG_OUT)

    plt.show()


if __name__ == "__main__":
    main()
