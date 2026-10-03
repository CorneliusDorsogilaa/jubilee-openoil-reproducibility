#!/usr/bin/env python3
"""
Script 76: VERTICAL DEPTH robust vertical-depth statistics.

Purpose
-------
Replace the unstable single maximum vertical excursion with ensemble-based
upper-tail statistics from the completed 20-seed ensemble.

Scenarios analyzed
------------------
C2_Exchange
C3_Full_fate

For each seed:
1. Convert z to positive depth below the sea surface: depth = max(0, -z).
2. For each trajectory, compute its maximum depth reached during the 48 h run.
3. Compute p95 and p99 across those per-trajectory maximum depths.
4. Also report the single absolute maximum only as a non-converged tail
   diagnostic for comparison, not as the recommended reported metric.
5. Compute final-time p95 and p99 depth as an additional robust diagnostic.

Across the 20 seeds:
- mean
- SD
- 95% CI
- min/max across seeds

Outputs
-------
diagnostics/validation/vertical_depth_per_seed.csv
diagnostics/validation/vertical_depth_summary.csv
diagnostics/validation/vertical_depth_summary.json
diagnostics/validation/vertical_depth_summary.txt

figures/vertical_depth_percentiles.png
figures/vertical_depth_percentiles.pdf
figures/vertical_depth_percentiles.svg
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "diagnostics" / "validation"
FIGDIR = ROOT / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

ENSEMBLE_DIR = ROOT / "outputs" / "seed_ensemble"

SCENARIOS = {
    "C2": "C2_Exchange",
    "C3": "C3_Full_fate",
}

SEEDS = list(range(20220901, 20220921))
SEED_RE = re.compile(r"seed(\d+)", re.I)


def seed_from_name(path: Path):
    m = SEED_RE.search(path.name)
    return int(m.group(1)) if m else None


def latest_by_seed(paths):
    chosen = {}
    for p in paths:
        seed = seed_from_name(p)
        if seed is None or seed not in SEEDS:
            continue
        if seed not in chosen or p.stat().st_mtime > chosen[seed].stat().st_mtime:
            chosen[seed] = p
    return chosen


def collect_files():
    out = {}
    for short, full in SCENARIOS.items():
        candidates = list(
            ENSEMBLE_DIR.glob(
                f"{full}_N1000_dt1800s_Kh100_seed*.nc"
            )
        )
        chosen = latest_by_seed(candidates)
        missing = [s for s in SEEDS if s not in chosen]
        if missing:
            raise RuntimeError(
                f"{short}: expected 20 seeds, found {len(chosen)}; missing {missing}"
            )
        out[short] = chosen
    return out


def orient_z(da: xr.DataArray):
    arr = np.asarray(da.values, dtype=float)
    dims = list(da.dims)
    if "trajectory" not in dims or "time" not in dims:
        raise ValueError(f"Unexpected z dims: {dims}")

    arr = np.moveaxis(
        arr,
        [dims.index("trajectory"), dims.index("time")],
        [0, 1],
    )
    return arr


def t95_halfwidth(values):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    if n < 2:
        return np.nan
    tcrit = 2.093024054 if n == 20 else 1.96
    return float(tcrit * np.std(vals, ddof=1) / math.sqrt(n))


def summarize(values):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    mean = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
    hw = t95_halfwidth(vals)
    return {
        "n": int(len(vals)),
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - hw,
        "ci95_high": mean + hw,
        "seed_min": float(np.min(vals)),
        "seed_max": float(np.max(vals)),
    }


def analyze_file(path: Path):
    with xr.open_dataset(path) as ds:
        z = orient_z(ds["z"])

        # Positive depth below sea surface.
        depth = np.where(np.isfinite(z), np.maximum(0.0, -z), np.nan)

        # Maximum excursion reached by each trajectory over the whole simulation.
        per_traj_max = np.nanmax(depth, axis=1)

        # Final-time depth distribution.
        final_depth = depth[:, -1]

        metrics = {
            "trajectory_max_depth_p95_m": float(
                np.nanpercentile(per_traj_max, 95)
            ),
            "trajectory_max_depth_p99_m": float(
                np.nanpercentile(per_traj_max, 99)
            ),
            "absolute_max_depth_m": float(np.nanmax(per_traj_max)),
            "final_depth_p95_m": float(
                np.nanpercentile(final_depth, 95)
            ),
            "final_depth_p99_m": float(
                np.nanpercentile(final_depth, 99)
            ),
            "median_trajectory_max_depth_m": float(
                np.nanmedian(per_traj_max)
            ),
        }

        return metrics


def main():
    files = collect_files()
    records = []

    for scenario, by_seed in files.items():
        for seed in SEEDS:
            path = by_seed[seed]
            print(
                f"Analyzing {scenario} | seed={seed} | {path.name}",
                flush=True,
            )
            m = analyze_file(path)
            records.append({
                "scenario": scenario,
                "seed": seed,
                "file": str(path.relative_to(ROOT)),
                **m,
            })

    df = pd.DataFrame(records)

    per_seed_csv = OUTDIR / "vertical_depth_per_seed.csv"
    df.to_csv(per_seed_csv, index=False)

    metric_labels = {
        "trajectory_max_depth_p95_m":
            "p95 of per-particle maximum depth (m)",
        "trajectory_max_depth_p99_m":
            "p99 of per-particle maximum depth (m)",
        "absolute_max_depth_m":
            "absolute maximum depth (m)",
        "final_depth_p95_m":
            "final-time p95 depth (m)",
        "final_depth_p99_m":
            "final-time p99 depth (m)",
        "median_trajectory_max_depth_m":
            "median per-particle maximum depth (m)",
    }

    summary_rows = []
    summary_json = {}

    for scenario in SCENARIOS:
        summary_json[scenario] = {}
        d = df[df["scenario"] == scenario]

        for metric, label in metric_labels.items():
            s = summarize(d[metric].to_numpy(dtype=float))
            summary_json[scenario][metric] = s
            summary_rows.append({
                "scenario": scenario,
                "metric": metric,
                "label": label,
                **s,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = OUTDIR / "vertical_depth_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    bundle = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "definition": {
            "depth": "positive depth below sea surface = max(0, -z)",
            "trajectory_max": (
                "maximum depth reached by each trajectory over the 48 h run"
            ),
            "recommended_metric": (
                "p95 or p99 across per-trajectory maximum depths, summarized "
                "across 20 independent seeds"
            ),
            "absolute_maximum_status": (
                "retained only as a non-converged tail diagnostic"
            ),
        },
        "summary": summary_json,
    }

    json_path = OUTDIR / "vertical_depth_summary.json"
    json_path.write_text(
        json.dumps(bundle, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "VERTICAL DEPTH — ROBUST VERTICAL-DEPTH STATISTICS",
        "=" * 94,
        "Definition: for each trajectory, calculate its maximum depth reached over",
        "the 48 h run, then take p95/p99 across trajectories. Repeat for 20 seeds.",
        "",
        "ENSEMBLE SUMMARY",
        "-" * 94,
    ]

    for scenario in SCENARIOS:
        lines.append(scenario)
        for metric, label in metric_labels.items():
            s = summary_json[scenario][metric]
            lines.append(
                f"  {label}: mean={s['mean']:.4f}, SD={s['sd']:.4f}, "
                f"95% CI=[{s['ci95_low']:.4f}, {s['ci95_high']:.4f}], "
                f"seed range=[{s['seed_min']:.4f}, {s['seed_max']:.4f}]"
            )
        lines.append("")

    lines.extend([
        "INTERPRETATION",
        "-" * 94,
        "The absolute maximum is intentionally retained only for comparison with the",
        "original SI. The recommended replacement diagnostic is the ensemble mean",
        "p95 or p99 of per-particle maximum depth, with its 95% confidence interval.",
        "This is far less sensitive to a single extreme particle-time realization.",
    ])

    txt_path = OUTDIR / "vertical_depth_summary.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Figure: per-seed p95/p99 distributions and unstable absolute maximum.
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))

    for scenario in SCENARIOS:
        d = df[df["scenario"] == scenario]
        x = np.arange(len(d))

        axes[0].plot(
            x, d["trajectory_max_depth_p95_m"],
            marker="o", linewidth=1.1, markersize=3.5,
            label=scenario,
        )
        axes[1].plot(
            x, d["trajectory_max_depth_p99_m"],
            marker="o", linewidth=1.1, markersize=3.5,
            label=scenario,
        )
        axes[2].plot(
            x, d["absolute_max_depth_m"],
            marker="o", linewidth=1.1, markersize=3.5,
            label=scenario,
        )

    for ax in axes:
        ax.set_xlabel("Seed index (1–20)")
        ax.set_ylabel("Depth below surface (m)")
        ax.grid(True, alpha=0.25)

    axes[0].set_title("(a) p95 of trajectory maxima")
    axes[1].set_title("(b) p99 of trajectory maxima")
    axes[2].set_title("(c) Absolute maximum, tail diagnostic")

    axes[0].legend(frameon=False)

    plt.tight_layout()

    png = FIGDIR / "vertical_depth_percentiles.png"
    pdf = FIGDIR / "vertical_depth_percentiles.pdf"
    svg = FIGDIR / "vertical_depth_percentiles.svg"

    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(svg, bbox_inches="tight")

    print("")
    print("\n".join(lines))
    print("=" * 94)
    print(f"Saved: {per_seed_csv.relative_to(ROOT)}")
    print(f"Saved: {summary_csv.relative_to(ROOT)}")
    print(f"Saved: {json_path.relative_to(ROOT)}")
    print(f"Saved: {txt_path.relative_to(ROOT)}")
    print(f"Saved: {png.relative_to(ROOT)}")
    print(f"Saved: {pdf.relative_to(ROOT)}")
    print(f"Saved: {svg.relative_to(ROOT)}")

    plt.show()


if __name__ == "__main__":
    main()
