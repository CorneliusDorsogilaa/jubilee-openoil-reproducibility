#!/usr/bin/env python3
"""
Script 71: PARTICLE COUNT ensemble-based particle-count convergence analysis.

Purpose
-------
Analyze the completed 20-seed C3 ensembles at:
    N = 500, 1000, 2000, 4000
with dt = 30 min and Kh = 100 m2/s.

Data sources
------------
N = 500, 2000, 4000:
    outputs/particle_count_ensemble/

N = 1000:
    outputs/seed_ensemble/

The script dynamically imports scripts/25_run_production_scenarios.py and
uses its authoritative extract_metrics() function, so centroid displacement,
A90, surface exposure, and shoreline-contact probability are computed exactly
as in the production analysis.

Outputs
-------
diagnostics/validation/particle_count_convergence_per_seed.csv
diagnostics/validation/particle_count_convergence_summary.csv
diagnostics/validation/particle_count_convergence_paired.csv
diagnostics/validation/particle_count_convergence_summary.json
diagnostics/validation/particle_count_convergence_summary.txt

figures/particle_count_convergence_ensemble.png
figures/particle_count_convergence_ensemble.pdf
figures/particle_count_convergence_ensemble.svg
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SCRIPT25 = ROOT / "scripts" / "25_run_production_scenarios.py"

DIAG = ROOT / "diagnostics" / "validation"
FIGDIR = ROOT / "figures"
DIAG.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

SEEDS = list(range(20220901, 20220921))
PARTICLE_COUNTS = [500, 1000, 2000, 4000]
PAIRS = [(500, 1000), (1000, 2000), (2000, 4000)]

RELATIVE_CRITERION_PERCENT = 5.0
SEED_RE = re.compile(r"seed(\d+)", re.I)


def load_script25():
    spec = importlib.util.spec_from_file_location(
        "production_runner_metrics_m4",
        SCRIPT25,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {SCRIPT25}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    files = {}

    particle_dir = ROOT / "outputs" / "particle_count_ensemble"
    seed_dir = ROOT / "outputs" / "seed_ensemble"

    for n in [500, 2000, 4000]:
        candidates = list(
            particle_dir.glob(f"C3_N{n}_dt1800s_Kh100_seed*.nc")
        )
        files[n] = latest_by_seed(candidates)

    candidates = (
        list(seed_dir.glob("C3_Full_fate_N1000_dt1800s_Kh100_seed*.nc"))
        + list(seed_dir.glob("C3_N1000_dt1800s_Kh100_seed*.nc"))
    )
    files[1000] = latest_by_seed(candidates)

    for n in PARTICLE_COUNTS:
        missing = [s for s in SEEDS if s not in files[n]]
        if missing:
            raise RuntimeError(
                f"N={n}: expected 20 seeds, missing {missing}. "
                f"Found {len(files[n])} unique seeds."
            )

    return files


def t95_halfwidth(values):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    if n < 2:
        return np.nan
    tcrit = 2.093024054 if n == 20 else 1.96
    return float(tcrit * np.std(vals, ddof=1) / math.sqrt(n))


def summary_stats(values):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    mean = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1))
    ci = t95_halfwidth(vals)
    cv = float(100.0 * sd / mean) if mean != 0 else np.nan
    return {
        "n": int(len(vals)),
        "mean": mean,
        "sd": sd,
        "cv_percent": cv,
        "ci95_low": mean - ci,
        "ci95_high": mean + ci,
        "ci95_halfwidth": ci,
    }


def main():
    prod = load_script25()
    if not hasattr(prod, "extract_metrics"):
        raise RuntimeError("Script 25 does not expose extract_metrics().")

    files = collect_files()
    records = []

    for n in PARTICLE_COUNTS:
        for seed in SEEDS:
            p = files[n][seed]
            print(f"Extracting metrics | N={n} | seed={seed} | {p.name}", flush=True)
            m = prod.extract_metrics(p)

            records.append({
                "N": n,
                "seed": seed,
                "file": str(p.relative_to(ROOT)),
                "centroid_displacement_km": float(
                    m["centroid"]["displacement_km"]
                ),
                "A90_km2": float(
                    m["central_90_footprint_area_km2"]
                ),
                "surface_exposure_fraction": float(
                    m["surface_exposure_fraction_6h_to_48h"]
                ),
                "shoreline_contact_probability": float(
                    m["shoreline_contact_probability"]
                ),
            })

    df = pd.DataFrame(records)
    per_seed_csv = DIAG / "particle_count_convergence_per_seed.csv"
    df.to_csv(per_seed_csv, index=False)

    metrics = {
        "centroid_displacement_km": "Centroid displacement (km)",
        "A90_km2": r"$A_{90}$ (km$^2$)",
        "surface_exposure_fraction": "Surface exposure fraction",
        "shoreline_contact_probability": "Shoreline-contact probability",
    }

    summary_rows = []
    summary_json = {}

    for n in PARTICLE_COUNTS:
        d = df[df["N"] == n]
        summary_json[str(n)] = {}

        for metric in metrics:
            stats = summary_stats(d[metric].to_numpy(dtype=float))
            summary_json[str(n)][metric] = stats
            summary_rows.append({
                "N": n,
                "metric": metric,
                **stats,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = DIAG / "particle_count_convergence_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    paired_rows = []
    paired_json = {}

    for low_n, high_n in PAIRS:
        low = df[df["N"] == low_n].set_index("seed").sort_index()
        high = df[df["N"] == high_n].set_index("seed").sort_index()

        pair_key = f"{low_n}_to_{high_n}"
        paired_json[pair_key] = {}

        for metric in metrics:
            low_vals = low.loc[SEEDS, metric].to_numpy(dtype=float)
            high_vals = high.loc[SEEDS, metric].to_numpy(dtype=float)

            diff = high_vals - low_vals
            low_mean = float(np.mean(low_vals))
            high_mean = float(np.mean(high_vals))

            rel_mean_diff = (
                100.0 * (high_mean - low_mean) / low_mean
                if low_mean != 0 else np.nan
            )

            diff_stats = summary_stats(diff)

            if metric == "shoreline_contact_probability":
                decision = "REPORT_ABSOLUTE"
            else:
                decision = (
                    "PASS"
                    if abs(rel_mean_diff) <= RELATIVE_CRITERION_PERCENT
                    else "FAIL"
                )

            row = {
                "comparison": pair_key,
                "N_low": low_n,
                "N_high": high_n,
                "metric": metric,
                "mean_low": low_mean,
                "mean_high": high_mean,
                "ensemble_mean_relative_difference_percent": rel_mean_diff,
                "paired_mean_difference": diff_stats["mean"],
                "paired_difference_ci95_low": diff_stats["ci95_low"],
                "paired_difference_ci95_high": diff_stats["ci95_high"],
                "decision_5pct": decision,
            }
            paired_rows.append(row)
            paired_json[pair_key][metric] = row

    paired_df = pd.DataFrame(paired_rows)
    paired_csv = DIAG / "particle_count_convergence_paired.csv"
    paired_df.to_csv(paired_csv, index=False)

    bundle = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scenario": "C3_Full_fate",
        "particle_counts": PARTICLE_COUNTS,
        "seeds": SEEDS,
        "n_seeds_per_N": 20,
        "fixed_settings": {
            "calculation_timestep_minutes": 30,
            "horizontal_diffusivity_m2_s": 100,
        },
        "relative_convergence_criterion_percent": RELATIVE_CRITERION_PERCENT,
        "ensemble_summary": summary_json,
        "adjacent_N_comparisons": paired_json,
    }

    json_path = DIAG / "particle_count_convergence_summary.json"
    json_path.write_text(
        json.dumps(bundle, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "PARTICLE COUNT — ENSEMBLE-BASED PARTICLE-COUNT CONVERGENCE",
        "=" * 94,
        f"Generated UTC: {bundle['generated_utc']}",
        "Scenario: C3_Full_fate",
        "Fixed dt = 30 min, Kh = 100 m2/s",
        "20 seeds per particle count",
        "",
        "ENSEMBLE MEANS ± 95% CI",
        "-" * 94,
    ]

    for n in PARTICLE_COUNTS:
        lines.append(f"N={n}")
        for metric, label in metrics.items():
            s = summary_json[str(n)][metric]
            lines.append(
                f"  {label}: mean={s['mean']:.6g}, SD={s['sd']:.6g}, "
                f"95% CI=[{s['ci95_low']:.6g}, {s['ci95_high']:.6g}], "
                f"CV={s['cv_percent']:.3f}%"
            )
        lines.append("")

    lines.extend([
        "ADJACENT-N ENSEMBLE-MEAN COMPARISONS",
        "-" * 94,
    ])

    for low_n, high_n in PAIRS:
        key = f"{low_n}_to_{high_n}"
        lines.append(f"N={low_n} -> N={high_n}")
        for metric, label in metrics.items():
            r = paired_json[key][metric]
            lines.append(
                f"  {label}: relative mean difference="
                f"{r['ensemble_mean_relative_difference_percent']:.3f}% | "
                f"paired mean diff={r['paired_mean_difference']:.6g} | "
                f"95% CI(diff)=[{r['paired_difference_ci95_low']:.6g}, "
                f"{r['paired_difference_ci95_high']:.6g}] | "
                f"{r['decision_5pct']}"
            )
        lines.append("")

    lines.extend([
        "Interpretation:",
        "- PASS/FAIL uses the pre-existing 5% relative criterion for centroid, A90,",
        "  and surface exposure, now applied to 20-seed ensemble means.",
        "- Shoreline-contact probability is reported as an absolute difference because",
        "  its baseline is near zero.",
        "- These tests support convergence of the reported trajectory diagnostics only,",
        "  not convergence of gridded surface-oil concentration fields.",
    ])

    txt_path = DIAG / "particle_count_convergence_summary.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    plt.rcParams.update({
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
    })

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.8))
    axes = axes.ravel()

    for ax, (metric, label) in zip(axes, metrics.items()):
        means, lo, hi = [], [], []

        for n in PARTICLE_COUNTS:
            s = summary_json[str(n)][metric]
            means.append(s["mean"])
            lo.append(s["mean"] - s["ci95_low"])
            hi.append(s["ci95_high"] - s["mean"])

        yerr = np.vstack([lo, hi])

        ax.errorbar(
            PARTICLE_COUNTS,
            means,
            yerr=yerr,
            fmt="o-",
            capsize=4,
            linewidth=1.5,
            markersize=5.5,
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(PARTICLE_COUNTS)
        ax.set_xticklabels([str(n) for n in PARTICLE_COUNTS])
        ax.set_xlabel("Particle count, N")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)

    axes[0].set_title("(a) Centroid displacement")
    axes[1].set_title(r"(b) Central 90% footprint, $A_{90}$")
    axes[2].set_title("(c) Post-release surface exposure")
    axes[3].set_title("(d) Shoreline-contact probability")

    fig.text(
        0.5, 0.008,
        "Points show 20-seed ensemble means; error bars show 95% confidence intervals. "
        "C3 full-fate configuration, dt = 30 min, Kh = 100 m$^2$ s$^{-1}$.",
        ha="center",
        fontsize=9,
    )

    plt.tight_layout(rect=[0, 0.035, 1, 1])

    png = FIGDIR / "particle_count_convergence_ensemble.png"
    pdf = FIGDIR / "particle_count_convergence_ensemble.pdf"
    svg = FIGDIR / "particle_count_convergence_ensemble.svg"

    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(svg, bbox_inches="tight")

    print("")
    print("\n".join(lines))
    print("=" * 94)
    print(f"Saved: {per_seed_csv.relative_to(ROOT)}")
    print(f"Saved: {summary_csv.relative_to(ROOT)}")
    print(f"Saved: {paired_csv.relative_to(ROOT)}")
    print(f"Saved: {json_path.relative_to(ROOT)}")
    print(f"Saved: {txt_path.relative_to(ROOT)}")
    print(f"Saved: {png.relative_to(ROOT)}")
    print(f"Saved: {pdf.relative_to(ROOT)}")
    print(f"Saved: {svg.relative_to(ROOT)}")

    plt.show()


if __name__ == "__main__":
    main()
