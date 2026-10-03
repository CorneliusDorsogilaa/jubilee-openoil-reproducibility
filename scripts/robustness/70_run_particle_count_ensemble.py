#!/usr/bin/env python3
"""
Script 70: PARTICLE COUNT particle-count ensemble runs.

Purpose
-------
Replace the old single-realization particle-count convergence test with
ensemble-based C3 convergence at fixed:
    dt = 30 min
    Kh = 100 m2/s
    forcing/release/oil/process settings = authoritative production settings

New runs:
    N = 500, 2000, 4000
    seeds = 20220901 ... 20220920

The existing 20-seed N=1000 ensemble is intentionally NOT rerun.
A later analysis script will combine:
    N=500/2000/4000 from outputs/particle_count_ensemble/
    N=1000 from outputs/seed_ensemble/

Implementation
--------------
Each simulation is executed in a fresh subprocess. The worker dynamically
imports scripts/25_run_production_scenarios.py, overrides only N_ELEMENTS
and MASTER_SEED, and calls its authoritative C3 worker_run().

Outputs
-------
outputs/particle_count_ensemble/*.nc
diagnostics/validation/particle_count_ensemble_run_summary.json
diagnostics/validation/particle_count_ensemble_run_summary.txt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT25 = ROOT / "scripts" / "25_run_production_scenarios.py"

OUTDIR = ROOT / "outputs" / "particle_count_ensemble"
DIAGDIR = ROOT / "diagnostics" / "validation"
OUTDIR.mkdir(parents=True, exist_ok=True)
DIAGDIR.mkdir(parents=True, exist_ok=True)

PARTICLE_COUNTS = [500, 2000, 4000]
SEEDS = list(range(20220901, 20220921))

DT_SECONDS = 1800
KH = 100
SCENARIO = "C3_Full_fate"


def load_script25():
    spec = importlib.util.spec_from_file_location(
        "production_runner_for_m4",
        SCRIPT25,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {SCRIPT25}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def worker(n: int, seed: int, outfile: Path) -> int:
    prod = load_script25()

    # Override ONLY the numerical ensemble dimensions requested for PARTICLE COUNT.
    prod.N_ELEMENTS = int(n)
    prod.MASTER_SEED = int(seed)

    print(
        f"PARTICLE COUNT WORKER | N={n} | seed={seed} | outfile={outfile.name}",
        flush=True,
    )
    return prod.worker_run(SCENARIO, outfile)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--n", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--outfile")
    return p.parse_args()


def main():
    args = parse_args()

    if args.worker:
        if args.n is None or args.seed is None or not args.outfile:
            raise SystemExit("--worker requires --n, --seed, and --outfile")
        return worker(
            args.n,
            args.seed,
            Path(args.outfile).resolve(),
        )

    run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    summary = {
        "generated_utc": run_tag,
        "purpose": "particle-count ensemble-based particle-count convergence",
        "scenario": SCENARIO,
        "fixed_settings": {
            "calculation_timestep_seconds": DT_SECONDS,
            "horizontal_diffusivity_m2_s": KH,
        },
        "particle_counts_new": PARTICLE_COUNTS,
        "particle_count_reused": 1000,
        "seeds": SEEDS,
        "runs": [],
        "overall_status": "RUNNING",
    }

    summary_json = DIAGDIR / "particle_count_ensemble_run_summary.json"
    summary_txt = DIAGDIR / "particle_count_ensemble_run_summary.txt"

    def save():
        summary_json.write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        lines = [
            "PARTICLE COUNT — PARTICLE-COUNT ENSEMBLE RUN SUMMARY",
            "=" * 84,
            f"Generated UTC: {summary['generated_utc']}",
            f"Scenario: {SCENARIO}",
            f"New N values: {PARTICLE_COUNTS}",
            "Reused N=1000: existing 20-seed ensemble",
            f"Seeds: {SEEDS[0]} ... {SEEDS[-1]} ({len(SEEDS)} seeds)",
            f"Overall status: {summary['overall_status']}",
            "",
        ]
        for r in summary["runs"]:
            lines.append(
                f"N={r['N']} | seed={r['seed']} | "
                f"status={r['status']} | source={r['source']} | "
                f"runtime_s={r.get('runtime_seconds')} | "
                f"file={r['file']}"
            )
        summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    save()

    total = len(PARTICLE_COUNTS) * len(SEEDS)
    counter = 0

    for n in PARTICLE_COUNTS:
        for seed in SEEDS:
            counter += 1

            existing = sorted(
                OUTDIR.glob(
                    f"C3_N{n}_dt{DT_SECONDS}s_Kh{KH}_seed{seed}_*.nc"
                ),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )

            # Reuse only if script25's completion test passes under the
            # current overridden N/seed context.
            reusable = None
            if existing:
                prod = load_script25()
                prod.N_ELEMENTS = int(n)
                prod.MASTER_SEED = int(seed)
                for p in existing:
                    try:
                        if prod.is_complete_trajectory(p):
                            reusable = p
                            break
                    except Exception:
                        pass

            if reusable is not None:
                print(
                    f"[{counter}/{total}] REUSE | N={n} | seed={seed} | "
                    f"{reusable.name}",
                    flush=True,
                )
                summary["runs"].append({
                    "N": n,
                    "seed": seed,
                    "status": "PASS",
                    "source": "reused_complete_output",
                    "runtime_seconds": None,
                    "file": str(reusable.relative_to(ROOT)),
                })
                save()
                continue

            outfile = OUTDIR / (
                f"C3_N{n}_dt{DT_SECONDS}s_Kh{KH}_seed{seed}_{run_tag}.nc"
            )

            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--n", str(n),
                "--seed", str(seed),
                "--outfile", str(outfile),
            ]

            print(
                f"[{counter}/{total}] RUN | N={n} | seed={seed}",
                flush=True,
            )

            t0 = time.perf_counter()
            completed = subprocess.run(cmd)
            runtime = time.perf_counter() - t0

            if completed.returncode != 0:
                summary["runs"].append({
                    "N": n,
                    "seed": seed,
                    "status": "FAIL",
                    "source": "fresh_isolated_worker",
                    "runtime_seconds": runtime,
                    "file": str(outfile.relative_to(ROOT)),
                    "returncode": completed.returncode,
                })
                summary["overall_status"] = "FAIL"
                save()
                raise RuntimeError(
                    f"particle-count worker failed: N={n}, seed={seed}, "
                    f"returncode={completed.returncode}"
                )

            summary["runs"].append({
                "N": n,
                "seed": seed,
                "status": "PASS",
                "source": "fresh_isolated_worker",
                "runtime_seconds": runtime,
                "file": str(outfile.relative_to(ROOT)),
            })
            save()

    summary["overall_status"] = "PASS"
    save()

    print("\n" + "=" * 84)
    print("PARTICLE COUNT PARTICLE-COUNT ENSEMBLE: PASS")
    print(f"Summary: {summary_json.relative_to(ROOT)}")
    print(f"Text   : {summary_txt.relative_to(ROOT)}")
    print("Next: run the ensemble convergence analysis script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
