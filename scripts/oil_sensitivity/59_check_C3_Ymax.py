#!/usr/bin/env python3
"""
Script 59: Check whether C3 emulsification reaches the ADIOS Y_max ceiling.

Purpose
-------
The C3 mean water fraction (~0.896) is close
close to the Bonny Light ADIOS maximum emulsion water fraction (Y_max = 0.9).

This script inspects the final-time water_fraction distribution in the
authoritative C3 production NetCDF and reports:
    - mean, median, min, max
    - p05, p25, p75, p95, p99
    - fraction/count of valid particles at >= 95%, 99%, 99.5%, and 99.9% of Y_max
    - fraction/count numerically equal to Y_max within tolerance

Outputs
-------
diagnostics/validation/C3_water_fraction_Ymax_check.txt
diagnostics/validation/C3_water_fraction_Ymax_check.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "diagnostics" / "validation"
OUTDIR.mkdir(parents=True, exist_ok=True)

YMAX = 0.9
TOL = 1.0e-6


def find_c3_file() -> Path:
    out_dir = ROOT / "outputs" / "production"
    files = sorted(
        out_dir.glob("C3_Full_fate_N1000_dt1800s_Kh100_seed20220901_*.nc"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(
            "No authoritative C3 production NetCDF found in outputs/production."
        )
    return files[0]


def final_time_values(da: xr.DataArray) -> np.ndarray:
    arr = np.asarray(da.values, dtype=float)
    if "time" in da.dims:
        arr = np.take(arr, -1, axis=da.dims.index("time"))
    return np.asarray(arr, dtype=float).ravel()


def main():
    c3_path = find_c3_file()

    with xr.open_dataset(c3_path) as ds:
        if "water_fraction" not in ds:
            raise KeyError(
                "water_fraction is not present in the C3 production NetCDF."
            )

        wf = final_time_values(ds["water_fraction"])
        wf = wf[np.isfinite(wf)]

        if wf.size == 0:
            raise RuntimeError("No finite final-time water_fraction values found.")

        quantiles = np.quantile(
            wf, [0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        )

        thresholds = [0.95, 0.99, 0.995, 0.999]
        threshold_results = {}
        for frac in thresholds:
            threshold = frac * YMAX
            mask = wf >= threshold
            threshold_results[f"ge_{frac:.3f}_of_Ymax"] = {
                "threshold": float(threshold),
                "count": int(mask.sum()),
                "fraction": float(mask.mean()),
            }

        at_cap = np.isclose(wf, YMAX, rtol=0.0, atol=TOL)

        result = {
            "generated_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "source_netcdf": str(c3_path.relative_to(ROOT)),
            "Ymax": YMAX,
            "tolerance_for_equal_to_cap": TOL,
            "n_valid": int(wf.size),
            "mean": float(np.mean(wf)),
            "median": float(np.median(wf)),
            "min": float(np.min(wf)),
            "max": float(np.max(wf)),
            "p05": float(quantiles[0]),
            "p25": float(quantiles[1]),
            "p75": float(quantiles[3]),
            "p95": float(quantiles[4]),
            "p99": float(quantiles[5]),
            "mean_as_percent_of_Ymax": float(100.0 * np.mean(wf) / YMAX),
            "thresholds": threshold_results,
            "equal_to_Ymax": {
                "count": int(at_cap.sum()),
                "fraction": float(at_cap.mean()),
            },
        }

    json_path = OUTDIR / "C3_water_fraction_Ymax_check.json"
    txt_path = OUTDIR / "C3_water_fraction_Ymax_check.txt"

    json_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "C3 WATER-FRACTION Y_MAX CHECK",
        "=" * 68,
        f"Source NetCDF: {result['source_netcdf']}",
        f"Y_max: {YMAX}",
        f"Valid final particles: {result['n_valid']}",
        "",
        f"Mean:   {result['mean']:.9f}",
        f"Median: {result['median']:.9f}",
        f"Min:    {result['min']:.9f}",
        f"Max:    {result['max']:.9f}",
        f"p05:    {result['p05']:.9f}",
        f"p25:    {result['p25']:.9f}",
        f"p75:    {result['p75']:.9f}",
        f"p95:    {result['p95']:.9f}",
        f"p99:    {result['p99']:.9f}",
        "",
        f"Mean / Y_max: {result['mean_as_percent_of_Ymax']:.3f}%",
        "",
    ]

    for label, info in result["thresholds"].items():
        lines.append(
            f"{label}: threshold={info['threshold']:.6f}, "
            f"count={info['count']}, fraction={info['fraction']:.6f}"
        )

    lines.extend([
        "",
        f"Equal to Y_max within atol={TOL}: "
        f"count={result['equal_to_Ymax']['count']}, "
        f"fraction={result['equal_to_Ymax']['fraction']:.6f}",
        "",
        "Interpretation guide:",
        "- If most particles are at or extremely near 0.9, the reported mean is",
        "  strongly constrained by the ADIOS emulsification ceiling.",
        "- If the distribution remains well below 0.9, the mean is a dynamical",
        "  result rather than simple saturation at Y_max.",
    ])

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print("=" * 68)
    print(f"Saved: {txt_path.relative_to(ROOT)}")
    print(f"Saved: {json_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
