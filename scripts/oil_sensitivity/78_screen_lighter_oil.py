#!/usr/bin/env python3
"""
Script 78: OIL SENSITIVITY lighter-oil candidate screen.

Purpose
-------
noted that the existing Bonny Light -> Cabinda Blend sensitivity
is one-sided because Cabinda is substantially more viscous. This script screens
the installed OpenDrift/ADIOS oil library for a valid LIGHTER / LESS VISCOUS
crude-oil contrast relative to the baseline AD01440 Bonny Light record.

Baseline
--------
AD01440 | BONNY LIGHT, SHELL OIL

Ranking logic
-------------
Candidates must, where properties are available:
1. be valid in OpenDrift,
2. have kinematic viscosity at 37.8 C lower than Bonny Light,
3. have density at 15 C lower than Bonny Light,
4. have API gravity greater than Bonny Light,
5. preferentially look like crude oils rather than refined fuels.

Candidates are ranked by closeness to Bonny Light in API and density while
remaining meaningfully less viscous. This is a SCREEN only. The selected
candidate must still be scientifically justified before being used in the
sensitivity simulations.

Outputs
-------
diagnostics/validation/lighter_oil_candidates.csv
diagnostics/validation/lighter_oil_candidates.json
diagnostics/validation/lighter_oil_candidates.txt
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from opendrift.models.openoil import OpenOil

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "diagnostics" / "validation"
OUTDIR.mkdir(parents=True, exist_ok=True)

BASELINE_ID = "AD01440"
CABINDA_ID = "AD01442"

T_DENSITY_C = 15.0
T_VISCOSITY_C = 37.8

# Refined-product terms to de-prioritize/exclude from a crude-oil contrast.
REFINED_TERMS = (
    "DIESEL",
    "GASOLINE",
    "FUEL OIL",
    "HEATING OIL",
    "GAS OIL",
    "BUNKER",
    "KEROSENE",
    "JET",
    "MARINE GAS",
    "IFO-",
    "ULTRA LOW SULFUR",
    "DISTILLATE",
)


def ffloat(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def extract_record(model: OpenOil, label=None):
    ot = model.oiltype
    g = getattr(ot, "gnome_oil", {}) or {}

    name = getattr(ot, "name", None) or getattr(model, "oil_name", None)
    oil_id = (
        getattr(ot, "oil_id", None)
        or getattr(ot, "id", None)
        or g.get("id")
        or g.get("oil_id")
    )

    try:
        valid = bool(ot.valid())
    except Exception:
        valid = False

    try:
        rho15 = float(ot.density_at_temp(T_DENSITY_C, unit="C"))
    except Exception:
        rho15 = np.nan

    try:
        nu378_m2s = float(ot.kvis_at_temp(T_VISCOSITY_C, unit="C"))
        nu378_cst = nu378_m2s * 1e6
    except Exception:
        nu378_m2s = np.nan
        nu378_cst = np.nan

    api = ffloat(g.get("api"))
    pour_point_k = ffloat(g.get("pour_point"))

    try:
        sigma = float(ot.oil_water_surface_tension())
    except Exception:
        sigma = np.nan

    try:
        ymax = float(ot.emulsion_water_fraction_max)
    except Exception:
        ymax = np.nan

    product_type = getattr(ot, "product_type", None)
    labels = getattr(ot, "labels", None)
    location = getattr(ot, "location", None)

    return {
        "label": label,
        "oil_id": oil_id,
        "name": name,
        "valid": valid,
        "product_type": product_type,
        "labels": str(labels) if labels is not None else None,
        "location": location,
        "api": api,
        "density_15C_kg_m3": rho15,
        "kvis_37p8C_cSt": nu378_cst,
        "kvis_37p8C_m2_s": nu378_m2s,
        "oil_water_surface_tension_N_m": sigma,
        "emulsion_water_fraction_max": ymax,
        "pour_point_K": pour_point_k,
    }


def crude_like(rec):
    text = " ".join(
        str(rec.get(k) or "")
        for k in ("name", "product_type", "labels")
    ).upper()

    if any(term in text for term in REFINED_TERMS):
        return False

    # If product type says crude, that is strong evidence.
    if "CRUDE" in text:
        return True

    # Many ADIOS crude records are named simply by field/blend and do not
    # contain the word CRUDE. Keep them as "possible crude" unless clearly
    # refined. The output retains metadata for manual scientific review.
    return True


def score_candidate(rec, baseline):
    """
    Lower score is better.
    Prioritize similar API/density, but require lower viscosity.
    """
    api0 = baseline["api"]
    rho0 = baseline["density_15C_kg_m3"]
    nu0 = baseline["kvis_37p8C_cSt"]

    api = rec["api"]
    rho = rec["density_15C_kg_m3"]
    nu = rec["kvis_37p8C_cSt"]

    if not all(np.isfinite([api0, rho0, nu0, api, rho, nu])):
        return np.nan

    d_api = abs(api - api0) / 5.0
    d_rho = abs(rho - rho0) / 25.0

    # Prefer viscosity roughly 30-70% lower rather than an extreme condensate.
    visc_ratio = nu / nu0
    visc_target_penalty = abs(visc_ratio - 0.60)

    return float(d_api + d_rho + visc_target_penalty)


def main():
    model = OpenOil(loglevel=50)

    # Exact known records first.
    model.set_oiltype_by_id(BASELINE_ID)
    baseline = extract_record(model, "baseline")

    model.set_oiltype_by_id(CABINDA_ID)
    cabinda = extract_record(model, "existing_heavier_contrast")

    print("Baseline:")
    print(json.dumps(baseline, indent=2, default=str))
    print("\nExisting heavier contrast:")
    print(json.dumps(cabinda, indent=2, default=str))

    rows = []
    failures = []

    # The OpenOil constructor exposes all locally available oil names.
    oil_names = list(model.oiltypes)
    print(f"\nScreening {len(oil_names)} installed OpenOil oil names ...")

    for i, name in enumerate(oil_names, start=1):
        try:
            model.set_oiltype(name)
            rec = extract_record(model)
            rec["screen_index"] = i
            rows.append(rec)
        except Exception as exc:
            failures.append({
                "name": name,
                "error": f"{type(exc).__name__}: {exc}",
            })

    df = pd.DataFrame(rows)

    for col in [
        "api",
        "density_15C_kg_m3",
        "kvis_37p8C_cSt",
        "oil_water_surface_tension_N_m",
        "emulsion_water_fraction_max",
        "pour_point_K",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    api0 = baseline["api"]
    rho0 = baseline["density_15C_kg_m3"]
    nu0 = baseline["kvis_37p8C_cSt"]

    df["api_delta_vs_bonny"] = df["api"] - api0
    df["density_delta_vs_bonny_kg_m3"] = df["density_15C_kg_m3"] - rho0
    df["viscosity_ratio_vs_bonny"] = df["kvis_37p8C_cSt"] / nu0
    df["crude_like"] = [
        crude_like(row)
        for row in df.to_dict(orient="records")
    ]

    df["lighter_than_bonny"] = (
        (df["api"] > api0)
        & (df["density_15C_kg_m3"] < rho0)
        & (df["kvis_37p8C_cSt"] < nu0)
    )

    scores = []
    for rec in df.to_dict(orient="records"):
        if rec["lighter_than_bonny"] and rec["crude_like"] and rec["valid"]:
            scores.append(score_candidate(rec, baseline))
        else:
            scores.append(np.nan)
    df["candidate_score"] = scores

    candidates = (
        df[
            df["valid"]
            & df["crude_like"]
            & df["lighter_than_bonny"]
            & np.isfinite(df["candidate_score"])
        ]
        .sort_values(
            ["candidate_score", "viscosity_ratio_vs_bonny"],
            ascending=[True, True],
        )
        .copy()
    )

    # Exclude the baseline if its alias appears in the list.
    candidates = candidates[
        ~candidates["name"].str.contains("BONNY LIGHT", case=False, na=False)
    ]

    csv_path = OUTDIR / "lighter_oil_candidates.csv"
    candidates.to_csv(csv_path, index=False)

    top = candidates.head(25)

    bundle = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "baseline": baseline,
        "existing_heavier_contrast": cabinda,
        "screening_rules": {
            "valid_OpenDrift_record": True,
            "api_greater_than_bonny": True,
            "density_15C_lower_than_bonny": True,
            "kvis_37p8C_lower_than_bonny": True,
            "refined_products_deprioritized": True,
        },
        "n_oil_names_screened": len(oil_names),
        "n_successfully_extracted": len(rows),
        "n_failures": len(failures),
        "n_lighter_candidates": len(candidates),
        "top25": top.replace({np.nan: None}).to_dict(orient="records"),
        "failures": failures[:100],
    }

    json_path = OUTDIR / "lighter_oil_candidates.json"
    json_path.write_text(
        json.dumps(bundle, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    lines = [
        "OIL SENSITIVITY — LIGHTER-OIL CANDIDATE SCREEN",
        "=" * 104,
        f"Generated UTC: {bundle['generated_utc']}",
        "",
        "BASELINE",
        "-" * 104,
        f"{baseline['oil_id']} | {baseline['name']}",
        f"API={baseline['api']:.3f} | "
        f"density15={baseline['density_15C_kg_m3']:.3f} kg/m3 | "
        f"kvis37.8={baseline['kvis_37p8C_cSt']:.4f} cSt | "
        f"Ymax={baseline['emulsion_water_fraction_max']}",
        "",
        "EXISTING HEAVIER CONTRAST",
        "-" * 104,
        f"{cabinda['oil_id']} | {cabinda['name']}",
        f"API={cabinda['api']:.3f} | "
        f"density15={cabinda['density_15C_kg_m3']:.3f} kg/m3 | "
        f"kvis37.8={cabinda['kvis_37p8C_cSt']:.4f} cSt | "
        f"Ymax={cabinda['emulsion_water_fraction_max']}",
        "",
        f"Screened {len(oil_names)} names; "
        f"{len(candidates)} valid lighter candidates found.",
        "",
        "TOP 25 LIGHTER CANDIDATES",
        "-" * 104,
        "Rank | Oil ID | Name | API | rho15 kg/m3 | nu37.8 cSt | nu/Bonny | Ymax | product type",
    ]

    for rank, (_, r) in enumerate(top.iterrows(), start=1):
        lines.append(
            f"{rank:>2} | {r.get('oil_id')} | {r.get('name')} | "
            f"{r['api']:.2f} | {r['density_15C_kg_m3']:.2f} | "
            f"{r['kvis_37p8C_cSt']:.3f} | "
            f"{r['viscosity_ratio_vs_bonny']:.3f} | "
            f"{r['emulsion_water_fraction_max']} | "
            f"{r.get('product_type')}"
        )

    lines.extend([
        "",
        "NEXT STEP",
        "-" * 104,
        "Select one scientifically credible crude-oil record from this shortlist.",
        "Then run the existing oil-sensitivity workflow with Bonny Light, the lighter",
        "candidate, and Cabinda Blend to provide a two-sided oil-property sensitivity.",
        "Do not select solely by numerical score; inspect oil identity/product type",
        "and property provenance before the production rerun.",
    ])

    txt_path = OUTDIR / "lighter_oil_candidates.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print("")
    print(f"Saved: {csv_path.relative_to(ROOT)}")
    print(f"Saved: {json_path.relative_to(ROOT)}")
    print(f"Saved: {txt_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
