#!/usr/bin/env python3
"""
Script 58: Extract ADIOS oil properties for the oil-sensitivity analysis.

Purpose
-------
Extract the oil-property information actually available to the installed
OpenDrift/OpenOil + NOAA ADIOS database for:

    AD01440  BONNY LIGHT, SHELL OIL
    AD01442  CABINDA BLEND, SHELL OIL

This script does NOT alter or rerun the production simulations.

Outputs
-------
diagnostics/validation/oil_properties_ADIOS.txt
diagnostics/validation/oil_properties_ADIOS.json
diagnostics/validation/AD01440_raw_oil_record.json
diagnostics/validation/AD01442_raw_oil_record.json

The outputs are intended for the manuscript/SI property-comparison table
and for checking the maximum emulsion-water fraction (Y_max).
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from typing import Any

import numpy as np

from opendrift.models.openoil import OpenOil
from adios_db.computation.physical_properties import Density, KinematicViscosity


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "diagnostics" / "validation"
OUTDIR.mkdir(parents=True, exist_ok=True)

OILS = {
    "AD01440": "BONNY LIGHT, SHELL OIL",
    "AD01442": "CABINDA BLEND, SHELL OIL",
}

TEMPERATURES_C = [15.0, 20.0, 25.0, 37.8]


def to_jsonable(obj: Any):
    """Best-effort conversion of ADIOS/Pydantic/numpy objects to JSON-safe data."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    for method_name in ("model_dump", "dict"):
        method = getattr(obj, method_name, None)
        if callable(method):
            try:
                return to_jsonable(method())
            except Exception:
                pass

    if hasattr(obj, "__dict__"):
        try:
            return {
                str(k): to_jsonable(v)
                for k, v in vars(obj).items()
                if not str(k).startswith("_")
            }
        except Exception:
            pass

    return str(obj)


def maybe_get(d: Any, *path, default=None):
    cur = d
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def search_keys(obj: Any, keywords: tuple[str, ...], prefix=""):
    """Recursively return paths/values whose key contains any target keyword."""
    hits = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            p = f"{prefix}.{key}" if prefix else str(key)
            lk = str(key).lower()
            if any(word in lk for word in keywords):
                hits.append((p, value))
            hits.extend(search_keys(value, keywords, p))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            p = f"{prefix}[{i}]"
            hits.extend(search_keys(value, keywords, p))
    return hits


def safe_float(value):
    try:
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value, dtype=float).ravel()
            return arr.tolist()
        return float(value)
    except Exception:
        return value


def extract_one(oil_id: str) -> dict[str, Any]:
    model = OpenOil(seed=20220901, loglevel=40)
    model.set_oiltype_by_id(oil_id)
    oiltype = model.oiltype

    raw_oil = to_jsonable(oiltype.oil)
    raw_gnome = to_jsonable(getattr(oiltype, "gnome_oil", {}))

    # Save the complete raw ADIOS record available in this environment.
    raw_path = OUTDIR / f"{oil_id}_raw_oil_record.json"
    raw_path.write_text(
        json.dumps(raw_oil, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    density = Density(oiltype.oil)
    viscosity = KinematicViscosity(oiltype.oil)

    density_by_temp = {}
    viscosity_by_temp = {}
    for temp_c in TEMPERATURES_C:
        temp_k = temp_c + 273.15
        try:
            density_by_temp[f"{temp_c:g}_C"] = float(
                np.asarray(density.at_temp(temp_k)).squeeze()
            )
        except Exception as exc:
            density_by_temp[f"{temp_c:g}_C"] = f"ERROR: {exc}"

        try:
            # ADIOS returns kinematic viscosity in SI m2/s.
            kv_m2s = float(np.asarray(viscosity.at_temp(temp_k)).squeeze())
            viscosity_by_temp[f"{temp_c:g}_C"] = {
                "m2_per_s": kv_m2s,
                "cSt": kv_m2s * 1.0e6,
            }
        except Exception as exc:
            viscosity_by_temp[f"{temp_c:g}_C"] = f"ERROR: {exc}"

    try:
        interfacial_tension = float(oiltype.oil_water_surface_tension())
    except Exception as exc:
        interfacial_tension = f"ERROR: {exc}"

    # Values directly used by OpenOil's NOAA weathering implementation.
    result = {
        "oil_id": oil_id,
        "name": getattr(oiltype, "name", None),
        "valid": bool(oiltype.valid()),
        "density_kg_m3": density_by_temp,
        "kinematic_viscosity": viscosity_by_temp,
        "oil_water_surface_tension_N_m": interfacial_tension,
        "emulsion_water_fraction_max": safe_float(
            getattr(oiltype, "emulsion_water_fraction_max", None)
        ),
        "bulltime_s": safe_float(getattr(oiltype, "bulltime", None)),
        "bullwinkle": safe_float(getattr(oiltype, "bullwinkle", None)),
        "mass_fraction": to_jsonable(getattr(oiltype, "mass_fraction", None)),
        "molecular_weight": to_jsonable(
            getattr(oiltype, "molecular_weight", None)
        ),
        "boiling_point_K": to_jsonable(
            getattr(oiltype, "boiling_point", None)
        ),
        "gnome_oil": raw_gnome,
    }

    # Search the raw record for fields used in the oil-property comparison.
    result["raw_record_key_hits"] = {
        "api": to_jsonable(search_keys(raw_oil, ("api",))),
        "density": to_jsonable(search_keys(raw_oil, ("density",))),
        "viscosity": to_jsonable(search_keys(raw_oil, ("viscos",))),
        "pour_point": to_jsonable(search_keys(raw_oil, ("pour",))),
        "surface_or_interfacial_tension": to_jsonable(
            search_keys(raw_oil, ("surface_tension", "interfacial", "tension"))
        ),
        "water_or_emulsion": to_jsonable(
            search_keys(raw_oil, ("water", "emulsion"))
        ),
        "distillation_or_cut": to_jsonable(
            search_keys(raw_oil, ("distill", "cut", "boiling"))
        ),
    }

    return result


def format_value(v):
    if isinstance(v, float):
        if math.isfinite(v):
            return f"{v:.8g}"
    return str(v)


def main():
    try:
        od_version = version("opendrift")
    except PackageNotFoundError:
        od_version = "UNKNOWN"

    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    bundle = {
        "generated_utc": generated_utc,
        "opendrift_version": od_version,
        "purpose": "oil-property comparison and Y_max check",
        "oils": {},
    }

    for oil_id in OILS:
        print(f"Extracting {oil_id} ...", flush=True)
        bundle["oils"][oil_id] = extract_one(oil_id)

    json_path = OUTDIR / "oil_properties_ADIOS.json"
    json_path.write_text(
        json.dumps(bundle, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    lines = []
    lines.append("ADIOS OIL PROPERTY EXTRACTION")
    lines.append("=" * 72)
    lines.append(f"Generated UTC: {generated_utc}")
    lines.append(f"OpenDrift version: {od_version}")
    lines.append("")

    for oil_id, r in bundle["oils"].items():
        lines.append(f"{oil_id} | {r['name']}")
        lines.append("-" * 72)
        lines.append(f"Valid ADIOS/OpenOil record: {r['valid']}")
        lines.append(
            f"Oil-water surface tension [N/m]: "
            f"{format_value(r['oil_water_surface_tension_N_m'])}"
        )
        lines.append(
            f"Emulsion water fraction max (Y_max): "
            f"{format_value(r['emulsion_water_fraction_max'])}"
        )
        lines.append(f"Bulltime [s]: {format_value(r['bulltime_s'])}")
        lines.append(f"Bullwinkle: {format_value(r['bullwinkle'])}")

        lines.append("Density:")
        for t, value in r["density_kg_m3"].items():
            lines.append(f"  {t}: {format_value(value)} kg/m3")

        lines.append("Kinematic viscosity:")
        for t, value in r["kinematic_viscosity"].items():
            if isinstance(value, dict):
                lines.append(
                    f"  {t}: {value['cSt']:.6g} cSt "
                    f"({value['m2_per_s']:.8g} m2/s)"
                )
            else:
                lines.append(f"  {t}: {value}")

        gnome = r.get("gnome_oil", {})
        if isinstance(gnome, dict):
            for key in (
                "api",
                "pour_point",
                "emulsion_water_fraction_max",
                "bullwinkle_fraction",
                "bullwinkle_time",
            ):
                if key in gnome:
                    lines.append(f"GNOME {key}: {gnome[key]}")

        lines.append("")
        lines.append(
            f"Raw record saved to diagnostics/validation/"
            f"{oil_id}_raw_oil_record.json"
        )
        lines.append("")

    txt_path = OUTDIR / "oil_properties_ADIOS.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("")
    print("\n".join(lines))
    print("=" * 72)
    print(f"Saved: {txt_path.relative_to(ROOT)}")
    print(f"Saved: {json_path.relative_to(ROOT)}")
    print("Raw ADIOS records also saved for both oils.")


if __name__ == "__main__":
    main()
