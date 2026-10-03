#!/usr/bin/env python3
"""
Script 77: Close RNG and the MECHANISM surface-exposure bookkeeping question.

Purpose
-------
This source-code check addresses two validation questions using the exact installed
OpenDrift/OpenOil environment and the authoritative production analysis script.

RNG:
    Are stochastic processes driven by one shared NumPy RNG stream or by
    independent per-process RNG objects?

MECHANISM bookkeeping:
    How is f_s computed in the production analysis, and does natural dispersion
    deactivate elements or primarily update mass/state variables?

The script does not rerun trajectories.

Outputs
-------
diagnostics/validation/source_semantics_check.txt
diagnostics/validation/source_semantics_check.json
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import re
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from opendrift.models.openoil import OpenOil

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "diagnostics" / "validation"
OUTDIR.mkdir(parents=True, exist_ok=True)

SCRIPT25 = ROOT / "scripts" / "25_run_production_scenarios.py"

RNG_PATTERNS = (
    "np.random",
    "numpy.random",
    "default_rng",
    "RandomState",
    "random.",
)

PROCESS_KEYWORDS = (
    "horizontal_diff",
    "vertical",
    "mix",
    "wave",
    "entrain",
    "droplet",
    "diameter",
    "dispersion",
)

BOOKKEEPING_PATTERNS = (
    "deactivate",
    "mass_dispersed",
    "mass_oil",
    "natural_dispersion",
    "dispersion",
    "status",
    "active",
)


def load_script25():
    spec = importlib.util.spec_from_file_location("prod_semantics", SCRIPT25)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {SCRIPT25}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def safe_source(obj):
    try:
        return inspect.getsource(obj)
    except Exception as exc:
        return f"COULD NOT EXTRACT SOURCE: {type(exc).__name__}: {exc}"


def class_method_hits(cls, patterns):
    hits = []
    seen = set()

    for owner in cls.mro():
        for name, obj in owner.__dict__.items():
            if name in seen or not callable(obj):
                continue
            seen.add(name)

            try:
                src = inspect.getsource(obj)
            except Exception:
                continue

            matched_lines = []
            for i, line in enumerate(src.splitlines(), start=1):
                if any(p.lower() in line.lower() for p in patterns):
                    matched_lines.append({
                        "line_in_method": i,
                        "text": line.strip(),
                    })

            if matched_lines:
                hits.append({
                    "owner": f"{owner.__module__}.{owner.__name__}",
                    "method": name,
                    "source_file": inspect.getsourcefile(obj),
                    "matched_lines": matched_lines,
                    "source": src,
                })

    return hits


def package_search(patterns):
    import opendrift

    pkg_root = Path(opendrift.__file__).resolve().parent
    hits = []

    for path in sorted(pkg_root.rglob("*.py")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        for lineno, line in enumerate(text.splitlines(), start=1):
            if any(p.lower() in line.lower() for p in patterns):
                hits.append({
                    "file": str(path),
                    "line": lineno,
                    "text": line.strip(),
                })

    return str(pkg_root), hits


def obvious_rng_attributes():
    model = OpenOil(seed=20220901, loglevel=40)

    attrs = {}
    for name in dir(model):
        lname = name.lower()
        if "rng" in lname or "random" in lname or "seed" in lname:
            try:
                v = getattr(model, name)
                if callable(v):
                    attrs[name] = f"<callable {getattr(v, '__name__', type(v).__name__)}>"
                else:
                    attrs[name] = repr(v)
            except Exception as exc:
                attrs[name] = f"<ERROR: {exc}>"
    return attrs


def extract_surface_exposure_source(prod):
    src = safe_source(prod.extract_metrics)
    lines = src.splitlines()

    selected = []
    for i, line in enumerate(lines, start=1):
        low = line.lower()
        if (
            "surface_exposure" in low
            or "surface" in low
            or "active" in low
            or "status" in low
            or "z[" in low
            or "z " in low
        ):
            start = max(1, i - 3)
            end = min(len(lines), i + 5)
            selected.append({
                "focus_line": i,
                "context": [
                    {"line": j, "text": lines[j - 1]}
                    for j in range(start, end + 1)
                ],
            })

    return src, selected


def classify_rng_methods(rng_hits):
    classified = []
    for h in rng_hits:
        name_low = h["method"].lower()
        source_low = h["source"].lower()
        if any(k in name_low or k in source_low for k in PROCESS_KEYWORDS):
            classified.append(h)
    return classified


def main():
    prod = load_script25()
    od_version = version("opendrift")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    rng_hits = class_method_hits(OpenOil, RNG_PATTERNS)
    process_rng_hits = classify_rng_methods(rng_hits)

    book_hits = class_method_hits(OpenOil, BOOKKEEPING_PATTERNS)
    pkg_root, pkg_book_hits = package_search(BOOKKEEPING_PATTERNS)

    rng_attrs = obvious_rng_attributes()

    extract_src, surface_contexts = extract_surface_exposure_source(prod)

    has_generator_object = any(
        ("generator" in v.lower() or "randomstate" in v.lower())
        for v in rng_attrs.values()
    )

    process_rng_summary = []
    for h in process_rng_hits:
        process_rng_summary.append({
            "owner": h["owner"],
            "method": h["method"],
            "source_file": h["source_file"],
            "rng_lines": h["matched_lines"],
        })

    bundle = {
        "generated_utc": generated,
        "opendrift_version": od_version,
        "package_root": pkg_root,
        "obvious_rng_related_attributes": rng_attrs,
        "obvious_separate_generator_detected": has_generator_object,
        "process_methods_with_rng_calls": process_rng_summary,
        "production_extract_metrics_source": extract_src,
        "surface_exposure_source_contexts": surface_contexts,
        "openoil_methods_with_bookkeeping_terms": [
            {
                "owner": h["owner"],
                "method": h["method"],
                "source_file": h["source_file"],
                "matched_lines": h["matched_lines"],
            }
            for h in book_hits
        ],
        "package_wide_bookkeeping_hits": pkg_book_hits,
    }

    json_path = OUTDIR / "source_semantics_check.json"
    json_path.write_text(
        json.dumps(bundle, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    lines = [
        "RNG + source semantics — SOURCE SEMANTICS CHECK",
        "=" * 96,
        f"Generated UTC: {generated}",
        f"OpenDrift version: {od_version}",
        "",
        "RNG — RNG MANAGEMENT",
        "-" * 96,
        f"Obvious separate Generator/RandomState object detected: {has_generator_object}",
        "",
        "RNG/SEED-RELATED MODEL ATTRIBUTES",
    ]

    if rng_attrs:
        for k, v in rng_attrs.items():
            lines.append(f"  {k}: {v}")
    else:
        lines.append("  None detected.")

    lines.extend([
        "",
        "PROCESS METHODS CONTAINING RNG CALLS",
        "-" * 96,
    ])

    if process_rng_hits:
        for h in process_rng_hits:
            lines.append(
                f"{h['owner']}.{h['method']} | {h['source_file']}"
            )
            for item in h["matched_lines"]:
                lines.append(
                    f"  line {item['line_in_method']}: {item['text']}"
                )
            lines.append("")
    else:
        lines.append("No process-specific RNG methods found by source inspection.")

    lines.extend([
        "",
        "MECHANISM — PRODUCTION f_s CALCULATION",
        "-" * 96,
    ])

    for block in surface_contexts[:20]:
        lines.append(f"Focus line {block['focus_line']}:")
        for item in block["context"]:
            lines.append(f"  {item['line']:>4}: {item['text']}")
        lines.append("")

    lines.extend([
        "",
        "MECHANISM — OPENOIL BOOKKEEPING / DEACTIVATION SOURCE HITS",
        "-" * 96,
    ])

    for h in book_hits:
        method_low = h["method"].lower()
        if (
            "disp" in method_low
            or "weather" in method_low
            or "deact" in h["source"].lower()
            or "mass_dispersed" in h["source"].lower()
        ):
            lines.append(
                f"{h['owner']}.{h['method']} | {h['source_file']}"
            )
            for item in h["matched_lines"][:30]:
                lines.append(
                    f"  line {item['line_in_method']}: {item['text']}"
                )
            lines.append("")

    lines.extend([
        "",
        "INTERPRETATION CHECKLIST",
        "-" * 96,
        "RNG:",
        "  If process methods call np.random.* and no independent Generator objects",
        "  are present, the processes share NumPy's global RNG stream. Enabling an",
        "  additional stochastic process changes subsequent draws, so the same master",
        "  seed does not guarantee process-by-process matched random variates.",
        "",
        "MECHANISM:",
        "  Use the extract_metrics source above to state exactly what enters the f_s",
        "  numerator and denominator. Use the OpenOil source hits to determine whether",
        "  natural dispersion deactivates elements or updates mass/state while the",
        "  trajectory remains active.",
    ])

    txt_path = OUTDIR / "source_semantics_check.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines[:220]))
    print("")
    print(f"Saved: {txt_path.relative_to(ROOT)}")
    print(f"Saved: {json_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
