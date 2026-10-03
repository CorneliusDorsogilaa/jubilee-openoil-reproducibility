#!/usr/bin/env python3
"""
Script 72: RNG-management validation check.

Purpose
-------
Determine how OpenDrift/OpenOil 1.14.12 manages random-number generation
for the processes used in the C0-C3 experiment.

specifically questioned whether enabling vertical mixing changes
the random-number sequence subsequently consumed by horizontal diffusion.
This script inspects the installed source rather than inferring behavior.

It records:
1. OpenDrift package version.
2. OpenOil constructor/source relevant to seeding.
3. Every OpenOil/base-class method containing random-number calls.
4. Package-wide source lines containing np.random / numpy.random / random.
5. Whether an instantiated OpenOil model exposes an obvious per-process RNG
   object or instead relies on the NumPy global RNG.

Outputs
-------
diagnostics/validation/rng_management_check.txt
diagnostics/validation/rng_management_check.json
"""

from __future__ import annotations

import inspect
import json
import re
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

from opendrift.models.openoil import OpenOil

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "diagnostics" / "validation"
OUTDIR.mkdir(parents=True, exist_ok=True)

RNG_PATTERNS = (
    "np.random",
    "numpy.random",
    "random.",
    "default_rng",
    "RandomState",
)


def safe_source(obj):
    try:
        return inspect.getsource(obj)
    except Exception as exc:
        return f"COULD NOT EXTRACT SOURCE: {type(exc).__name__}: {exc}"


def method_rng_hits(cls):
    hits = []
    seen = set()

    for owner in cls.mro():
        for name, obj in owner.__dict__.items():
            if name in seen:
                continue
            seen.add(name)

            if not callable(obj):
                continue

            try:
                src = inspect.getsource(obj)
            except Exception:
                continue

            if any(p in src for p in RNG_PATTERNS):
                rng_lines = []
                for i, line in enumerate(src.splitlines(), start=1):
                    if any(p in line for p in RNG_PATTERNS):
                        rng_lines.append({
                            "line_in_method": i,
                            "text": line.strip(),
                        })

                hits.append({
                    "owner": f"{owner.__module__}.{owner.__name__}",
                    "method": name,
                    "source_file": inspect.getsourcefile(obj),
                    "rng_lines": rng_lines,
                    "source": src,
                })

    return hits


def package_rng_hits():
    import opendrift

    pkg_root = Path(opendrift.__file__).resolve().parent
    hits = []

    for path in sorted(pkg_root.rglob("*.py")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        for lineno, line in enumerate(text.splitlines(), start=1):
            if any(p in line for p in RNG_PATTERNS):
                hits.append({
                    "file": str(path),
                    "line": lineno,
                    "text": line.strip(),
                })

    return str(pkg_root), hits


def inspect_model_rng_attributes():
    model = OpenOil(seed=20220901, loglevel=40)

    attrs = {}
    for name in dir(model):
        lname = name.lower()
        if "rng" in lname or "random" in lname or "seed" in lname:
            try:
                value = getattr(model, name)
                if callable(value):
                    value_repr = f"<callable {getattr(value, '__name__', type(value).__name__)}>"
                else:
                    value_repr = repr(value)
            except Exception as exc:
                value_repr = f"<ERROR: {exc}>"

            attrs[name] = value_repr

    return attrs


def global_rng_probe():
    """
    Demonstrate whether constructing OpenOil(seed=...) resets/controls NumPy's
    global RNG state. This is diagnostic only; it does not run trajectories.
    """
    np.random.seed(123456)
    before = np.random.random(3).tolist()

    _ = OpenOil(seed=20220901, loglevel=40)

    after = np.random.random(3).tolist()

    np.random.seed(20220901)
    expected_if_global_seeded = np.random.random(3).tolist()

    return {
        "before_constructing_model_after_seed_123456": before,
        "after_constructing_OpenOil_seed_20220901": after,
        "first_values_if_numpy_global_seed_20220901": expected_if_global_seeded,
        "matches_global_seed_signature": bool(
            np.allclose(after, expected_if_global_seeded, rtol=0, atol=0)
        ),
    }


def main():
    od_version = version("opendrift")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    class_hits = method_rng_hits(OpenOil)
    pkg_root, package_hits = package_rng_hits()
    rng_attrs = inspect_model_rng_attributes()
    probe = global_rng_probe()

    init_source = safe_source(OpenOil.__init__)

    bundle = {
        "generated_utc": generated,
        "opendrift_version": od_version,
        "package_root": pkg_root,
        "OpenOil_init_source": init_source,
        "model_rng_related_attributes": rng_attrs,
        "global_rng_probe": probe,
        "OpenOil_and_baseclass_methods_with_rng_calls": class_hits,
        "package_wide_rng_lines": package_hits,
    }

    json_path = OUTDIR / "rng_management_check.json"
    json_path.write_text(
        json.dumps(bundle, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    lines = [
        "RNG — RNG MANAGEMENT CHECK",
        "=" * 92,
        f"Generated UTC: {generated}",
        f"OpenDrift version: {od_version}",
        f"Package root: {pkg_root}",
        "",
        "GLOBAL RNG PROBE",
        "-" * 92,
        f"After constructing OpenOil(seed=20220901): {probe['after_constructing_OpenOil_seed_20220901']}",
        f"Expected first NumPy-global values after np.random.seed(20220901): "
        f"{probe['first_values_if_numpy_global_seed_20220901']}",
        f"Matches global-seed signature: {probe['matches_global_seed_signature']}",
        "",
        "MODEL RNG/SEED-RELATED ATTRIBUTES",
        "-" * 92,
    ]

    if rng_attrs:
        for k, v in rng_attrs.items():
            lines.append(f"{k}: {v}")
    else:
        lines.append("No obvious rng/random/seed attributes found.")

    lines.extend([
        "",
        "OPENOIL / BASE-CLASS METHODS CONTAINING RNG CALLS",
        "-" * 92,
    ])

    if class_hits:
        for hit in class_hits:
            lines.append(
                f"{hit['owner']}.{hit['method']} | {hit['source_file']}"
            )
            for item in hit["rng_lines"]:
                lines.append(
                    f"  line {item['line_in_method']}: {item['text']}"
                )
            lines.append("")
    else:
        lines.append("No RNG calls found in inspected OpenOil class hierarchy.")

    lines.extend([
        "",
        "PACKAGE-WIDE RNG LINES",
        "-" * 92,
    ])

    for hit in package_hits:
        lines.append(
            f"{hit['file']}:{hit['line']}: {hit['text']}"
        )

    lines.extend([
        "",
        "INTERPRETATION GUIDE",
        "-" * 92,
        "If horizontal diffusion, vertical random walk, wave entrainment, and",
        "droplet sampling all call np.random.* without separate Generator objects,",
        "they share NumPy's global stream. Activating additional stochastic processes",
        "then changes how many random variates are consumed and can alter subsequent",
        "draws used by other processes even when the same master seed is supplied.",
        "",
        "If separate per-process Generator/RandomState objects are found, document",
        "those streams instead.",
    ])

    txt_path = OUTDIR / "rng_management_check.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines[:160]))
    print("")
    print(f"Saved: {txt_path.relative_to(ROOT)}")
    print(f"Saved: {json_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
