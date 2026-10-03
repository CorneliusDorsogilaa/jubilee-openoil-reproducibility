#!/usr/bin/env python3
"""
Script 80: Verify and analyze the OIL SENSITIVITY Qua Iboe lighter-oil sensitivity.

Purpose
-------
Script 79 completed the isolated Qua Iboe runs and intentionally stopped before
interpretation. This wrapper now reuses the already validated metadata-check and
oil-sensitivity analysis workflows (Scripts 50 and 51) on the isolated lighter-
oil directory.

It does NOT overwrite the completed Bonny/Cabinda diagnostics.

Expected lighter-oil inputs
---------------------------
outputs/oil_sensitivity_lighter/
    C2_Exchange_QuaIboe_AD01483.nc
    C3_Full_fate_QuaIboe_AD01483.nc

Outputs
-------
diagnostics/oil_sensitivity_lighter/...
generated helper copies:
scripts/_80_generated_verify_qua_iboe.py
scripts/_80_generated_analyze_qua_iboe.py
"""

from __future__ import annotations

from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]

VERIFY_SOURCE = ROOT / "scripts" / "50_verify_oil_sensitivity_metadata.py"
ANALYZE_SOURCE = ROOT / "scripts" / "51_analyze_oil_sensitivity.py"

VERIFY_PATCHED = ROOT / "scripts" / "_80_generated_verify_qua_iboe.py"
ANALYZE_PATCHED = ROOT / "scripts" / "_80_generated_analyze_qua_iboe.py"


def patch_workflow(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Required validated workflow not found: {source}")

    src = source.read_text(encoding="utf-8")

    # Sanity check: these were the original one-sided sensitivity workflows.
    if "oil_sensitivity" not in src:
        raise RuntimeError(
            f"{source.name} does not contain the expected oil_sensitivity token."
        )

    # Isolate all outputs/diagnostics from the original Bonny/Cabinda workflow.
    src = src.replace("oil_sensitivity", "oil_sensitivity_lighter")

    # Replace the heavier contrast identity with the selected lighter contrast.
    replacements = [
        ("AD01442", "AD01483"),
        ("CABINDA BLEND", "QUA IBOE"),
        ("Cabinda Blend", "Qua Iboe"),
        ("cabinda blend", "qua iboe"),
        ("CABINDA", "QUA_IBOE"),
        ("Cabinda", "QuaIboe"),
        ("cabinda", "qua_iboe"),
    ]
    for old, new in replacements:
        src = src.replace(old, new)

    destination.write_text(src, encoding="utf-8")


def main():
    lighter_dir = ROOT / "outputs" / "oil_sensitivity_lighter"

    expected = [
        lighter_dir / "C2_Exchange_QuaIboe_AD01483.nc",
        lighter_dir / "C3_Full_fate_QuaIboe_AD01483.nc",
    ]
    missing = [str(p.relative_to(ROOT)) for p in expected if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Script 79 outputs are incomplete. Missing: " + ", ".join(missing)
        )

    patch_workflow(VERIFY_SOURCE, VERIFY_PATCHED)
    patch_workflow(ANALYZE_SOURCE, ANALYZE_PATCHED)

    print("=" * 92)
    print("OIL SENSITIVITY — QUA IBOE METADATA VERIFICATION")
    print("=" * 92)
    print(f"Input C2: {expected[0].relative_to(ROOT)}")
    print(f"Input C3: {expected[1].relative_to(ROOT)}")
    print(f"Verifier: {VERIFY_PATCHED.relative_to(ROOT)}")
    print("")
    runpy.run_path(str(VERIFY_PATCHED), run_name="__main__")

    print("")
    print("=" * 92)
    print("OIL SENSITIVITY — QUA IBOE SENSITIVITY ANALYSIS")
    print("=" * 92)
    print(f"Analyzer: {ANALYZE_PATCHED.relative_to(ROOT)}")
    print("")
    runpy.run_path(str(ANALYZE_PATCHED), run_name="__main__")

    print("")
    print("=" * 92)
    print("SCRIPT 80 COMPLETE")
    print("=" * 92)
    print("Qua Iboe lighter-side metadata verification and analysis completed.")
    print("Next step: combine Qua Iboe, Bonny Light, and Cabinda Blend into the")
    print("final two-sided OIL SENSITIVITY sensitivity table/figure.")


if __name__ == "__main__":
    main()
