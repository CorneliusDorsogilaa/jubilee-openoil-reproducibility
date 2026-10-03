#!/usr/bin/env python3
"""
Script 79: Run the lighter-oil sensitivity using the already validated
Script 49 oil-sensitivity workflow.

Selected lower-bound oil
------------------------
AD01483 | QUA IBOE, SHELL OIL

Why this record
---------------
The installed ADIOS/OpenOil screen identified AD01483 as a valid, lighter and
less viscous crude than baseline Bonny Light AD01440, while remaining a
West African light-crude contrast.

This wrapper deliberately reuses the exact Script 49 workflow rather than
reimplementing the simulation logic.

Safety
------
The patched workflow writes into isolated folders:
    outputs/oil_sensitivity_lighter/
    diagnostics/oil_sensitivity_lighter/

so the completed Bonny/Cabinda sensitivity outputs are not overwritten.
"""

from __future__ import annotations

from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "49_run_oil_sensitivity.py"
PATCHED = ROOT / "scripts" / "_79_generated_qua_iboe_workflow.py"

if not SOURCE.exists():
    raise FileNotFoundError(f"Required validated workflow not found: {SOURCE}")

src = SOURCE.read_text(encoding="utf-8")

required_tokens = ["AD01442", "oil_sensitivity"]
missing = [x for x in required_tokens if x not in src]
if missing:
    raise RuntimeError(
        "Script 49 does not contain the expected validated Cabinda workflow "
        f"tokens: {missing}. Refusing to guess."
    )

# Preserve the original Cabinda workflow by isolating all lighter-oil outputs.
src = src.replace("oil_sensitivity", "oil_sensitivity_lighter")

# Replace the existing heavier contrast with the selected lower-bound record.
src = src.replace("AD01442", "AD01483")
src = src.replace("CABINDA BLEND", "QUA IBOE")
src = src.replace("Cabinda Blend", "Qua Iboe")
src = src.replace("cabinda blend", "qua iboe")
src = src.replace("CABINDA", "QUA_IBOE")
src = src.replace("Cabinda", "QuaIboe")
src = src.replace("cabinda", "qua_iboe")

PATCHED.write_text(src, encoding="utf-8")

print("=" * 88)
print("OIL SENSITIVITY — LIGHTER-OIL SENSITIVITY RUN")
print("=" * 88)
print("Validated source workflow :", SOURCE.relative_to(ROOT))
print("Generated patched workflow:", PATCHED.relative_to(ROOT))
print("Oil record                : AD01483 | QUA IBOE")
print("Output root               : outputs/oil_sensitivity_lighter/")
print("Diagnostics root          : diagnostics/oil_sensitivity_lighter/")
print("")
print("Executing isolated lighter-oil workflow ...")
print("=" * 88)

# Execute the generated copy exactly as a standalone script.
runpy.run_path(str(PATCHED), run_name="__main__")
