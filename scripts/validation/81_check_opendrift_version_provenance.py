#!/usr/bin/env python3
"""
Script 81: OpenDrift version-provenance check.

Purpose
-------
Resolve the apparent version discrepancy observed during checks:

    importlib.metadata.version("opendrift") -> 1.14.12
    opendrift.__version__                  -> 1.14.11
    NetCDF attribute opendrift_version     -> 1.14.11

This script determines where each version string comes from and whether the
NetCDF metadata is populated from the package's internal __version__ value.

It also checks representative production and Qua Iboe NetCDF files.

Outputs
-------
diagnostics/validation/opendrift_version_provenance.txt
diagnostics/validation/opendrift_version_provenance.json
"""

from __future__ import annotations

import inspect
import json
import re
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import xarray as xr
import opendrift

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "diagnostics" / "validation"
OUTDIR.mkdir(parents=True, exist_ok=True)

PKG_ROOT = Path(opendrift.__file__).resolve().parent


def safe_text(path: Path):
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def find_source_hits(patterns):
    hits = []
    for path in sorted(PKG_ROOT.rglob("*.py")):
        text = safe_text(path)
        if not text:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if any(re.search(p, line, re.I) for p in patterns):
                hits.append({
                    "file": str(path),
                    "line": lineno,
                    "text": line.strip(),
                })
    return hits


def candidate_nc_files():
    candidates = []

    # Production C0-C3
    prod = ROOT / "outputs" / "production"
    if prod.exists():
        candidates.extend(sorted(prod.glob("*.nc")))

    # Qua Iboe sensitivity
    q = ROOT / "outputs" / "oil_sensitivity_lighter"
    if q.exists():
        candidates.extend(sorted(q.glob("*.nc")))

    # Existing Bonny/Cabinda sensitivity
    s = ROOT / "outputs" / "oil_sensitivity"
    if s.exists():
        candidates.extend(sorted(s.glob("*.nc")))

    # Keep representative files only if many exist.
    seen = []
    for p in candidates:
        if p not in seen:
            seen.append(p)
    return seen[:30]


def nc_version_attrs(path: Path):
    try:
        with xr.open_dataset(path) as ds:
            attrs = {}
            for k, v in ds.attrs.items():
                if "version" in k.lower() or "opendrift" in k.lower():
                    attrs[k] = str(v)
            return attrs
    except Exception as exc:
        return {"ERROR": f"{type(exc).__name__}: {exc}"}


def main():
    dist_version = metadata.version("opendrift")
    module_version = getattr(opendrift, "__version__", None)

    init_path = Path(opendrift.__file__).resolve()
    init_text = safe_text(init_path)

    dist = metadata.distribution("opendrift")
    dist_info = {
        "name": dist.metadata.get("Name"),
        "version": dist.version,
        "location": str(Path(dist.locate_file("")).resolve()),
    }

    version_assignment_hits = find_source_hits([
        r"__version__\s*=",
        r"opendrift_version",
        r"version\(",
    ])

    nc_files = []
    for p in candidate_nc_files():
        nc_files.append({
            "file": str(p.relative_to(ROOT)),
            "version_attrs": nc_version_attrs(p),
        })

    interpretation = {
        "distribution_version": dist_version,
        "module___version__": module_version,
        "versions_match": str(dist_version) == str(module_version),
        "likely_netcdf_source": (
            "To be determined from source hits below. If NetCDF writer assigns "
            "opendrift_version from opendrift.__version__, then 1.14.11 in NetCDF "
            "is an internal-version-string provenance issue rather than evidence "
            "that the installed distribution is 1.14.11."
        ),
    }

    bundle = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python_package": {
            "distribution": dist_info,
            "module_file": str(init_path),
            "module___version__": module_version,
            "distribution_version": dist_version,
            "module_init_excerpt": [
                line for line in init_text.splitlines()
                if "__version__" in line or "version" in line.lower()
            ][:30],
        },
        "source_hits": version_assignment_hits,
        "netcdf_files": nc_files,
        "interpretation": interpretation,
    }

    json_path = OUTDIR / "opendrift_version_provenance.json"
    json_path.write_text(
        json.dumps(bundle, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    lines = [
        "/ SOFTWARE VERSION — OPENDRIFT VERSION PROVENANCE CHECK",
        "=" * 100,
        f"Generated UTC: {bundle['generated_utc']}",
        "",
        "INSTALLED PACKAGE",
        "-" * 100,
        f"Distribution metadata version : {dist_version}",
        f"Module __version__            : {module_version}",
        f"Module path                   : {init_path}",
        f"Versions match                : {dist_version == module_version}",
        "",
        "MODULE __init__.py VERSION LINES",
        "-" * 100,
    ]

    for line in bundle["python_package"]["module_init_excerpt"]:
        lines.append(line)

    lines.extend([
        "",
        "SOURCE HITS RELEVANT TO VERSION WRITING",
        "-" * 100,
    ])

    for hit in version_assignment_hits:
        lines.append(
            f"{hit['file']}:{hit['line']}: {hit['text']}"
        )

    lines.extend([
        "",
        "REPRESENTATIVE NETCDF VERSION ATTRIBUTES",
        "-" * 100,
    ])

    for item in nc_files:
        lines.append(item["file"])
        if item["version_attrs"]:
            for k, v in item["version_attrs"].items():
                lines.append(f"  {k}: {v}")
        else:
            lines.append("  no version-related global attrs found")

    lines.extend([
        "",
        "INTERPRETATION",
        "-" * 100,
        f"Installed distribution metadata reports OpenDrift {dist_version}.",
        f"The imported module reports __version__={module_version}.",
        "If the NetCDF writer uses the internal module __version__, archived files",
        "will carry that internal string even when the installed distribution metadata",
        "reports a different release number. Use the source hits above to document",
        "the exact provenance rather than guessing or silently rewriting metadata.",
    ])

    txt_path = OUTDIR / "opendrift_version_provenance.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print("")
    print(f"Saved: {txt_path.relative_to(ROOT)}")
    print(f"Saved: {json_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
