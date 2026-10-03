#!/usr/bin/env python3
"""Publication-ready Figure 1: production transport maps for C0 to C3.

Key design changes relative to the first draft:
- removes the oversized in-figure "Figure 1..." title
- moves the shared legend below the panels so it cannot overlap titles
- uses compact panel titles and a separate metric line
- uses one restrained color per scenario for the 90% footprint/final cloud
- keeps trajectories very light to reduce clutter
- uses identical geographic limits across all panels
- uses mathtext for A_90
- saves high-resolution PNG, PDF, and SVG

Outputs
-------
figures/figure1_production_map_pub.png
figures/figure1_production_map_pub.pdf
figures/figure1_production_map_pub.svg
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = ROOT / "diagnostics" / "production" / "production_summary_latest.json"
FIG_DIR = ROOT / "figures"

PNG_OUT = FIG_DIR / "figure1_production_map_pub.png"
PDF_OUT = FIG_DIR / "figure1_production_map_pub.pdf"
SVG_OUT = FIG_DIR / "figure1_production_map_pub.svg"

SCENARIO_ORDER = ["C0_Hydro", "C1_Surface", "C2_Exchange", "C3_Full_fate"]
DISPLAY_NAMES = {
    "C0_Hydro": "C0  Hydrodynamic transport",
    "C1_Surface": "C1  Surface drift",
    "C2_Exchange": "C2  Vertical exchange",
    "C3_Full_fate": "C3  Full fate",
}
PANEL_LABELS = ["a", "b", "c", "d"]

# Restrained, color-blind-friendly sequence.
SCENARIO_COLORS = {
    "C0_Hydro": "#4C78A8",
    "C1_Surface": "#F58518",
    "C2_Exchange": "#54A24B",
    "C3_Full_fate": "#B279A2",
}

RELEASE_COLOR = "#111111"
CENTROID_COLOR = "#D62728"

TRAJECTORY_ALPHA = 0.055
TRAJECTORY_LINEWIDTH = 0.45
MAX_TRAJ_TO_PLOT = 220

FINAL_SCATTER_SIZE = 7
FINAL_SCATTER_ALPHA = 0.22

RELEASE_MARKER_SIZE = 72
CENTROID_MARKER_SIZE = 92

EARTH_RADIUS_M = 6_371_008.8


def orient_trajectory_time(da: xr.DataArray) -> np.ndarray:
    arr = np.asarray(da.values)
    dims = list(da.dims)

    if arr.ndim != 2:
        raise ValueError(f"{da.name} expected 2-D data, got shape {arr.shape}")

    if dims == ["trajectory", "time"]:
        return arr

    if "trajectory" in dims and "time" in dims:
        return np.moveaxis(
            arr,
            [dims.index("trajectory"), dims.index("time")],
            [0, 1],
        )

    raise ValueError(f"{da.name} does not contain trajectory/time dimensions: {dims}")


def lonlat_to_local_xy_m(lon, lat, lon0, lat0):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    dlon = np.radians(lon - lon0)
    dlat = np.radians(lat - lat0)

    x = EARTH_RADIUS_M * math.cos(math.radians(lat0)) * dlon
    y = EARTH_RADIUS_M * dlat
    return x, y


def local_xy_to_lonlat(x_m, y_m, lon0, lat0):
    x_m = np.asarray(x_m, dtype=float)
    y_m = np.asarray(y_m, dtype=float)

    lon = lon0 + np.degrees(
        x_m / (EARTH_RADIUS_M * math.cos(math.radians(lat0)))
    )
    lat = lat0 + np.degrees(y_m / EARTH_RADIUS_M)

    return lon, lat


def convex_hull(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    pts = pts[np.all(np.isfinite(pts), axis=1)]

    if len(pts) <= 1:
        return pts

    pts = np.unique(pts, axis=0)
    if len(pts) <= 2:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (
            (a[0] - o[0]) * (b[1] - o[1])
            - (a[1] - o[1]) * (b[0] - o[0])
        )

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))

    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def central_90_hull_lonlat(lon_final, lat_final, lon0, lat0):
    valid = np.isfinite(lon_final) & np.isfinite(lat_final)
    lon = np.asarray(lon_final)[valid]
    lat = np.asarray(lat_final)[valid]

    if len(lon) < 3:
        return None

    x, y = lonlat_to_local_xy_m(lon, lat, lon0, lat0)

    cx = float(np.mean(x))
    cy = float(np.mean(y))

    r = np.hypot(x - cx, y - cy)
    cutoff = float(np.quantile(r, 0.90))
    keep = r <= cutoff

    pts = np.column_stack([x[keep], y[keep]])

    if len(pts) < 3:
        return None

    hull_xy = convex_hull(pts)
    hull_lon, hull_lat = local_xy_to_lonlat(
        hull_xy[:, 0],
        hull_xy[:, 1],
        lon0,
        lat0,
    )

    return np.column_stack([hull_lon, hull_lat])


def load_summary():
    return json.loads(SUMMARY_PATH.read_text())


def choose_netcdf(run_record: dict) -> Path:
    p = ROOT / run_record["output_netcdf"]
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def shared_extent(all_lon, all_lat, pad_fraction=0.06):
    lon = np.asarray(all_lon, dtype=float)
    lat = np.asarray(all_lat, dtype=float)

    valid = np.isfinite(lon) & np.isfinite(lat)
    lon = lon[valid]
    lat = lat[valid]

    xmin = float(np.min(lon))
    xmax = float(np.max(lon))
    ymin = float(np.min(lat))
    ymax = float(np.max(lat))

    dx = xmax - xmin
    dy = ymax - ymin

    xpad = max(0.015, pad_fraction * dx)
    ypad = max(0.015, pad_fraction * dy)

    return xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad


def main():
    summary = load_summary()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    release_lon = summary["release"]["longitude_deg_east"]
    release_lat = summary["release"]["latitude_deg_north"]

    scenario_data = {}
    all_lon = []
    all_lat = []

    for scenario in SCENARIO_ORDER:
        run = summary["runs"][scenario]
        nc_path = choose_netcdf(run)

        ds = xr.open_dataset(nc_path)
        try:
            lon = orient_trajectory_time(ds["lon"]).astype(float)
            lat = orient_trajectory_time(ds["lat"]).astype(float)

            lon_final = lon[:, -1]
            lat_final = lat[:, -1]

            hull = central_90_hull_lonlat(
                lon_final,
                lat_final,
                release_lon,
                release_lat,
            )

            scenario_data[scenario] = {
                "lon": lon,
                "lat": lat,
                "lon_final": lon_final,
                "lat_final": lat_final,
                "hull": hull,
                "metrics": run["metrics"],
            }

            all_lon.append(lon[np.isfinite(lon)])
            all_lat.append(lat[np.isfinite(lat)])

        finally:
            ds.close()

    all_lon = np.concatenate(all_lon)
    all_lat = np.concatenate(all_lat)

    xmin, xmax, ymin, ymax = shared_extent(all_lon, all_lat)

    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.2, 8.2),
        sharex=True,
        sharey=True,
    )
    axes = axes.ravel()

    for ax, scenario, panel in zip(
        axes,
        SCENARIO_ORDER,
        PANEL_LABELS,
    ):
        d = scenario_data[scenario]
        m = d["metrics"]
        color = SCENARIO_COLORS[scenario]

        lon = d["lon"]
        lat = d["lat"]
        lon_final = d["lon_final"]
        lat_final = d["lat_final"]
        hull = d["hull"]

        ntraj = lon.shape[0]
        stride = max(1, ntraj // MAX_TRAJ_TO_PLOT)

        for i in range(0, ntraj, stride):
            ax.plot(
                lon[i],
                lat[i],
                color=color,
                linewidth=TRAJECTORY_LINEWIDTH,
                alpha=TRAJECTORY_ALPHA,
                zorder=1,
            )

        valid_final = np.isfinite(lon_final) & np.isfinite(lat_final)

        ax.scatter(
            lon_final[valid_final],
            lat_final[valid_final],
            s=FINAL_SCATTER_SIZE,
            facecolor=color,
            edgecolor="none",
            alpha=FINAL_SCATTER_ALPHA,
            zorder=2,
        )

        if hull is not None and len(hull) >= 3:
            ax.add_patch(
                Polygon(
                    hull,
                    closed=True,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=1.7,
                    alpha=0.16,
                    zorder=3,
                )
            )
            closed = np.vstack([hull, hull[0]])
            ax.plot(
                closed[:, 0],
                closed[:, 1],
                color=color,
                linewidth=1.7,
                zorder=4,
            )

        ax.scatter(
            release_lon,
            release_lat,
            marker="^",
            s=RELEASE_MARKER_SIZE,
            facecolor=RELEASE_COLOR,
            edgecolor="white",
            linewidth=0.7,
            zorder=6,
        )

        centroid_lon = m["centroid"]["longitude_deg_east"]
        centroid_lat = m["centroid"]["latitude_deg_north"]

        ax.scatter(
            centroid_lon,
            centroid_lat,
            marker="*",
            s=CENTROID_MARKER_SIZE,
            facecolor=CENTROID_COLOR,
            edgecolor="white",
            linewidth=0.7,
            zorder=7,
        )

        # Panel label inside the axes to avoid collisions.
        ax.text(
            0.02,
            0.98,
            panel,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=12,
        )

        ax.set_title(
            DISPLAY_NAMES[scenario],
            pad=8,
            fontweight="semibold",
        )

        metric_text = (
            rf"$D_c$ = {m['centroid']['displacement_km']:.1f} km   "
            rf"$A_{{90}}$ = {m['central_90_footprint_area_km2']:.1f} km$^2$" "\n"
            rf"$f_s$ = {m['surface_exposure_fraction_6h_to_48h']:.3f}   "
            rf"$f_{{shore}}$ = {m['shoreline_contact_probability']:.3f}"
        )

        ax.text(
            0.50,
            0.975,
            metric_text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.8,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="none",
                alpha=0.78,
            ),
            zorder=8,
        )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax.grid(
            True,
            linewidth=0.45,
            alpha=0.22,
        )

        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")

        # Keep geographic aspect approximately correct.
        mid_lat = 0.5 * (ymin + ymax)
        ax.set_aspect(1 / math.cos(math.radians(mid_lat)))

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    # Shared legend below the panels.
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="None",
            markerfacecolor=RELEASE_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=8,
            label="Release point",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="None",
            markerfacecolor=CENTROID_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=10,
            label="Final centroid",
        ),
        Line2D(
            [0],
            [0],
            color="#666666",
            linewidth=1.8,
            label="Central 90% footprint",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.985,
        top=0.965,
        bottom=0.10,
        wspace=0.16,
        hspace=0.22,
    )

    fig.savefig(
        PNG_OUT,
        dpi=500,
        bbox_inches="tight",
    )
    fig.savefig(
        PDF_OUT,
        bbox_inches="tight",
    )
    fig.savefig(
        SVG_OUT,
        bbox_inches="tight",
    )

    print("Wrote:", PNG_OUT)
    print("Wrote:", PDF_OUT)
    print("Wrote:", SVG_OUT)


if __name__ == "__main__":
    main()
