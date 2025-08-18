#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create seasonal HLI anomaly panels (DJF/MAM/JJA/SON) across multiple SSP scenarios.

Inputs (from previous steps):
  - Baseline seasonal means (1985–2014), one per season:
      average_HLI_data_<SEASON>_1985_2014.nc
  - Projection seasonal means for a scenario S over [Y0–Y1], one per season:
      projection_HLI_data_<SEASON>_<Y0>_<Y1>_<S>.nc
    (file names are those produced by Projections_HLI_Seasonal_MME_Generic.py)

This script:
  - loads baseline HLI and per-scenario HLI (same grid),
  - computes anomalies = (projection − baseline),
  - applies a land mask,
  - builds a 4x4 panel (rows = scenarios, cols = seasons),
  - uses a discrete diverging color scale with custom negative/positive bins,
  - saves a single PNG panel to --outdir.

Directory expectations (configurable):
  --baseline-dir contains the 4 baseline NetCDF (one per season)
  --projections-root contains per-scenario subfolders with the 4 projection NetCDF (one per season)
     e.g. <projections-root>/ssp585/projection_HLI_data_DJF_2076_2100_ssp585.nc

Example:
  python HLI_Anomalies_Panel.py \
    --baseline-dir ./outputs/baseline \
    --projections-root ./outputs/projections \
    --scenarios ssp126 ssp245 ssp370 ssp585 \
    --year-start 2076 --year-end 2100 \
    --outdir ./figures --panel-name HLI_Anomalies_all_ssps_2076-2100.png
"""

import argparse
import os
from pathlib import Path
import math
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import geopandas as gpd
from shapely.geometry import mapping
from rasterio import features
from affine import Affine

from matplotlib.colors import ListedColormap, BoundaryNorm


SEASONS = ['DJF', 'MAM', 'JJA', 'SON']


# --------------------------- CLI ---------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Build multi-SSP seasonal HLI anomaly panel.")
    p.add_argument("--baseline-dir", required=True,
                   help="Folder with baseline NetCDF: average_HLI_data_<SEASON>_1985_2014.nc")
    p.add_argument("--projections-root", required=True,
                   help="Root folder containing per-scenario subfolders with projection NetCDFs.")
    p.add_argument("--scenarios", nargs="+", required=True,
                   help="Scenario list, e.g., ssp126 ssp245 ssp370 ssp585")
    p.add_argument("--year-start", type=int, required=True, help="Projection start year (inclusive).")
    p.add_argument("--year-end", type=int, required=True, help="Projection end year (inclusive).")
    p.add_argument("--outdir", default="./figures", help="Output directory for the PNG panel.")
    p.add_argument("--panel-name", default=None,
                   help="Output PNG filename (default: auto based on years).")
    p.add_argument("--mask-land", action="store_true",
                   help="Apply land mask to anomalies before plotting.")
    # Color bin options
    p.add_argument("--neg-min", type=float, default=-4.0,
                   help="Lower bound for negative anomalies (default: -4).")
    p.add_argument("--neg-steps", type=int, default=4,
                   help="Number of labeled negative edges between neg-min and 0 (default: 4 -> -4,-3,-2,-1,0).")
    p.add_argument("--pos-step", type=float, default=1.0,
                   help="Step for positive anomaly bins (default: 1).")
    p.add_argument("--cap-at-data", action="store_true",
                   help="Cap positive upper bound at data max (otherwise include last full +pos_step).")
    return p


# --------------------------- IO utils ---------------------------

def read_hli(filepath: Path):
    if not filepath.is_file():
        raise FileNotFoundError(f"Missing file: {filepath}")
    with nc.Dataset(filepath, "r") as ds:
        lats = ds.variables["lat"][:]
        lons = ds.variables["lon"][:]
        data = ds.variables["HLI"][:]
    return lats, lons, np.array(data)


def build_paths(baseline_dir: Path, projections_root: Path, scenario: str, season: str, y0: int, y1: int):
    base_nc = baseline_dir / f"average_HLI_data_{season}_1985_2014.nc"
    proj_dir = projections_root / scenario
    proj_nc = proj_dir / f"projection_HLI_data_{season}_{y0}_{y1}_{scenario}.nc"
    return base_nc, proj_nc


# --------------------------- Land mask ---------------------------

def create_land_mask(lats, lons):
    """
    Build a 1/0 land mask for a regular lat/lon grid (NaturalEarth lowres).
    Assumes lons are increasing and represent cell centers.
    """
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    land_geom = [mapping(world.unary_union)]

    x_res = (lons[-1] - lons[0]) / (len(lons) - 1)
    y_res = (lats[-1] - lats[0]) / (len(lats) - 1)
    transform = Affine.translation(lons[0] - x_res / 2, lats[0] - y_res / 2) * Affine.scale(x_res, y_res)

    mask = features.rasterize(
        shapes=land_geom,
        out_shape=(len(lats), len(lons)),
        transform=transform,
        fill=0,          # ocean
        default_value=1, # land
        dtype=np.uint8
    )
    return mask


def apply_land_mask(arr, land_mask):
    return np.where(land_mask == 1, arr, np.nan)


# --------------------------- Plot helpers ---------------------------

def make_colormap(bounds):
    """
    Discrete diverging 'RdBu_r' with stronger blues for negatives and reds for positives.
    The number of colors equals (len(bounds)-1).
    """
    rd_bu_r = plt.cm.get_cmap('RdBu_r')
    n_bins = len(bounds) - 1
    # Split negative vs positive around 0 to pull separately from the cmap
    zero_idx = np.where(np.isclose(bounds, 0))[0]
    if len(zero_idx) == 0:
        # no explicit 0 boundary; fallback to uniform sampling
        colors = rd_bu_r(np.linspace(0.0, 1.0, n_bins))
        return ListedColormap(colors)
    zero_idx = zero_idx[0]
    n_neg = zero_idx
    n_pos = n_bins - n_neg
    # A bit more saturated ends, leave a small white-ish gap around center
    neg_colors = rd_bu_r(np.linspace(0.25, 0.45, max(n_neg, 1)))
    pos_colors = rd_bu_r(np.linspace(0.50, 1.00, max(n_pos, 1)))
    colors = list(neg_colors[:n_neg]) + list(pos_colors[:n_pos])
    return ListedColormap(colors)


def seasonal_subplot(ax, lats, lons, data, title, cmap, norm):
    ax.set_title(title, fontsize=14)
    ax.coastlines(linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 9, 'color': 'black'}
    gl.ylabel_style = {'size': 9, 'color': 'black'}
    ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())

    # ensure ascending latitude for plotting
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        data = data[::-1, :]

    im = ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm, shading='auto', transform=ccrs.PlateCarree())
    return im


# --------------------------- Main ---------------------------

def main():
    args = build_parser().parse_args()

    baseline_dir = Path(args.baseline_dir)
    projections_root = Path(args.projections_root)
    scenarios = [s.strip().lower() for s in args.scenarios]
    y0, y1 = args.year_start, args.year_end

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load baseline once (assume consistent grid among seasons)
    baseline = {}
    ref_lats, ref_lons = None, None
    for season in SEASONS:
        f_base, _ = build_paths(baseline_dir, projections_root, scenarios[0], season, y0, y1)
        lats, lons, base = read_hli(f_base)
        baseline[season] = base
        if ref_lats is None:
            ref_lats, ref_lons = lats, lons
        else:
            if (lats.shape != ref_lats.shape or lons.shape != ref_lons.shape or
                not np.allclose(lats, ref_lats) or not np.allclose(lons, ref_lons)):
                raise ValueError(f"Baseline grid mismatch for {season}")

    # 2) Load projections per scenario, compute anomalies
    anomalies = {s: {} for s in scenarios}
    for ssp in scenarios:
        for season in SEASONS:
            f_base, f_proj = build_paths(baseline_dir, projections_root, ssp, season, y0, y1)
            lats_p, lons_p, proj = read_hli(f_proj)

            # Sanity: grid must match baseline
            if (lats_p.shape != ref_lats.shape or lons_p.shape != ref_lons.shape or
                not np.allclose(lats_p, ref_lats) or not np.allclose(lons_p, ref_lons)):
                raise ValueError(f"Projection grid mismatch for {ssp} {season}")

            anomalies[ssp][season] = proj - baseline[season]

    # 3) Optional land mask
    if args.mask_land:
        land_mask = create_land_mask(ref_lats, ref_lons)
        for ssp in scenarios:
            for season in SEASONS:
                anomalies[ssp][season] = apply_land_mask(anomalies[ssp][season], land_mask)

    # 4) Global min/max over all anomalies (for binning)
    all_vals = np.concatenate([anomalies[ssp][season].ravel()
                               for ssp in scenarios for season in SEASONS])
    raw_vmin = np.nanmin(all_vals)
    raw_vmax = np.nanmax(all_vals)

    # 5) Build discrete bounds:
    #    negatives from neg_min to 0 with 'neg_steps' labeled edges (inclusive of 0),
    #    positives from 0 to max with pos_step.
    neg_min = args.neg_min
    neg_steps = max(args.neg_steps, 1)
    pos_step = args.pos_step

    # Negative bounds (including 0)
    # e.g., neg_min=-4, neg_steps=4 -> [-4,-3,-2,-1,0]
    neg_bounds = np.linspace(neg_min, 0.0, neg_steps + 1)

    # Positive bounds start at 0; stop at either raw_vmax or next full step
    pos_max = raw_vmax if args.cap_at_data else math.ceil(raw_vmax / pos_step) * pos_step

    if pos_max <= 0:
        bounds = neg_bounds
    else:
        pos_bounds = np.arange(0.0, pos_max + 0.5 * pos_step, pos_step)  # include last edge
        # merge, avoiding duplicated zero
        bounds = np.concatenate((neg_bounds, pos_bounds[1:]))

    # 6) Colormap & norm
    cmap = make_colormap(bounds)
    norm = BoundaryNorm(bounds, ncolors=len(bounds) - 1)

    # 7) Plot panel (rows=scenarios, cols=seasons)
    nrows, ncols = len(scenarios), len(SEASONS)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(4.6 * ncols, 3.8 * nrows),
                             subplot_kw={'projection': ccrs.PlateCarree()})

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    im = None
    for r, ssp in enumerate(scenarios):
        for c, season in enumerate(SEASONS):
            title = season if r > 0 else season  # (keep simple column titles)
            ax = axes[r, c]
            im = seasonal_subplot(ax, ref_lats, ref_lons, anomalies[ssp][season], title, cmap, norm)
            if c == 0:
                # left labels: scenario name (uppercase)
                ax.text(-0.10, 0.5, ssp.upper(), va='center', ha='right',
                        fontsize=14, fontweight='bold', rotation='vertical',
                        transform=ax.transAxes)

    # colorbar
    cbar_ax = fig.add_axes([0.20, 0.06, 0.60, 0.02])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', ticks=bounds, extend='neither')
    cbar.set_label('HLI Anomaly (projection − baseline)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # layout & save
    plt.tight_layout(rect=[0.04, 0.10, 0.98, 0.96])
    panel_name = args.panel_name or f"HLI_Anomalies_all_ssps_{args.year_start}-{args.year_end}.png"
    out_png = Path(args.outdir) / panel_name
    plt.savefig(out_png, dpi=450, bbox_inches='tight')
    plt.show()
    print(f"Saved panel → {out_png}")


if __name__ == "__main__":
    main()

