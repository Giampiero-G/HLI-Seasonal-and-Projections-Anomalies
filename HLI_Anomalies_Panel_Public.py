#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HLI anomaly panel (DJF/MAM/JJA/SON) across multiple scenarios.

Inputs produced by previous scripts:
  - Baseline (1985–2014), one file per season:
      average_HLI_data_<SEASON>_1985_2014.nc
  - Projections for each scenario over [YEAR_START–YEAR_END], one file per season:
      projection_HLI_data_<SEASON>_<YEAR_START>_<YEAR_END>.nc
    OR
      projection_HLI_data_<SEASON>_<YEAR_START>_<YEAR_END>_<SCENARIO>.nc
    stored inside a per-scenario folder.

This script:
  - loads baseline & projections (same grid),
  - computes anomalies = projection − baseline,
  - (optionally) applies a land mask,
  - builds a panel (rows=scenarios, cols=seasons),
  - uses a discrete diverging color scale,
  - saves one PNG to OUTDIR.
"""

# ----------------------- USER SETTINGS (Spyder-friendly) -----------------------
BASELINE_DIR     = r"your path"      # folder with average_HLI_data_<SEASON>_1985_2014.nc
PROJECTIONS_ROOT = r"your path"   # root containing one subfolder per scenario
SCENARIOS        = ["ssp245", "ssp126", "ssp245", "ssp370" , "ssp585"]   # scenario folder names inside PROJECTIONS_ROOT
YEAR_START       = 20xx
YEAR_END         = 20xx
OUTDIR           = r"your path"
PANEL_NAME       = None   # e.g. "HLI_Anomalies_all_ssps_2076-2100.png"; if None, auto-name
APPLY_LAND_MASK  = True

# Color bin options for anomalies
NEG_MIN    = -4.0   # lower bound for negatives
NEG_STEPS  = 4      # labels from NEG_MIN to 0 -> e.g. -4,-3,-2,-1,0
POS_STEP   = 1.0    # positive step size
CAP_AT_DATA = False # True: cap upper bound to data max; False: include last full POS_STEP
# ------------------------------------------------------------------------------

import os
import math
from pathlib import Path

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

from shapely.ops import unary_union
from shapely.geometry import mapping
from rasterio import features
from affine import Affine
from matplotlib.colors import ListedColormap, BoundaryNorm

SEASONS = ['DJF', 'MAM', 'JJA', 'SON']


# --------------------------- IO utils ---------------------------

def read_hli(filepath: Path):
    """Read lat/lon/HLI from a NetCDF file."""
    if not filepath.is_file():
        raise FileNotFoundError(f"Missing file: {filepath}")
    with nc.Dataset(filepath, "r") as ds:
        lats = ds.variables["lat"][:]
        lons = ds.variables["lon"][:]
        data = ds.variables["HLI"][:]
    return lats, lons, np.array(data)


def projection_file(proj_dir: Path, season: str, y0: int, y1: int, scenario: str):
    """
    Resolve a projection filename inside proj_dir for a given season and years.
    Tries both patterns (with and without scenario suffix).
    """
    cand1 = proj_dir / f"projection_HLI_data_{season}_{y0}_{y1}.nc"
    cand2 = proj_dir / f"projection_HLI_data_{season}_{y0}_{y1}_{scenario}.nc"
    if cand1.is_file():
        return cand1
    if cand2.is_file():
        return cand2
    raise FileNotFoundError(f"Missing projection for {scenario} {season}: tried\n  {cand1}\n  {cand2}")


def build_paths(baseline_dir: Path, projections_root: Path, scenario: str, season: str, y0: int, y1: int):
    base_nc = baseline_dir / f"average_HLI_data_{season}_1985_2014.nc"
    proj_dir = projections_root / scenario
    proj_nc = projection_file(proj_dir, season, y0, y1, scenario)
    return base_nc, proj_nc


# --------------------------- Land mask ---------------------------

def create_land_mask(lats, lons, ne_res="110m"):
    """
    Build a 1/0 land mask for a regular lat/lon grid using Natural Earth (via Cartopy).
    Assumes lons/lats are increasing and represent cell centers.
    """
    shp_path = shpreader.natural_earth(resolution=ne_res, category="physical", name="land")
    geoms = list(shpreader.Reader(shp_path).geometries())
    land_union = unary_union(geoms)
    land_geom = [mapping(land_union)]

    x_res = (lons[-1] - lons[0]) / (len(lons) - 1)
    y_res = (lats[-1] - lats[0]) / (len(lats) - 1)
    transform = Affine.translation(lons[0] - x_res/2, lats[0] - y_res/2) * Affine.scale(x_res, y_res)

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
    Discrete diverging 'RdBu_r' with separate sampling for negatives and positives.
    """
    rd_bu_r = plt.cm.get_cmap('RdBu_r')
    n_bins = len(bounds) - 1
    zero_idx = np.where(np.isclose(bounds, 0))[0]
    if len(zero_idx) == 0:
        return ListedColormap(rd_bu_r(np.linspace(0.0, 1.0, n_bins)))
    zero_idx = zero_idx[0]
    n_neg = zero_idx
    n_pos = n_bins - n_neg
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

    if lats[0] > lats[-1]:
        lats = lats[::-1]
        data = data[::-1, :]

    im = ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm, shading='auto', transform=ccrs.PlateCarree())
    return im


# --------------------------- Main (Spyder-friendly) ---------------------------

def main():
    baseline_dir = Path(BASELINE_DIR)
    projections_root = Path(PROJECTIONS_ROOT)
    scenarios = [s.strip().lower() for s in SCENARIOS]
    y0, y1 = int(YEAR_START), int(YEAR_END)

    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load baseline (assume same grid among seasons)
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

            if (lats_p.shape != ref_lats.shape or lons_p.shape != ref_lons.shape or
                not np.allclose(lats_p, ref_lats) or not np.allclose(lons_p, ref_lons)):
                raise ValueError(f"Projection grid mismatch for {ssp} {season}")

            anomalies[ssp][season] = proj - baseline[season]

    # 3) Optional land mask
    if APPLY_LAND_MASK:
        land_mask = create_land_mask(ref_lats, ref_lons)
        for ssp in scenarios:
            for season in SEASONS:
                anomalies[ssp][season] = apply_land_mask(anomalies[ssp][season], land_mask)

    # 4) Global min/max over all anomalies (for binning)
    all_vals = np.concatenate([anomalies[ssp][season].ravel()
                               for ssp in scenarios for season in SEASONS])
    raw_vmin = np.nanmin(all_vals)
    raw_vmax = np.nanmax(all_vals)

    # 5) Build discrete bounds
    neg_bounds = np.linspace(raw_vmin, 0.0, max(NEG_STEPS, 1) + 1)
    pos_max = raw_vmax if CAP_AT_DATA else math.ceil(raw_vmax / POS_STEP) * POS_STEP
    if pos_max <= 0:
        bounds = neg_bounds
    else:
        pos_bounds = np.arange(0.0, pos_max + 0.5 * POS_STEP, POS_STEP)
        bounds = np.concatenate((neg_bounds, pos_bounds[1:]))

    cmap = make_colormap(bounds)
    norm = BoundaryNorm(bounds, ncolors=len(bounds) - 1)

    # 6) Plot panel (rows=scenarios, cols=seasons)
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
            ax = axes[r, c]
            im = seasonal_subplot(ax, ref_lats, ref_lons, anomalies[ssp][season], season, cmap, norm)
            if c == 0:
                ax.text(-0.10, 0.5, ssp.upper(), va='center', ha='right',
                        fontsize=14, fontweight='bold', rotation='vertical',
                        transform=ax.transAxes)

    # colorbar
    cbar_ax = fig.add_axes([0.20, 0.06, 0.60, 0.02])
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal', ticks=bounds, extend='neither')
    cbar.set_label('HLI Anomaly (projection − baseline)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout(rect=[0.04, 0.10, 0.98, 0.96])

    panel_name = PANEL_NAME or f"HLI_Anomalies_all_ssps_{YEAR_START}-{YEAR_END}.png"
    out_png = Path(OUTDIR) / panel_name
    plt.savefig(out_png, dpi=450, bbox_inches='tight')
    plt.show()
    print(f"Saved panel → {out_png}")


# Run (Spyder-friendly)
if __name__ == "__main__":
    main()

