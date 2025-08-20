#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Seasonal HLI anomaly panel (DJF/MAM/JJA/SON) across multiple CMIP6 SSP scenarios.

Inputs (produced by your baseline/projection scripts)
----------------------------------------------------
Baseline (1985–2014), one file per season:
    <BASELINE_DIR>/average_HLI_data_<SEASON>_1985_2014.nc

Projections (YEAR_START–YEAR_END), one file per season and scenario, stored in
<PROJECTIONS_ROOT>/<scenario>/, with this file names:
    projection_HLI_data_<SEASON>_<Y0>_<Y1>.nc

What this script does
---------------------
- loads baseline & projections (same grid),
- computes anomalies = projection − baseline,
- optionally applies a land mask (oceans → NaN),
- builds a 4×4 panel (rows = scenarios, cols = seasons),
- uses a discrete diverging color scale:
    negatives fixed at [-4, -3, -2, -1, 0]; positives from 0 to ceil(max) by 1,
- shows only integer ticks on the colorbar,
- saves one PNG.

Run style
---------
Spyder-friendly: edit the USER SETTINGS below and press Run (F5).
"""

# ----------------------- USER SETTINGS (edit these) -----------------------
BASELINE_DIR     = r"your path" #eg., ".../Dataset/HLI_CliNO_Season_file_nc/CliNO"
PROJECTIONS_ROOT = r"your path" #eg., ".../Dataset/HLI_Projections_file_nc_2026-2050"
SCENARIOS        = ["ssp126", "ssp245", "ssp370", "ssp585"]     # folder names inside PROJECTIONS_ROOT
YEAR_START       = 2026
YEAR_END         = 2050
OUTDIR           = r"your path"
PANEL_NAME       = None      # e.g. "HLI_Anomalies_all_ssps_2076-2100.png"; if None -> auto
APPLY_LAND_MASK  = True
# -------------------------------------------------------------------------

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

SEASONS = ["DJF", "MAM", "JJA", "SON"]

# ---------- I/O helpers ----------

def read_hli(filepath: Path):
    """Return lat, lon, HLI arrays from a NetCDF file."""
    if not filepath.is_file():
        raise FileNotFoundError(f"Missing file: {filepath}")
    with nc.Dataset(filepath, "r") as ds:
        lats = ds.variables["lat"][:]
        lons = ds.variables["lon"][:]
        data = ds.variables["HLI"][:]
    return lats, lons, np.array(data)

def projection_file(proj_dir: Path, season: str, y0: int, y1: int, scenario: str) -> Path:
    """Resolve projection filename (with/without scenario suffix)."""
    cand1 = proj_dir / f"projection_HLI_data_{season}_{y0}_{y1}.nc"
    cand2 = proj_dir / f"projection_HLI_data_{season}_{y0}_{y1}_{scenario}.nc"
    if cand1.is_file(): return cand1
    if cand2.is_file(): return cand2
    raise FileNotFoundError(f"Missing projection for {scenario} {season}:\n  {cand1}\n  {cand2}")

def build_paths(baseline_dir: Path, projections_root: Path, scenario: str, season: str, y0: int, y1: int):
    """Return (baseline_nc, projection_nc) paths for a given season/scenario."""
    base_nc = baseline_dir / f"average_HLI_data_{season}_1985_2014.nc"
    proj_nc = projection_file(projections_root / scenario, season, y0, y1, scenario)
    return base_nc, proj_nc

# ---------- Land mask ----------

def create_land_mask(lats, lons, ne_res="110m"):
    """
    Build a 1/0 land mask on a regular lat/lon grid using Natural Earth (via Cartopy).
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
    """Return array with ocean cells (0) masked to NaN."""
    return np.where(land_mask == 1, arr, np.nan)

# ---------- Plotting helpers ----------

def make_colormap(bounds):
    """
    Build a discrete diverging 'RdBu_r' colormap with separate sampling for negatives/positives.
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
    """Draw one seasonal map on a Cartopy axis."""
    ax.set_title(title, fontsize=16, fontfamily='Times New Roman')
    ax.coastlines(linewidth=0.4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10, 'color': 'black', 'fontfamily': 'Times New Roman'}
    gl.ylabel_style = {'size': 10, 'color': 'black', 'fontfamily': 'Times New Roman'}
    ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())

    if lats[0] > lats[-1]:   # ensure ascending latitude
        lats = lats[::-1]
        data = data[::-1, :]

    im = ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm, shading='auto', transform=ccrs.PlateCarree())
    return im

def scenario_label(ssp: str) -> str:
    """Pretty label for common SSP codes (ssp126 -> 'SSP1-2.6')."""
    m = {
        "ssp126": "SSP1-2.6",
        "ssp245": "SSP2-4.5",
        "ssp370": "SSP3-7.0",
        "ssp585": "SSP5-8.5",
    }
    return m.get(ssp.lower(), ssp.upper())

# ---------- Main ----------

def main():
    baseline_dir = Path(BASELINE_DIR)
    projections_root = Path(PROJECTIONS_ROOT)
    scenarios = [s.strip().lower() for s in SCENARIOS]
    y0, y1 = int(YEAR_START), int(YEAR_END)

    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load baseline (assume identical grid for all seasons)
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
            _, f_proj = build_paths(baseline_dir, projections_root, ssp, season, y0, y1)
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

    # 4) Global min/max for binning
    all_vals = np.concatenate([anomalies[ssp][season].ravel()
                               for ssp in scenarios for season in SEASONS])
    raw_vmax = np.nanmax(all_vals)
    vmax = int(math.ceil(raw_vmax))  # upper bound (integer)
    # Fixed negative side as requested
    neg_bounds = np.array([-4, -3, -2, -1, 0], dtype=float)
    pos_bounds = np.arange(0, vmax + 1, 1, dtype=float)
    bounds = np.concatenate((neg_bounds, pos_bounds[1:]))

    # Colormap & norm
    cmap = make_colormap(bounds)
    norm = BoundaryNorm(bounds, ncolors=len(bounds) - 1)

    # 5) Figure & axes
    nrows, ncols = len(scenarios), len(SEASONS)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(4.8 * ncols, 3.8 * nrows),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    # compact spacing between small maps
    fig.subplots_adjust(hspace=-0.55, wspace=0.08)

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
        # left-side scenario label, aligned to the row
        left_ax = axes[r, 0]
        left_ax.text(
            -0.15, 0.5, scenario_label(ssp),
            va='center', ha='right',
            fontsize=16, fontweight='bold', rotation='vertical',
            transform=left_ax.transAxes, fontfamily='Times New Roman'
        )

    # leave a little room at the bottom for the colorbar
    fig.tight_layout(rect=[0.05, 0.13, 0.98, 0.97])

    # Colorbar (close to panels)
    cbar_ax = fig.add_axes([0.22, 0.2, 0.56, 0.022])  # [left, bottom, width, height]
    integer_ticks = np.arange(int(bounds[0]), int(bounds[-1]) + 1, 1)
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal',
                        ticks=integer_ticks, extend='neither')
    cbar.set_label('HLI Anomalies', fontsize=16, fontfamily='Times New Roman')
    cbar.ax.tick_params(labelsize=14)

    # save
    panel_name = PANEL_NAME or f"HLI_Anomalies_all_ssps_{YEAR_START}-{YEAR_END}.png"
    out_png = Path(OUTDIR) / panel_name
    plt.savefig(out_png, dpi=600, bbox_inches='tight')
    plt.show()
    print(f"Saved panel → {out_png}")

# Run
if __name__ == "__main__":
    main()


