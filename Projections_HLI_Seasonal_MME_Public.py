#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seasonal HLI projections (equal-weight MME, calendar-aware, day-weighted)

Assumptions:
- Your base folder already contains the single scenario & time window you want,
  organized as:
      BASE_DIR/
        ├─ tas/      *.nc
        ├─ hurs/     *.nc
        ├─ sfcWind/  *.nc
        └─ rsds/     *.nc
- Files can be multiple years and multiple models. We:
  1) read ALL *.nc in each variable folder,
  2) keep ONLY timestamps within [YEAR_START, YEAR_END] (inclusive),
  3) for each MODEL, compute seasonal means as day-weighted averages,
  4) compute equal-weight MME across models,
  5) compute T_bg and HLI, apply land mask, save PNG + NetCDF per season.

Outputs (in OUTDIR):
- PNG:   HLI_map_<SEASON>_<YEAR_START>-<YEAR_END>.png
- NetCDF: projection_HLI_data_<SEASON>_<YEAR_START>_<YEAR_END>.nc
"""

# ----------------------- USER SETTINGS -----------------------
BASE_DIR    = r"your path"  # <-- set this
YEAR_START  = 2051
YEAR_END    = 2075
OUTDIR      = r"your path"                         # <-- set this
SHOW_FIGS   = False   # True to display plots in interactive sessions
# -------------------------------------------------------------

import os
import re
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import netCDF4 as nc
from netCDF4 import num2date
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

from shapely.ops import unary_union
from shapely.geometry import mapping
from rasterio import features
from affine import Affine


SEASONS = {'DJF': (12, 1, 2), 'MAM': (3, 4, 5), 'JJA': (6, 7, 8), 'SON': (9, 10, 11)}


# ----------------------- QC corrections -----------------------

def apply_corrections(arr, varname):
    """Basic QC/physical limits per variable."""
    arr = np.where(arr >= 1e20, np.nan, arr)
    if varname == 'tas':
        arr = np.where(arr < 0, np.nan, arr)           # < 0 K is unphysical
    elif varname == 'rsds':
        arr = np.where(arr < 0, 0, arr)                # no negative shortwave
        arr = np.clip(arr, 0, 1000)                    # reasonable cap
    elif varname == 'hurs':
        arr = np.where(arr < 0, 0, arr)
        arr = np.where(arr > 100, 100, arr)
    elif varname == 'sfcWind':
        arr = np.where(arr < 0, 0, arr)
    return arr


# ----------------------- HLI components -----------------------

def calculate_T_bg(tas, rsds):
    """Black Globe Temperature (°C) from tas (K) and rsds (W/m²)."""
    eps = 1e-8
    tas_c = tas - 273.15
    tas_c_sqrt = np.sqrt(np.maximum(tas_c, 0))
    rsds = np.clip(rsds, eps, 1000)
    T_bg = 1.33 * tas_c - 2.65 * tas_c_sqrt + 3.21 * np.log(rsds + 1) + 3.5
    return np.clip(T_bg, -100, 100)


def calculate_HLI(hurs, T_bg, sfcWind):
    """Heat Load Index (dimensionless), blending around T_bg=25°C."""
    S = 1 / (1 + np.exp(-((T_bg - 25) / 2.25)))
    HLI_hi = 8.62 + 0.38*hurs + 1.55*T_bg - 0.5*sfcWind + np.exp(2.4 - sfcWind)
    HLI_lo = 10.66 + 0.28*hurs + 1.3*T_bg - sfcWind
    HLI = S * HLI_hi + (1 - S) * HLI_lo
    return np.clip(HLI, 0, None)


# ----------------------- Land mask (Natural Earth via Cartopy) -----------------------

def create_land_mask(lats, lons, ne_res="110m"):
    """
    Build a 1/0 land/ocean mask on the provided regular lat/lon grid using
    Natural Earth 'land' polygons via Cartopy.
    'lats' and 'lons' are cell centers in ascending order.
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


def apply_land_mask(data, land_mask):
    """Keep land (1), set oceans (0) to NaN."""
    return np.where(land_mask == 1, data, np.nan)


# ----------------------- Plotting -----------------------

def plot_HLI_on_map(HLI_data, lats, lons, title, show=True, out_png=None):
    fig = plt.figure(figsize=(18, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='-')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 14, 'color': 'black'}
    gl.ylabel_style = {'size': 14, 'color': 'black'}

    # ensure ascending latitude
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        HLI_data = HLI_data[::-1, :]

    vmax = float(np.nanmax(HLI_data))
    levels = np.arange(0, vmax + 5, 3)

    cmap = plt.colormaps.get_cmap('RdBu_r')
    try:
        cmap = cmap.with_extremes(under='white')
    except Exception:
        try: cmap.set_under('white')
        except Exception: pass

    cs = ax.contourf(lons, lats, np.ma.masked_invalid(HLI_data),
                     levels=levels, cmap=cmap, extend='neither', transform=ccrs.PlateCarree())

    cbar = plt.colorbar(cs, orientation='horizontal', pad=0.1)
    cbar.set_label('Heat Load Index (HLI)', fontsize=18, labelpad=15)
    cbar.ax.tick_params(labelsize=12)

    plt.title(title, fontsize=20)
    plt.tight_layout(pad=2)

    if show:
        plt.show()
    if out_png:
        fig.savefig(out_png, dpi=450)
        print(f"Saved figure → {out_png}")
    plt.close(fig)


# ----------------------- Helpers -----------------------

def list_nc_files(folder):
    """List all NetCDF files in a folder (sorted)."""
    return [str(Path(folder)/fn) for fn in sorted(os.listdir(folder)) if fn.endswith(".nc")]


def get_lat_lon_from_any_file(folder):
    """Read lat/lon from the first NetCDF in folder."""
    for fp in list_nc_files(folder):
        with nc.Dataset(fp) as ds:
            return ds.variables['lat'][:], ds.variables['lon'][:]
    return None, None


def infer_model_name(filename):
    """
    Infer model name from typical NEX-GDDP-CMIP6 filename.
    Example: hurs_day_ACCESS-CM2_ssp245_r1i1p1f1_gn_2051_v1.1.nc -> ACCESS-CM2
    """
    name = os.path.basename(filename)
    m = re.search(r'^[a-zA-Z]+_day_([A-Za-z0-9\-]+)_', name)
    return m.group(1) if m else os.path.splitext(name)[0]


# ----------------------- MME seasonal means (day-weighted) -----------------------

def seasonal_MME_for_variable(files, varname, y0, y1):
    """
    Equal-weight MME seasonal means for one variable over [y0, y1].

    Per model:
      - read all daily files for that model (from 'files' list),
      - keep only timestamps with year in [y0, y1],
      - accumulate daily values by season (DJF/MAM/JJA/SON),
      - seasonal mean = sum(values)/count(days), honoring each file's calendar.

    Across models:
      - simple mean (equal weights).
    """
    if not files:
        return {s: None for s in SEASONS}

    # group filepaths by model name
    by_model = defaultdict(list)
    for fp in files:
        by_model[infer_model_name(fp)].append(fp)

    mme_tot = {s: None for s in SEASONS}
    mme_cnt = {s: 0 for s in SEASONS}
    ref_lat, ref_lon = None, None

    for model, fplist in by_model.items():
        mod_tot = {s: None for s in SEASONS}
        mod_cnt = {s: 0 for s in SEASONS}

        for fp in fplist:
            with nc.Dataset(fp, "r") as ds:
                data = ds.variables[varname][:]         # (time, lat, lon)
                lats = ds.variables['lat'][:]
                lons = ds.variables['lon'][:]

                # grid consistency
                if ref_lat is None:
                    ref_lat, ref_lon = lats, lons
                else:
                    if (lats.shape != ref_lat.shape or lons.shape != ref_lon.shape or
                        not np.allclose(lats, ref_lat) or not np.allclose(lons, ref_lon)):
                        raise ValueError(f"Grid mismatch in {fp} (model {model})")

                # decode time (calendar-aware)
                tvar = ds.variables['time']
                times = num2date(tvar[:], units=tvar.units,
                                 calendar=tvar.calendar if 'calendar' in tvar.ncattrs() else 'standard')

                # QC
                data = apply_corrections(data, varname)

                # accumulate daily -> seasonal, filtering year range
                for i, ts in enumerate(times):
                    y = ts.year
                    if y < y0 or y > y1:
                        continue
                    m = ts.month
                    for season, months in SEASONS.items():
                        if m in months:
                            slice_ = data[i]
                            if mod_tot[season] is None:
                                mod_tot[season] = np.zeros_like(slice_, dtype=np.float64)
                            valid = ~np.isnan(slice_)
                            mod_tot[season][valid] += slice_[valid]
                            mod_cnt[season] += 1
                            break

        # finalize per-model seasonal means and accumulate to MME
        for season in SEASONS:
            if mod_cnt[season] > 0:
                mean_ = mod_tot[season] / mod_cnt[season]
                if mme_tot[season] is None:
                    mme_tot[season] = np.zeros_like(mean_, dtype=np.float64)
                mme_tot[season] += mean_
                mme_cnt[season] += 1

    # finalize equal-weight MME
    out = {}
    for season in SEASONS:
        out[season] = mme_tot[season] / mme_cnt[season] if mme_cnt[season] > 0 else None
    return out


# ----------------------- Main -----------------------

def main():
    t0 = time.time()
    base = Path(BASE_DIR)
    out = Path(OUTDIR)
    out.mkdir(parents=True, exist_ok=True)

    # variable folders (must exist)
    tas_dir  = base / "tas"
    hurs_dir = base / "hurs"
    wind_dir = base / "sfcWind"
    rsds_dir = base / "rsds"
    for d in (tas_dir, hurs_dir, wind_dir, rsds_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"Missing folder: {d}")

    # list files per variable
    tas_files  = list_nc_files(tas_dir)
    hurs_files = list_nc_files(hurs_dir)
    wind_files = list_nc_files(wind_dir)
    rsds_files = list_nc_files(rsds_dir)
    if not all([tas_files, hurs_files, wind_files, rsds_files]):
        raise FileNotFoundError("One or more variable folders have no NetCDF files.")

    # seasonal means (per variable) -> equal-weight MME
    print(f"Calculating MME seasonal means for {YEAR_START}-{YEAR_END}")
    tas_season  = seasonal_MME_for_variable(tas_files,  "tas",     YEAR_START, YEAR_END)
    hurs_season = seasonal_MME_for_variable(hurs_files, "hurs",    YEAR_START, YEAR_END)
    wind_season = seasonal_MME_for_variable(wind_files, "sfcWind", YEAR_START, YEAR_END)
    rsds_season = seasonal_MME_for_variable(rsds_files, "rsds",    YEAR_START, YEAR_END)

    # lat/lon from any tas file
    lats, lons = get_lat_lon_from_any_file(tas_dir)
    if lats is None or lons is None:
        raise FileNotFoundError("Could not read lat/lon from TAS folder.")

    # convert longitudes 0–360 -> −180–180 and sort
    lons_conv = (lons + 180) % 360 - 180
    sort_idx = np.argsort(lons_conv)
    lons_sorted = lons_conv[sort_idx]

    # land mask once
    land_mask = create_land_mask(lats, lons_sorted)

    for season in SEASONS.keys():
        print(f"\nComputing HLI for season {season} — years {YEAR_START}-{YEAR_END}")

        T = tas_season[season]
        H = hurs_season[season]
        W = wind_season[season]
        R = rsds_season[season]

        if any(x is None for x in (T, H, W, R)):
            print(f"Missing data for {season}. Skipping.")
            continue

        # sort along lon
        T = T[:, sort_idx]; H = H[:, sort_idx]; W = W[:, sort_idx]; R = R[:, sort_idx]

        # HLI
        Tbg = calculate_T_bg(T, R)
        HLI = calculate_HLI(H, Tbg, W)

        print(f"HLI {season}: min {np.nanmin(HLI):.2f}, max {np.nanmax(HLI):.2f}")

        # mask ocean
        HLI_land = apply_land_mask(HLI, land_mask)

        # figure
        fig_png = out / f"HLI_map_{season}_{YEAR_START}-{YEAR_END}.png"
        plot_HLI_on_map(HLI_land, lats, lons_sorted,
                        title=f"HLI {season} — {YEAR_START}-{YEAR_END}",
                        show=SHOW_FIGS, out_png=fig_png)

        # NetCDF
        nc_path = out / f"projection_HLI_data_{season}_{YEAR_START}_{YEAR_END}.nc"
        with nc.Dataset(nc_path, "w", format="NETCDF4_CLASSIC") as dso:
            dso.createDimension('lat', len(lats))
            dso.createDimension('lon', len(lons_sorted))
            vlat = dso.createVariable('lat', np.float32, ('lat',))
            vlon = dso.createVariable('lon', np.float32, ('lon',))
            vHLI = dso.createVariable('HLI', np.float32, ('lat', 'lon',),
                                      zlib=True, complevel=4, fill_value=np.float32(np.nan))
            vlat[:] = lats
            vlon[:] = lons_sorted
            vHLI[:, :] = HLI_land
            vlat.units = 'degrees_north'
            vlon.units = 'degrees_east'
            vHLI.long_name = f'Heat Load Index seasonal mean ({season}) {YEAR_START}-{YEAR_END}'
            vHLI.units = '1'
            dso.title = f'HLI seasonal mean {season} {YEAR_START}-{YEAR_END} (MME equal-weight, day-weighted)'
            dso.source = 'Derived from NEX-GDDP-CMIP6 daily tas/hurs/sfcWind/rsds'
            dso.history = 'Created by Projections_HLI_Seasonal_MME_Public.py'
            dso.Conventions = 'CF-1.8'
        print(f"Saved NetCDF → {nc_path}")

    print(f"\nDone in {time.time() - t0:.1f} s.")


# Run (Spyder-friendly)
if __name__ == "__main__":
    main()


