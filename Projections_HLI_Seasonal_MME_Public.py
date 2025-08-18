#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute seasonal Heat Load Index (HLI) projections from NEX-GDDP-CMIP6 daily data
(tas, hurs, sfcWind, rsds) using a calendar-aware, equal-weight Multi-Model
Ensemble (MME), for any SSP scenario and any year interval.

Assumptions/choices:
  - Per-model seasonal means are *day-weighted* (arithmetic mean across all days in the season),
    honoring each file's calendar.
  - The MME is an equal-weight (simple) average across models available for the period.
  - Directory layout can be either:
        BASE_DIR/<variable>/<scenario>/*.nc
    or  BASE_DIR/<scenario>/<variable>/*.nc
    (the script tries both).

Outputs (in --outdir):
  - PNG:   HLI_map_<SCENARIO>_<SEASON>_<START>-<END>.png
  - NetCDF: projection_HLI_data_<SEASON>_<START>_<END>_<SCENARIO>.nc

Examples:
  python Projections_HLI_Seasonal_MME_Generic.py \
      --base-dir /data/NEX-GDDP-CMIP6 \
      --scenario ssp585 \
      --year-start 2076 --year-end 2100 \
      --outdir ./outputs --no-show
"""

import argparse
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import netCDF4 as nc
from netCDF4 import num2date
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import geopandas as gpd
from shapely.geometry import mapping
from rasterio import features
from affine import Affine


# ----------------------- CLI -----------------------

def build_parser():
    p = argparse.ArgumentParser(description="Seasonal HLI projections (MME equal-weight, calendar-aware).")
    p.add_argument("--base-dir", required=True,
                   help="Base folder containing variable/scenario subfolders (see header).")
    p.add_argument("--scenario", required=True,
                   help="Scenario name (e.g., ssp126, ssp245, ssp370, ssp585). Case-insensitive.")
    p.add_argument("--year-start", type=int, required=True, help="Start year (inclusive).")
    p.add_argument("--year-end", type=int, required=True, help="End year (inclusive).")
    p.add_argument("--outdir", default="./outputs", help="Output directory (default: ./outputs).")
    p.add_argument("--no-show", action="store_true", help="Do not display figures (still saved to disk).")
    return p


# ----------------------- QC corrections -----------------------

def apply_corrections(current_data, variable_name):
    """Variable-specific QC/physical corrections."""
    current_data = np.where(current_data >= 1e+20, np.nan, current_data)

    if variable_name == 'tas':
        # Temperature below 0 K is unphysical -> NaN
        current_data = np.where(current_data < 0, np.nan, current_data)
    elif variable_name == 'rsds':
        # No negative shortwave radiation; clip to reasonable max
        current_data = np.where(current_data < 0, 0, current_data)
        current_data = np.clip(current_data, 0, 1000)
    elif variable_name == 'hurs':
        # Constrain relative humidity to [0,100] %
        current_data = np.where(current_data < 0, 0, current_data)
        current_data = np.where(current_data > 100, 100, current_data)
    elif variable_name == 'sfcWind':
        # No negative wind speed
        current_data = np.where(current_data < 0, 0, current_data)
    return current_data


# ----------------------- HLI components -----------------------

def calculate_T_bg(tas, rsds):
    """Black Globe Temperature (T_bg, °C) from air temperature (K) and shortwave radiation (W/m²)."""
    epsilon = 1e-8
    tas_c = tas - 273.15
    tas_c_sqrt = np.sqrt(np.maximum(tas_c, 0))
    rsds = np.clip(rsds, epsilon, 1000)
    T_bg = 1.33 * tas_c - 2.65 * tas_c_sqrt + 3.21 * np.log(rsds + 1) + 3.5
    return np.clip(T_bg, -100, 100)


def calculate_HLI(hurs, T_bg, sfcWind):
    """Heat Load Index (HLI) from RH, T_bg and wind speed; sigmoid blend around 25 °C."""
    S_BGT = 1 / (1 + np.exp(-((T_bg - 25) / 2.25)))
    HLI_above_25 = 8.62 + (0.38 * hurs) + (1.55 * T_bg) - (0.5 * sfcWind) + np.exp(2.4 - sfcWind)
    HLI_below_25 = 10.66 + (0.28 * hurs) + (1.3 * T_bg) - sfcWind
    HLI = S_BGT * HLI_above_25 + (1 - S_BGT) * HLI_below_25
    return np.clip(HLI, 0, None)


# ----------------------- Land mask -----------------------

def create_land_mask(lats, lons):
    """Create a land mask (1 land, 0 ocean) on the given regular lat/lon grid."""
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    land_geom = [mapping(world.unary_union)]

    x_res = (lons[-1] - lons[0]) / (len(lons) - 1)
    y_res = (lats[-1] - lats[0]) / (len(lats) - 1)
    transform = Affine.translation(lons[0] - x_res / 2, lats[0] - y_res / 2) * Affine.scale(x_res, y_res)

    mask = features.rasterize(
        shapes=land_geom,
        out_shape=(len(lats), len(lons)),
        transform=transform,
        fill=0,          # Ocean
        default_value=1, # Land
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
    gl.xlabel_style = {'size': 16, 'color': 'black'}
    gl.ylabel_style = {'size': 16, 'color': 'black'}

    if lats[0] > lats[-1]:
        lats = lats[::-1]
        HLI_data = HLI_data[::-1, :]

    HLI_max = np.nanmax(HLI_data)
    HLI_min = 0
    print(f"Using HLI max: {HLI_max} for color scale.")

    HLI_data = np.ma.masked_invalid(HLI_data)
    levels = np.arange(HLI_min, HLI_max + 5, 3)

    cmap = plt.colormaps.get_cmap('RdBu_r')
    try:
        cmap = cmap.with_extremes(under='white')
    except Exception:
        try:
            cmap.set_under('white')
        except Exception:
            pass

    cs = ax.contourf(lons, lats, HLI_data, levels=levels, cmap=cmap, extend='neither', transform=ccrs.PlateCarree())

    cbar = plt.colorbar(cs, orientation='horizontal', pad=0.1)
    cbar.set_label('Heat Load Index (HLI)', fontsize=22, labelpad=20)
    cbar.ax.tick_params(labelsize=15)

    plt.title(title, fontsize=22)
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(top=0.95, bottom=0.1)

    if show:
        plt.show()
    if out_png is not None:
        fig.savefig(out_png, dpi=600)
        print(f"Saved figure → {out_png}")
    plt.close(fig)


# ----------------------- Helpers -----------------------

def get_var_dir(base_dir, variable, scenario):
    """
    Resolve data directory for a given variable and scenario. Tries both:
      base_dir/variable/scenario  and  base_dir/scenario/variable
    """
    cand1 = Path(base_dir) / variable / scenario
    cand2 = Path(base_dir) / scenario / variable
    if cand1.is_dir():
        return str(cand1)
    if cand2.is_dir():
        return str(cand2)
    raise FileNotFoundError(f"No data directory for variable={variable} scenario={scenario} under {base_dir}")


def get_lat_lon_from_directory(directory_path):
    """Extract lat/lon arrays from the first NetCDF in a directory."""
    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith(".nc"):
            with nc.Dataset(os.path.join(directory_path, filename), 'r') as data:
                lats = data.variables["lat"][:]
                lons = data.variables["lon"][:]
                return lats, lons
    return None, None


def infer_model_name(filename: str) -> str:
    """
    Infer a model name from a typical NEX-GDDP-CMIP6 filename.

    Example: hurs_day_ACCESS-CM2_historical_r1i1p1f1_gn_1985_v1.1.nc  -> ACCESS-CM2
    """
    name = os.path.basename(filename)
    m = re.search(r'^[a-zA-Z]+_day_([A-Za-z0-9\-]+)_(?:historical|ssp\d[\.\-]?\d?)_', name, re.IGNORECASE)
    if m:
        return m.group(1)
    # fallbacks
    for pat in (r'_[A-Za-z]+-day_([A-Za-z0-9\-]+)_',
                r'^[a-zA-Z]+_([A-Za-z0-9\-]+)_',
                r'_(?:ssp\d[\.\-]?\d?)_([A-Za-z0-9\-]+)_',
                r'_([A-Za-z0-9\-]+)_ssp'):
        m = re.search(pat, name, re.IGNORECASE)
        if m:
            return m.group(1)
    return os.path.splitext(name)[0]


# ----------------------- MME seasonal means (day-weighted, calendar-aware) -----------------------

def seasonal_MME_for_variable(var_dir: str, variable_name: str, year_start: int, year_end: int):
    """
    Equal-weight MME seasonal means for one variable over [year_start, year_end].

    Per model:
      - read all daily files,
      - keep only timestamps within [year_start, year_end],
      - accumulate daily values by season (DJF/MAM/JJA/SON) using months (Dec,Jan,Feb etc.)
        with DECEMBER OF THE SAME YEAR convention,
      - seasonal mean = (sum over days in season) / (number of days in season present in files).

    Across models:
      - simple average (equal weights) of per-model seasonal means.
    """
    seasons = {'DJF': (12, 1, 2), 'MAM': (3, 4, 5), 'JJA': (6, 7, 8), 'SON': (9, 10, 11)}

    # group files by model
    files_by_model = defaultdict(list)
    for filename in sorted(os.listdir(var_dir)):
        if filename.endswith(".nc"):
            model = infer_model_name(filename)
            files_by_model[model].append(os.path.join(var_dir, filename))

    if not files_by_model:
        return {s: None for s in seasons}

    mme_totals = {s: None for s in seasons}
    mme_counts = {s: 0 for s in seasons}

    ref_lat, ref_lon = None, None

    for model, filelist in files_by_model.items():
        model_totals = {s: None for s in seasons}
        model_counts = {s: 0 for s in seasons}

        for fp in sorted(filelist):
            with nc.Dataset(fp, 'r') as ds:
                data = ds.variables[variable_name][:]  # time, lat, lon
                lats = ds.variables['lat'][:]
                lons = ds.variables['lon'][:]

                # grid consistency check
                if ref_lat is None:
                    ref_lat, ref_lon = lats, lons
                else:
                    if (lats.shape != ref_lat.shape or lons.shape != ref_lon.shape or
                        not np.allclose(lats, ref_lat) or not np.allclose(lons, ref_lon)):
                        raise ValueError(f"Grid mismatch in {fp} for model {model}")

                # time axis
                tvar = ds.variables['time']
                tunits = tvar.units
                tcal = tvar.calendar if 'calendar' in tvar.ncattrs() else 'standard'
                times = num2date(tvar[:], units=tunits, calendar=tcal)

                # QC
                data = apply_corrections(data, variable_name)

                # accumulate by season for years in [year_start, year_end]
                for idx, ts in enumerate(times):
                    y = ts.year
                    if y < year_start or y > year_end:
                        continue
                    m = ts.month
                    for season, months in seasons.items():
                        if m in months:
                            slice_ = data[idx]
                            if model_totals[season] is None:
                                model_totals[season] = np.zeros_like(slice_, dtype=np.float64)
                            valid = ~np.isnan(slice_)
                            model_totals[season][valid] += slice_[valid]
                            model_counts[season] += 1
                            break

        # finalize per-model seasonal means and add to MME accumulators
        for season in seasons:
            if model_counts[season] > 0:
                model_mean = model_totals[season] / model_counts[season]
                if mme_totals[season] is None:
                    mme_totals[season] = np.zeros_like(model_mean, dtype=np.float64)
                mme_totals[season] += model_mean
                mme_counts[season] += 1

    # finalize equal-weight MME
    seasonal_mme = {}
    for season in seasons:
        if mme_counts[season] > 0:
            seasonal_mme[season] = mme_totals[season] / mme_counts[season]
        else:
            seasonal_mme[season] = None

    return seasonal_mme


# ----------------------- Main -----------------------

def main():
    args = build_parser().parse_args()
    base_dir = args.base_dir
    scenario = args.scenario.strip().lower()
    year_start = args.year_start
    year_end = args.year_end
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Resolve variable directories (accept 2 layouts)
    try:
        tas_dir = get_var_dir(base_dir, "tas", scenario)
        hurs_dir = get_var_dir(base_dir, "hurs", scenario)
        wind_dir = get_var_dir(base_dir, "sfcWind", scenario)
        rsds_dir = get_var_dir(base_dir, "rsds", scenario)
    except FileNotFoundError as e:
        raise SystemExit(str(e))

    # Compute seasonal MME means (calendar-aware, day-weighted) for each variable
    print(f"Calculating MME seasonal means for 'tas' in {scenario} [{year_start}-{year_end}]")
    tas_seasonal = seasonal_MME_for_variable(tas_dir, "tas", year_start, year_end)

    print(f"Calculating MME seasonal means for 'hurs' in {scenario} [{year_start}-{year_end}]")
    hurs_seasonal = seasonal_MME_for_variable(hurs_dir, "hurs", year_start, year_end)

    print(f"Calculating MME seasonal means for 'sfcWind' in {scenario} [{year_start}-{year_end}]")
    sfc_seasonal = seasonal_MME_for_variable(wind_dir, "sfcWind", year_start, year_end)

    print(f"Calculating MME seasonal means for 'rsds' in {scenario} [{year_start}-{year_end}]")
    rsds_seasonal = seasonal_MME_for_variable(rsds_dir, "rsds", year_start, year_end)

    # Lat/lon from any TAS file
    lats, lons = get_lat_lon_from_directory(tas_dir)
    if lats is None or lons is None:
        raise FileNotFoundError(f"No .nc files found in {tas_dir}")

    # Convert longitudes 0–360 -> −180–180 and sort index
    lons_conv = (lons + 180) % 360 - 180
    sorted_idx = np.argsort(lons_conv)
    lons_sorted = lons_conv[sorted_idx]

    # Land mask (create once on final lon grid)
    land_mask = create_land_mask(lats, lons_sorted)

    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    for season in seasons:
        print(f"\nComputing HLI for season {season} — {scenario} {year_start}-{year_end}")

        tas_avg = tas_seasonal[season]
        hurs_avg = hurs_seasonal[season]
        sfc_avg = sfc_seasonal[season]
        rsds_avg = rsds_seasonal[season]

        if any(arr is None for arr in [tas_avg, hurs_avg, sfc_avg, rsds_avg]):
            print(f"Missing data for season {season}. Skipping.")
            continue

        # sort along longitude axis consistently
        tas_avg = tas_avg[:, sorted_idx]
        hurs_avg = hurs_avg[:, sorted_idx]
        sfc_avg = sfc_avg[:, sorted_idx]
        rsds_avg = rsds_avg[:, sorted_idx]

        # Compute T_bg and HLI
        T_bg = calculate_T_bg(tas_avg, rsds_avg)
        HLI = calculate_HLI(hurs_avg, T_bg, sfc_avg)

        print(f"HLI {season}: min {np.nanmin(HLI):.2f}, max {np.nanmax(HLI):.2f}")

        # Apply land mask
        HLI_masked = apply_land_mask(HLI, land_mask)

        # Plot and save figure
        fig_path = outdir / f"HLI_map_{scenario.upper()}_{season}_{year_start}-{year_end}.png"
        title = f"HLI {season} — {scenario.upper()} {year_start}-{year_end}"
        plot_HLI_on_map(HLI_masked, lats, lons_sorted, title=title,
                        show=not args.no_show, out_png=fig_path)

        # Save seasonal HLI to NetCDF
        nc_path = outdir / f"projection_HLI_data_{season}_{year_start}_{year_end}_{scenario}.nc"
        with nc.Dataset(nc_path, 'w', format='NETCDF4_CLASSIC') as ds_out:
            ds_out.createDimension('lat', len(lats))
            ds_out.createDimension('lon', len(lons_sorted))

            latitudes = ds_out.createVariable('lat', np.float32, ('lat',))
            longitudes = ds_out.createVariable('lon', np.float32, ('lon',))
            HLI_var = ds_out.createVariable('HLI', np.float32, ('lat', 'lon',),
                                            zlib=True, complevel=4, fill_value=np.float32(np.nan))

            latitudes[:] = lats
            longitudes[:] = lons_sorted
            HLI_var[:, :] = HLI_masked

            # metadata
            latitudes.units = 'degrees_north'
            longitudes.units = 'degrees_east'
            HLI_var.long_name = f'Heat Load Index seasonal mean ({season}) — {scenario.upper()} {year_start}-{year_end}'
            HLI_var.units = '1'  # dimensionless index
            ds_out.title = f'HLI {seasonal_title(season)} {scenario.upper()} {year_start}-{year_end} (MME equal-weight)'
            ds_out.source = 'Derived from NEX-GDDP-CMIP6 daily variables tas/hurs/sfcWind/rsds'
            ds_out.history = 'Created by Projections_HLI_Seasonal_MME_Generic.py'
            ds_out.Conventions = 'CF-1.8'

        print(f"Saved NetCDF → {nc_path}")

    print(f"\nDone in {time.time() - t0:.1f} s.")


def seasonal_title(season):
    return {'DJF': 'DJF', 'MAM': 'MAM', 'JJA': 'JJA', 'SON': 'SON'}.get(season, season)


if __name__ == "__main__":
    main()
