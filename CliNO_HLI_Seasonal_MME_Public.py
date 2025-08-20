#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute baseline (1985–2014) seasonal Heat Load Index (HLI) from NEX-GDDP-CMIP6 daily data
(tas, hurs, sfcWind, rsds) using a calendar-consistent, equal-weight Multi-Model Ensemble (MME).

Pipeline:
  1) For each model: accumulate DAILY values into SEASONAL means (DJF/MAM/JJA/SON),
     dividing by the actual number of contributing days (day-weighted; honors each file’s calendar).
  2) Across models: equal-weight average (each model weight = 1/N).
  3) Compute black-globe temperature (T_bg) and HLI from the seasonal fields.
  4) Apply land mask, plot, and save seasonal HLI to NetCDF.

Outputs (in --outdir):
  - PNG:   HLI_map_HLI Season <SEASON>.png
  - NetCDF: average_HLI_data_<SEASON>_1985_2014.nc

Example:
  python CliNO_HLI_Seasonal_MME.py --base-dir ./CliNO_nc_files --outdir ./outputs --no-show
"""

import argparse
import os
import time
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import netCDF4 as nc
from netCDF4 import num2date
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from shapely.geometry import mapping
from shapely.ops import unary_union
from cartopy.io import shapereader as shpreader
from rasterio import features
from affine import Affine


# ----------------------- CLI -----------------------

def build_parser():
    p = argparse.ArgumentParser(description="Baseline seasonal HLI with equal-weight MME (day-weighted).")
    p.add_argument(
        "--base-dir",
        default="your path",
        help="Base dir containing tas/, hurs/, sfcWind/, rsds/ (default: ./CliNO_nc_files)"
    )
    p.add_argument(
        "--outdir",
        default="your path",
        help="Output directory for NetCDF and PNG (default: ./outputs)"
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display figures (still saved to disk)."
    )
    return p


# ----------------------- QC corrections -----------------------

def apply_corrections(current_data, variable_name):
    """Variable-specific QC/physical corrections."""
    current_data = np.where(current_data >= 1e+20, np.nan, current_data)

    if variable_name == 'tas':
        # Temperature below 0 K is unphysical
        current_data = np.where(current_data < 0, np.nan, current_data)
    elif variable_name == 'rsds':
        # No negative shortwave radiation; clip to a reasonable max
        current_data = np.where(current_data < 0, 0, current_data)
        current_data = np.clip(current_data, 0, 1000)
    elif variable_name == 'hurs':
        # Constrain relative humidity to [0, 100] %
        current_data = np.where(current_data < 0, 0, current_data)
        current_data = np.where(current_data > 100, 100, current_data)
    elif variable_name == 'sfcWind':
        # No negative wind speed
        current_data = np.where(current_data < 0, 0, current_data)

    return current_data


# ----------------------- HLI components -----------------------

def calculate_T_bg(tas, rsds):
    """Black Globe Temperature (T_bg, °C) from air temperature (K) and shortwave radiation (W/m²)."""
    epsilon = 1e-8  # avoid log(0)
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

def create_land_mask(lats, lons, ne_res="110m"):
    """
    Create a 1/0 land/ocean mask on the provided lat/lon grid
    using Natural Earth shapefiles via Cartopy.
    lats and lons are the centers of the grid cells, in ascending order.
    """
    # Read the "land" dataset from Natural Earth
    shp_path = shpreader.natural_earth(resolution=ne_res, category="physical", name="land")
    reader = shpreader.Reader(shp_path)
    geoms = list(reader.geometries())
    land_union = unary_union(geoms)  # merge all polygons (continents and islands)
    land_geom = [mapping(land_union)]

    # Build affine transform for the regular grid
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

    # Ensure ascending latitude order for plotting
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


# ----------------------- I/O helpers -----------------------

def get_lat_lon_from_dataset(directory_path):
    """Extract lat/lon arrays from the first NetCDF in a directory."""
    for filename in os.listdir(directory_path):
        if filename.endswith(".nc"):
            with nc.Dataset(os.path.join(directory_path, filename), 'r') as data:
                lats = data.variables["lat"][:]
                lons = data.variables["lon"][:]
                return lats, lons
    return None, None


# ----------------------- Equal-weight MME (day-weighted seasons) -----------------------

def infer_model_name(filename: str) -> str:
    """
    Infer model name from a NEX-GDDP-CMIP6-like filename.

    Example:
      hurs_day_ACCESS-CM2_historical_r1i1p1f1_gn_1985_v1.1.nc  -> ACCESS-CM2
    """
    name = os.path.basename(filename)
    m = re.search(r'^[a-zA-Z]+_day_([A-Za-z0-9\-]+)_(?:historical|ssp\d[\.\-]?\d?)_', name)
    if m:
        return m.group(1)
    for pat in (r'_[A-Za-z]+-day_([A-Za-z0-9\-]+)_',
                r'^[a-zA-Z]+_([A-Za-z0-9\-]+)_',
                r'_(?:ssp\d[\.\-]?\d?)_([A-Za-z0-9\-]+)_',
                r'_([A-Za-z0-9\-]+)_ssp'):
        m = re.search(pat, name)
        if m:
            return m.group(1)
    return os.path.splitext(name)[0]


def calculate_seasonal_MME_in_directory(directory_path: str, variable_name: str):
    """
    Equal-weight MME with SEASONAL means computed as day-weighted averages.

    Per model:
      - read all daily files,
      - sum daily values for each season (DJF/MAM/JJA/SON),
      - divide by the actual number of contributing days (respects each file's calendar).

    Across models:
      - simple mean (each model weight = 1/N).

    Returns: dict season -> 2D array (lat, lon)
    """
    seasons = {'DJF': (12, 1, 2), 'MAM': (3, 4, 5), 'JJA': (6, 7, 8), 'SON': (9, 10, 11)}

    # Group files by model
    files_by_model = defaultdict(list)
    for filename in os.listdir(directory_path):
        if filename.endswith(".nc"):
            model = infer_model_name(filename)
            files_by_model[model].append(os.path.join(directory_path, filename))

    if not files_by_model:
        return {s: None for s in seasons}

    mme_totals = {s: None for s in seasons}
    mme_counts = {s: 0 for s in seasons}

    ref_lat, ref_lon = None, None

    # Per-model seasonal means (day-weighted)
    for model, filelist in files_by_model.items():
        model_totals = {s: None for s in seasons}
        model_counts = {s: 0 for s in seasons}

        filelist = sorted(filelist)

        for fp in filelist:
            with nc.Dataset(fp, 'r') as ds:
                data = ds.variables[variable_name][:]  # shape: time, lat, lon
                lats = ds.variables['lat'][:]
                lons = ds.variables['lon'][:]

                # Grid consistency check
                if ref_lat is None:
                    ref_lat, ref_lon = lats, lons
                else:
                    if (lats.shape != ref_lat.shape or lons.shape != ref_lon.shape or
                        not np.allclose(lats, ref_lat) or not np.allclose(lons, ref_lon)):
                        raise ValueError(f"Grid mismatch in {fp} for model {model}")

                # Decode time with file's native calendar
                tvar = ds.variables['time']
                tunits = tvar.units
                tcal = tvar.calendar if 'calendar' in tvar.ncattrs() else 'standard'
                times = num2date(tvar[:], units=tunits, calendar=tcal)

                # QC corrections
                data = apply_corrections(data, variable_name)

                # Accumulate daily -> seasonal (day-weighted)
                for idx, ts in enumerate(times):
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

        # Finalize per-model seasonal means and add to MME
        for season in seasons:
            if model_counts[season] > 0:
                model_mean = model_totals[season] / model_counts[season]
                if mme_totals[season] is None:
                    mme_totals[season] = np.zeros_like(model_mean, dtype=np.float64)
                mme_totals[season] += model_mean
                mme_counts[season] += 1

    # Finalize equal-weight MME
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
    base_directory = args.base_dir
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Variable folders
    temp_directory = os.path.join(base_directory, "tas")
    humidity_directory = os.path.join(base_directory, "hurs")
    wind_directory = os.path.join(base_directory, "sfcWind")
    radiation_directory = os.path.join(base_directory, "rsds")

    # Compute day-weighted, equal-weight MME seasonal means for each variable
    print("Calculating MME seasonal means for 'tas' [day-weighted]")
    tas_seasonal = calculate_seasonal_MME_in_directory(temp_directory, "tas")

    print("Calculating MME seasonal means for 'hurs' [day-weighted]")
    hurs_seasonal = calculate_seasonal_MME_in_directory(humidity_directory, "hurs")

    print("Calculating MME seasonal means for 'sfcWind' [day-weighted]")
    sfcWind_seasonal = calculate_seasonal_MME_in_directory(wind_directory, "sfcWind")

    print("Calculating MME seasonal means for 'rsds' [day-weighted]")
    rsds_seasonal = calculate_seasonal_MME_in_directory(radiation_directory, "rsds")

    # Get lat/lon from any file in temp_directory
    lats, lons = get_lat_lon_from_dataset(temp_directory)
    if lats is None or lons is None:
        raise FileNotFoundError(f"No .nc files found in {temp_directory}")

    # Convert longitudes 0–360 -> −180–180 and sort index
    lons_converted = (lons + 180) % 360 - 180
    sorted_indices = np.argsort(lons_converted)
    lons_sorted = lons_converted[sorted_indices]

    # Land mask (create once on final lon grid)
    land_mask = create_land_mask(lats, lons_sorted)

    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    for season in seasons:
        print(f"\nCalculating HLI for season {season}")

        tas_avg = tas_seasonal[season]
        hurs_avg = hurs_seasonal[season]
        sfcWind_avg = sfcWind_seasonal[season]
        rsds_avg = rsds_seasonal[season]

        if any(data is None for data in [tas_avg, hurs_avg, sfcWind_avg, rsds_avg]):
            print(f"Missing data for season {season}. Skipping this season.")
            continue

        # Sort along longitude axis consistently
        tas_avg_sorted = tas_avg[:, sorted_indices]
        hurs_avg_sorted = hurs_avg[:, sorted_indices]
        sfcWind_avg_sorted = sfcWind_avg[:, sorted_indices]
        rsds_avg_sorted = rsds_avg[:, sorted_indices]

        # Compute T_bg and HLI
        T_bg = calculate_T_bg(tas_avg_sorted, rsds_avg_sorted)
        HLI = calculate_HLI(hurs_avg_sorted, T_bg, sfcWind_avg_sorted)

        print(f"HLI for {season}: min {np.nanmin(HLI)}, max {np.nanmax(HLI)}")

        # Apply land mask
        HLI_masked = apply_land_mask(HLI, land_mask)

        # Plot and save figure
        fig_path = outdir / f'HLI_map_HLI Season {season}.png'
        plot_HLI_on_map(
            HLI_masked, lats, lons_sorted, title=f"HLI Season {season}",
            show=not args.no_show, out_png=fig_path
        )

        # Save seasonal HLI to NetCDF
        output_filename = outdir / f'average_HLI_data_{season}_1985_2014.nc'
        with nc.Dataset(output_filename, 'w', format='NETCDF4_CLASSIC') as new_dataset:
            new_dataset.createDimension('lat', len(lats))
            new_dataset.createDimension('lon', len(lons_sorted))
            latitudes = new_dataset.createVariable('lat', np.float32, ('lat',))
            longitudes = new_dataset.createVariable('lon', np.float32, ('lon',))
            HLI_var = new_dataset.createVariable(
                'HLI', np.float32, ('lat', 'lon',), zlib=True, complevel=4, fill_value=np.float32(np.nan)
            )

            latitudes[:] = lats
            longitudes[:] = lons_sorted
            HLI_var[:, :] = HLI_masked

            # Minimal, useful metadata
            latitudes.units = 'degrees_north'
            longitudes.units = 'degrees_east'
            HLI_var.long_name = 'Heat Load Index (seasonal mean, baseline 1985–2014; equal-weight MME, day-weighted)'
            HLI_var.units = '1'  # dimensionless index
            new_dataset.title = f'HLI seasonal mean {season} (1985–2014, MME equal-weight, day-weighted)'
            new_dataset.source = 'Derived from NEX-GDDP-CMIP6 daily variables tas/hurs/sfcWind/rsds'
            new_dataset.history = 'Created by CliNO_HLI_Seasonal_MME.py [day-weighted]'
            new_dataset.Conventions = 'CF-1.8'

            print(f"Saved NetCDF → {output_filename}")

    elapsed_time = time.time() - start_time
    print(f"\nProcessing complete. Time elapsed: {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    main()



