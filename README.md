# HLI-Seasonal-and-Projections-Anomalies
Python scripts for seasonal HLI baseline, projections, and anomalies

This repository provides the Python scripts used to process climate model outputs and compute the Heat Load Index (HLI) under historical and future climate scenarios (CMIP6).

Repository contents

CliNO_HLI_Seasonal_MME_Public.py
Computes the baseline (1985–2014) seasonal climatology of HLI from 27 CMIP6 GCMs, producing a multi-model ensemble (MME).

Projections_HLI_Seasonal_MME_Public.py
Computes seasonal HLI climatology for each climate scenario (SSP) and 25-year projection window (e.g., 2026–2050, 2051–2075, 2076–2100), using the same ensemble approach.

HLI_Anomalies_Panel_Public.py
Generates anomaly maps (projection minus baseline) and assembles multi-scenario seasonal panels for visualization.

Usage

Each script is intended to be run separately.
Typical workflow:

Run CliNO_HLI_Seasonal_MME_Public.py to create the baseline climatology files.

Run Projections_HLI_Seasonal_MME_Public.py for each scenario and time window.

Run HLI_Anomalies_Panel_Public.py to generate seasonal anomaly maps across scenarios.

Inputs and directory paths are configurable through command-line arguments or script variables.
The scripts assume CMIP6 NetCDF input files structured by model, scenario, and time period.

Requirements

Python 3.8+

Libraries: numpy, netCDF4, matplotlib, cartopy, geopandas, rasterio
