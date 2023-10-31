import os
import re
import sys
import time
import glob
import shutil
import fnmatch
from pathlib import Path
#
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
from netCDF4 import Dataset
#
import argparse
#
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.basemap import Basemap
import cartopy
#
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
common_files = ['lws.ssp.landwaterstorage', 'ocean.tlm.sterodynamics']
local_files  =  'k14vlm.kopp14.verticallandmotion'
#
WF_file_patterns = {
    "wf1e": ['emuGrIS.emulandice.GrIS', 'emuAIS.emulandice.AIS', 'emuglaciers.emulandice.glaciers'],
    "wf2e": ['emuGrIS.emulandice.GrIS', 'larmip.larmip.AIS', 'emuglaciers.emulandice.glaciers'],
    "wf3e": ['emuGrIS.emulandice.GrIS', 'deconto21.deconto21.AIS_AIS', 'emuglaciers.emulandice.glaciers'],
    #
    "wf1f": ['GrIS1f.FittedISMIP.GrIS_GIS', 'ar5AIS.ipccar5.icesheets_AIS', 'ar5glaciers.ipccar5.glaciers'],
    "wf2f": ['GrIS1f.FittedISMIP.GrIS_GIS', 'larmip.larmip.AIS', 'ar5glaciers.ipccar5.glaciers'],
    "wf3f": ['GrIS1f.FittedISMIP.GrIS_GIS', 'deconto21.deconto21.AIS_AIS', 'ar5glaciers.ipccar5.glaciers'],
    "wf4": ['bamber19.bamber19.icesheets_GIS', 'bamber19.bamber19.icesheets_AIS', 'ar5glaciers.ipccar5.glaciers']
}
PB_file_patterns = {
    "pb_1e": ['emuAIS.emulandice.AIS', 'larmip.larmip.AIS', 'emuGrIS.emulandice.GrIS', 'emuglaciers.emulandice.glaciers'],
    "pb_2e": ['emuAIS.emulandice.AIS', 'larmip.larmip.AIS', 'deconto21.deconto21.AIS_AIS', 'bamber19.bamber19.icesheets_AIS',
              'emuGrIS.emulandice.GrIS', 'bamber19.bamber19.icesheets_GIS', 'emuglaciers.emulandice.glaciers', 'ar5glaciers.ipccar5.glaciers'],
    #
    "pb_1f": ['ar5AIS.ipccar5.icesheets_AIS', 'larmip.larmip.AIS', 'GrIS1f.FittedISMIP.GrIS_GIS','ar5glaciers.ipccar5.glaciers'],
	"pb_2f": ['ar5AIS.ipccar5.icesheets_AIS','larmip.larmip.AIS','deconto21.deconto21.AIS_AIS','bamber19.bamber19.icesheets_AIS',
			  'GrIS1f.FittedISMIP.GrIS_GIS','bamber19.bamber19.icesheets_GIS','emuglaciers.emulandice.glaciers', 'ar5glaciers.ipccar5.glaciers']
}
# ^^^
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Generic Function block
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
def create_directory(dir_name: str) -> str:
    full_path = os.path.join(os.getcwd(), dir_name)
    shutil.rmtree(full_path, ignore_errors=True)
    os.makedirs(full_path)
    return full_path
#
def delete_files_with_pattern(folder, pattern, exclusion_pattern):
    folder_path = Path(folder)
    for file_path in folder_path.glob(pattern):
        if not file_path.match(exclusion_pattern):
            file_path.unlink()
#
def copy_filename_with_pattern(source_dir,destination_dir,pattern):
    source_file_pattern = os.path.join(source_dir, pattern)
    matching_files = glob.glob(source_file_pattern)
    for file_path in matching_files:
        shutil.copy2(file_path, destination_dir)   
#
def copy_all_files_from(srcDIR,dstnDIR):
    file_list = os.listdir(srcDIR)
    for file_name in file_list:
        source_file = os.path.join(srcDIR, file_name)
        destination_file = os.path.join(dstnDIR, file_name)
        shutil.copy2(source_file, destination_file)
# 
def find_filename_with_pattern(path, search_term):
    path_obj = Path(path)
    matching_files = list(path_obj.glob(f"*{search_term}*"))
    if len(matching_files) > 1:
        raise ValueError("There are multiple files with the same keyword.")
    elif not matching_files:
        return None
    else:
        return matching_files[0].name
# ^^^

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Convert Samples to Quantiles.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Samples_to_Quantiles(in_file, out_file):
	#
	valid_variables = {
		"sea_level_change": {"units": "mm", "scale_factor": 1.0},
		"sea_level_change_rate": {"units": "mm per year", "scale_factor": 0.1}
	}
	quantiles = np.unique(np.append(np.round(np.linspace(0,1,101), 3), (0.001, 0.005, 0.01, 0.05, 0.167, 0.5, 0.833, 0.95, 0.99, 0.995, 0.999)))
	#
	missing_value = np.iinfo(np.int16).min
	#
	# Open the file
	with xr.open_dataset(in_file) as dataset:
		variable_name = next((var for var in valid_variables if var in dataset.variables), None)
		if not variable_name:
			print(f"No valid variable name exists in {in_file}")  # Changed input_file to in_file
			return 1
		#
		variable_data = {
			"units": valid_variables[variable_name]["units"],
			"missing_value": missing_value
		}
		scale_factor = valid_variables[variable_name]["scale_factor"]
		#
		lats = dataset['lat']
		lons = dataset['lon']
		years = dataset['years']
		location_ids = dataset['locations'].values
		quantile_values = np.nanquantile(dataset[variable_name], quantiles, axis=0)
		#
		output_data = xr.Dataset({
			variable_name: (("quantiles", "years", "locations"), quantile_values, variable_data),
			"lat": (("locations"), lats.data), "lon": (("locations"), lons.data)
		}, coords={"years": years.data, "locations": location_ids.data, "quantiles": quantiles}, attrs=dataset.attrs)
		#
		encoding = {
			variable_name: {
				"scale_factor": scale_factor,
				"dtype": "i2",
				"zlib": True,
				"complevel": 4,
				"_FillValue": missing_value
			}
		}
		output_data.to_netcdf(out_file, encoding=encoding)
	#
	return None
#^^^


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Generate P-box files.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def load_infiles(infiles, years):
    valid_varnames = ["sea_level_change", "sea_level_change_rate"]
    valid_varunits = {"sea_level_change": "mm", "sea_level_change_rate": "mm per year"}
    valid_varscale = {"sea_level_change": 1.0, "sea_level_change_rate": 0.1}
	#
    localsl_q = []
    ds = xr.open_dataset(infiles[0], engine='netcdf4')
	#
    varname = next((v for v in valid_varnames if v in ds.variables), None)
    if not varname:
        raise ValueError(f"No valid variable name exists in {infiles[0]}")
	#
    varunit = valid_varunits[varname]
    varscale = valid_varscale[varname]
    ids = ds['locations'].values
    lats = ds['lat'].values
    lons = ds['lon'].values
    qvar = np.round(ds['quantiles'].values, 3)
	#
    for infile in infiles:
        with xr.open_dataset(infile, engine='netcdf4') as ds:
            localsl_q.append(ds[varname].sel(years=years).values)
	#
    return np.array(localsl_q), varname, varunit, varscale, ids, lats, lons, qvar
#
# ...............................................................................................
def generate_pbox(infiles, outfile, pyear_start, pyear_end, pyear_step):
    years = np.arange(pyear_start, pyear_end+1, pyear_step)
    component_data, varname, varunit, varscale, ids, lats, lons, qvar = load_infiles(infiles, years)
	#
    median_idx = np.flatnonzero(qvar == 0.5)
    above_idx = np.arange(median_idx + 1, len(qvar))
    below_idx = np.arange(median_idx)
	#
    pbox = np.full(component_data.shape[1:], np.nan)
    pbox[median_idx,:,:] = np.mean(component_data[:,median_idx,:,:], axis=0)
    pbox[below_idx,:,:] = np.amin(component_data[:,below_idx,:,:], axis=0)
    pbox[above_idx,:,:] = np.amax(component_data[:,above_idx,:,:], axis=0)
	#
    dataset = xr.Dataset({
        varname: (["quantiles", "years", "locations"], pbox, {
            "units": varunit,
            "missing_value": np.iinfo(np.int16).min,
            "scale_factor": varscale
        }),
        "lat": (["locations"], lats),
        "lon": (["locations"], lons),
        "locations": (["locations"], ids),
        "years": (["years"], years),
        "quantiles": (["quantiles"], qvar)
    }, attrs={
        "description": "Pbox generated from a FACTS sea-level change projection workflow",
        "history": f"Created {time.ctime(time.time())}",
        "source": f"Generated with files: {', '.join([os.path.basename(x) for x in infiles])}"
    })

    dataset.to_netcdf(outfile)
#^^^

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Generate Confidence level files.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def get_scenarios(directory: str) -> tuple:
	#
    scenario_types = list(PB_file_patterns.keys())
    scenarios = {stype: [x for x in os.listdir(os.path.join(directory, stype)) if not x.startswith('.')] for stype in scenario_types}
	#
    med_scenarios = sorted(list(set(scenarios["pb_1e"]) & set(scenarios["pb_1f"])))
    low_scenarios = sorted(list(set(scenarios["pb_2e"]) & set(scenarios["pb_2f"])))
	#
    return med_scenarios, low_scenarios

# ............................................................
def make_confidence_file(infile_e=None, infile_f=None, f_years=np.arange(2020, 2101, 10), outfile=None, is_rates=False):
    if infile_f is None and infile_e is None:
        return 1
    varname = "sea_level_change_rate" if is_rates else "sea_level_change"
    varscale = 0.1 if is_rates else 1.0
	#
    with xr.open_dataset(infile_f, engine='netcdf4') as nc_f:
        nc_out = nc_f.sel(years=f_years)
	#
    source_files = [infile_f]
    if infile_e is not None:
        with xr.open_dataset(infile_e, engine='netcdf4') as nc_e:
            nc_out = nc_e.combine_first(nc_f.sel(years=f_years))
        source_files.append(infile_e)
	#
    nc_missing_value = np.iinfo(np.int16).min
    nc_attrs = {
        "description": "Combined confidence output file for AR6 sea-level change projections",
        "history": f"Created {time.ctime(time.time())}",
        "source": f"Files Combined: {','.join(source_files)}"
    }
	#
    nc_out.attrs = nc_attrs
    nc_out.to_netcdf(outfile, encoding={
        varname: {
            "scale_factor": varscale,
            "dtype": "i2",
            "zlib": True,
            "complevel": 4,
            "_FillValue": nc_missing_value
        }
    })
# ............................................................
def generate_confidence_files(pboxdir: str, outdir: str):
    #
    is_rates = "rates" in pboxdir
    med_scenarios, low_scenarios = get_scenarios(pboxdir)
	#
    confidence_map = {"medium_confidence": med_scenarios,
                      "low_confidence": low_scenarios}
    sl_files = ["glaciers", "landwaterstorage", "sterodynamics", "AIS", "GIS", "total", "verticallandmotion"]
	#
    for conf_level, scenarios in confidence_map.items():
        for scenario in scenarios:
            scenario_dir = os.path.join(pboxdir, "pb_1f" if "medium" in conf_level else "pb_2f", scenario)
            #       
            files = {sl: f'{scenario_dir}/{find_filename_with_pattern(scenario_dir, sl)}' for sl in sl_files}
            #
            for key, file_path in files.items():
                if file_path is None or 'None' in file_path:
                    continue  
                #
                outpath = Path(outdir, conf_level, scenario); outpath.mkdir(parents=True, exist_ok=True)
                filename = f"{key}_{scenario}_{conf_level}_{'rates' if is_rates else 'values'}.nc"
                outfile = outpath / filename
                #
                make_confidence_file(infile_f=file_path, f_years=np.arange(2020, 2101, 10), outfile=str(outfile), is_rates=is_rates)

# ^^^