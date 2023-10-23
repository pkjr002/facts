import os
import re
import sys
import time
import glob
import shutil
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
# current_directory = os.getcwd()
PD = os.getcwd()
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Function block
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def delete_files_with_pattern(folder, pattern, exclusion_pattern):
    folder_path = Path(folder)
    for file_path in folder_path.glob(pattern):
        if not file_path.match(exclusion_pattern):
            file_path.unlink()
            # print(f"Deleted file: {file_path}")
# ^^^

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Copy all files to the FOLDER that match a pattern
def copy_filename_with_pattern(source_dir,destination_dir,pattern):
    source_file_pattern = os.path.join(source_dir, pattern)
    matching_files = glob.glob(source_file_pattern)
    for file_path in matching_files:
        shutil.copy2(file_path, destination_dir)
# ^^^   
  

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Copy all files in a folder to another folder
def copy_all_files_from(srcDIR,dstnDIR):
    file_list = os.listdir(srcDIR)
    for file_name in file_list:
        source_file = os.path.join(srcDIR, file_name)
        destination_file = os.path.join(dstnDIR, file_name)
        shutil.copy2(source_file, destination_file)
# ^^^   
 
 
 
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def find_filename_with_pattern(path, search_term):
    path_obj = Path(path)
    matching_files = list(path_obj.glob(f"*{search_term}*"))
    if len(matching_files) > 1:
        raise ValueError("There are multiple files with the same keyword.")
    if not matching_files:
        raise ValueError("No file found with the given keyword.")
    return matching_files[0].name
# ^^^


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def fileNAME(patH,name,name1):
    folder_path = patH
    search_term = name   # replace with the word you want to search for
    search_term1 = name1
    file_pattern = f"{folder_path}/*{search_term}*{search_term1}*"  # create a pattern to match files containing the search term
    matching_files = glob.glob(file_pattern)
    if len(matching_files)>1: 
        raise ValueError("There are 2 files with same keyword")
    fnme = os.path.basename(matching_files[0])
    return fnme
# ^^^
  


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Find a file with these 3 keywds (will fail when you have AIS as different cmponents.)
def fileNAME1(folder_path,search_term, search_term1, search_term2):
    matching_files = []
    for file_path in glob.glob(f"{folder_path}/*"):
        file_name = os.path.basename(file_path)
        if all(term in file_name for term in [search_term, search_term1, search_term2]):
            matching_files.append(file_path)
    if len(matching_files) > 1:
        raise ValueError("There are 2 or more files with the same keyword")
    fnme = os.path.basename(matching_files[0])
    return fnme
# ^^^


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Samples_to_Quantiles(in_file, out_file):
	#
	valid_variables = {
		"sea_level_change": {"units": "mm", "scale_factor": 1.0},
		"sea_level_change_rate": {"units": "mm per year", "scale_factor": 0.1}
	}
	#
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
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def load_infiles(infiles, years):
    valid_varnames = ["sea_level_change", "sea_level_change_rate"]
    valid_varunits = {"sea_level_change": "mm", "sea_level_change_rate": "mm per year"}
    valid_varscale = {"sea_level_change": 1.0, "sea_level_change_rate": 0.1}
	#
    localsl_q = []
    ds = xr.open_dataset(infiles[0])
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
        with xr.open_dataset(infile) as ds:
            localsl_q.append(ds[varname].sel(years=years).values)
	#
    return np.array(localsl_q), varname, varunit, varscale, ids, lats, lons, qvar
#
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

# Example usage:
# generate_pbox(list_of_input_files, "output_file.nc")


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CL fun
def GetScenarios(dir):

	# Get the scenario names from the available scenario directories
	# Ignore any hidden directories (i.e. .DS_Store)
	pb1e_scenarios = [x for x in os.listdir(os.path.join(dir, "pb_1e")) if not re.search(r"^\.", x)]
	pb1f_scenarios = [x for x in os.listdir(os.path.join(dir, "pb_1f")) if not re.search(r"^\.", x)]
	pb2e_scenarios = [x for x in os.listdir(os.path.join(dir, "pb_2e")) if not re.search(r"^\.", x)]
	pb2f_scenarios = [x for x in os.listdir(os.path.join(dir, "pb_2f")) if not re.search(r"^\.", x)]

	# Find the overlapping scenarios
	med_scenarios = list(set(pb1e_scenarios) & set(pb1f_scenarios))
	low_scenarios = list(set(pb2e_scenarios) & set(pb2f_scenarios))

	# Return the overlapping scenarios
	med_scenarios.sort()
	low_scenarios.sort()
	return(med_scenarios, low_scenarios)



def GetFiles(dir):

	# There should be a file for glaciers, landwaterstorage, oceandynamics,
	# AIS, GIS, and total...and for regional projections verticallandmotion.
	file_keys = ["glaciers", "landwaterstorage", "sterodynamics", "AIS", "GIS", "total", "verticallandmotion"]

	# Initialize list of matched file keys
	match_files = {}

	# Loop over the keys and find the associated files
	for this_key in file_keys:

		# Locate this file in the directory
		pattern = "*{}*.nc".format(this_key)
		this_file = fnmatch.filter(os.listdir(dir), pattern)

		# There should be only one match
		if len(this_file) == 1:

			# Append the match
			match_files[this_key] = os.path.join(dir, this_file[0])

		elif len(this_file) > 1:
			raise Exception("More than one file matched in {} for key {}".format(dir, this_key))

		else:

			match_files[this_key] = None

	# Return the dictionary of files
	return(match_files)



# def MakeConfidenceFile(infile_e=None, infile_f=None, f_years=np.arange(2020,2301,10), outfile=None, is_rates=False):
#-pk def MakeConfidenceFile(infile_e=None, infile_f=None, f_years=np.arange(2020,2151,10), outfile=None, is_rates=False):
def MakeConfidenceFile(infile_e=None, infile_f=None, f_years=np.arange(2020,2101,10), outfile=None, is_rates=False):

	# If both infile_e and infile_f are None, then there's no data for this component key.
	# Return and let the code move onto the next component key
	if infile_f is None and infile_e is None:
		return(1)

	# Variable names and attributes
	if is_rates:
		varname = "sea_level_change_rate"
		varscale = 0.1
	else:
		varname = "sea_level_change"
		varscale = 1.0

	# Open and subset the f file
	with xr.open_dataset(infile_f) as nc_f:
	#with xr.open_dataset(infile_f) as nc_f:
		nc_out = nc_f.sel(years=f_years)

	# Add the f file to the source list
	source_files = [infile_f]

	# If there's an e file, overlap it with the f file
	if infile_e is not None:
		with xr.open_dataset(infile_e) as nc_e:
		#with xr.open_dataset(infile_e) as nc_e:
			nc_out = nc_e.combine_first(nc_f.sel(years=f_years))

		# Append the e file to the source file list
		source_files.append(infile_e)

	# Define the missing value for the netCDF files
	nc_missing_value = np.iinfo(np.int16).min

	# Attributes for the output file
	nc_attrs = {"description": "Combined confidence output file for AR6 sea-level change projections",
			"history": "Created " + time.ctime(time.time()),
			"source": "Files Combined: {}".format(",".join(source_files))}

	# Put the attributes onto the output file
	nc_out.attrs = nc_attrs

	# Write the output file
	nc_out.to_netcdf(outfile, encoding={varname: {"scale_factor": varscale, "dtype": "i2", "zlib": True, "complevel":4, "_FillValue": nc_missing_value}})

	# Done
	return(None)

# ==> main loop
def GenerateConfidenceFiles(pboxdir, outdir):

	# Are we working with values or rates?
	is_rates = True if re.search(r"rates", pboxdir) is not None else False

	# Get the overlapping scenarios for each confidence level
	med_scenarios, low_scenarios = GetScenarios(pboxdir)

	# If these are rate pboxes...
	if is_rates:

		# Medium-confidence: pb_1f through 2150
		# Low-confidence: pb_2f through 2300

		# Loop over the medium scenarios
		for this_scenario in med_scenarios:

			# Get the list of files for this scenario
			pb1f_infiles = GetFiles(os.path.join(pboxdir, "pb_1f", this_scenario))

			# Loop over the available components
			for this_key in pb1f_infiles.keys():

				# Define the output file name
				outpath = Path(os.path.join(outdir, "medium_confidence", this_scenario))
				Path.mkdir(outpath, parents=True, exist_ok=True)
				outfile = os.path.join(outpath, "{}_{}_medium_confidence_rates.nc".format(this_key, this_scenario))

				# Make the output file
				#-pk  MakeConfidenceFile(infile_f=pb1f_infiles[this_key], f_years=np.arange(2020,2151,10), outfile=outfile, is_rates=is_rates)
				MakeConfidenceFile(infile_f=pb1f_infiles[this_key], f_years=np.arange(2020,2101,10), outfile=outfile, is_rates=is_rates)

		# Loop over the low scenarios
		for this_scenario in low_scenarios:

			# Get the list of files for this scenario
			pb1f_infiles = GetFiles(os.path.join(pboxdir, "pb_2f", this_scenario))

			# Loop over the available components
			for this_key in pb1f_infiles.keys():

				# Define the output file name
				outpath = Path(os.path.join(outdir, "low_confidence", this_scenario))
				Path.mkdir(outpath, parents=True, exist_ok=True)
				outfile = os.path.join(outpath, "{}_{}_low_confidence_rates.nc".format(this_key, this_scenario))

				# Make the output file
				# MakeConfidenceFile(infile_f=pb1f_infiles[this_key], f_years=np.arange(2020,2301,10), outfile=outfile, is_rates=is_rates, chunksize=chunksize)
				#-pk MakeConfidenceFile(infile_f=pb1f_infiles[this_key], f_years=np.arange(2020,2151,10), outfile=outfile, is_rates=is_rates)
				MakeConfidenceFile(infile_f=pb1f_infiles[this_key], f_years=np.arange(2020,2101,10), outfile=outfile, is_rates=is_rates)

	# These are value files...
	else:

		# Medium-confidence: pb_1e through 2100, pb_1f through 2150
		# Low-confidence: pb_2e through 2100, pb_2f through 2300

		# Loop over the medium scenarios
		for this_scenario in med_scenarios:

			# Get the list of files for this scenario
			pb1e_infiles = GetFiles(os.path.join(pboxdir, "pb_1e", this_scenario))
			pb1f_infiles = GetFiles(os.path.join(pboxdir, "pb_1f", this_scenario))

			# Loop over the available components
			for this_key in pb1e_infiles.keys():

				# Define the output file name
				outpath = Path(os.path.join(outdir, "medium_confidence", this_scenario))
				Path.mkdir(outpath, parents=True, exist_ok=True)
				outfile = os.path.join(outpath, "{}_{}_medium_confidence_values.nc".format(this_key, this_scenario))

				# Make the output file
				#-pk MakeConfidenceFile(infile_e=pb1e_infiles[this_key], infile_f=pb1f_infiles[this_key], f_years=np.arange(2020,2151,10), outfile=outfile)
				MakeConfidenceFile(infile_e=pb1e_infiles[this_key], infile_f=pb1f_infiles[this_key], f_years=np.arange(2020,2101,10), outfile=outfile)

		# Loop over the low scenarios
		for this_scenario in low_scenarios:

			# Get the list of files for this scenario
			pb1e_infiles = GetFiles(os.path.join(pboxdir, "pb_2e", this_scenario))
			pb1f_infiles = GetFiles(os.path.join(pboxdir, "pb_2f", this_scenario))

			# Loop over the available components
			for this_key in pb1e_infiles.keys():

				# Define the output file name
				outpath = Path(os.path.join(outdir, "low_confidence", this_scenario))
				Path.mkdir(outpath, parents=True, exist_ok=True)
				outfile = os.path.join(outpath, "{}_{}_low_confidence_values.nc".format(this_key, this_scenario))

				# Make the output file
				# MakeConfidenceFile(infile_e=pb1e_infiles[this_key], infile_f=pb1f_infiles[this_key], f_years=np.arange(2020,2301,10), outfile=outfile, chunksize=chunksize)
				#-pk MakeConfidenceFile(infile_e=pb1e_infiles[this_key], infile_f=pb1f_infiles[this_key], f_years=np.arange(2020,2151,10), outfile=outfile)
				MakeConfidenceFile(infile_e=pb1e_infiles[this_key], infile_f=pb1f_infiles[this_key], f_years=np.arange(2020,2101,10), outfile=outfile)

	# Done
	return(None)
# ^^^

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