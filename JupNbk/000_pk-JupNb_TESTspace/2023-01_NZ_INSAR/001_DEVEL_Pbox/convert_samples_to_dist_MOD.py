import numpy as np
import os
import sys
import re
import argparse
import fnmatch
import glob
from netCDF4 import Dataset
from pathlib import Path
import xarray as xr
import dask.array as da
import time
import shutil



# Use new format of files.  Manually chunk, generate temp intermediate files, then put everything together.
def ConvertSamplesToDist_dev_dev(infile, outfile):

	# Initialize valid variable names and units
	x_varname = None
	x_units = None
	x_scalefactor = None
	valid_varnames = ["sea_level_change", "sea_level_change_rate"]
	valid_varunits = {"sea_level_change": "mm", "sea_level_change_rate": "mm per year"}
	valid_scalefactors = {"sea_level_change": 1.0, "sea_level_change_rate": 0.1}

	# Quantiles to extract from the data
	q = np.unique(np.append(np.round(np.linspace(0,1,101), 3), (0.001, 0.005, 0.01, 0.05, 0.167, 0.5, 0.833, 0.95, 0.99, 0.995, 0.999)))

	# Missing value
	nc_missing_value = np.iinfo(np.int16).min

	# Open the file
	with xr.open_dataset(infile) as nc:

		# Check which variable we're dealing with
		for this_varname in valid_varnames:
			if this_varname in nc.keys():
				x_varname = this_varname
				x_scalefactor = valid_scalefactors[this_varname]
				x_units = valid_varunits[this_varname]
				break
		if x_varname is None:
			print("No valid variable name exists in {}".format(infile))
			return(1)


		# If the variable exists, extract and convert

		x_lats = nc['lat']
		x_lons = nc['lon']
		x_years = nc['years']
		x_ids = nc['locations'].values

		# Extract the attributes for this data
		x_attrs = nc.attrs

		# Loop over the chunks of samples
		nsites = len(x_ids)
			
		site_idx = np.arange(nsites)
		# Extract the data and make the quantile
		x = np.nanquantile(nc[x_varname].isel({"locations":site_idx}), q, axis=0)

		# Create the output array
		x_tempout = xr.Dataset({x_varname: (("quantiles", "years", "locations"), x.data, {"units":x_units, "missing_value":nc_missing_value}),
								"lat": (("locations"), x_lats[site_idx].data),
								"lon": (("locations"), x_lons[site_idx].data)},
			coords={"years": x_years.data, "locations": x_ids[site_idx].data, "quantiles": q.data}, attrs=x_attrs)

		# Write theoutput file
		x_tempout.to_netcdf(outfile, encoding={x_varname: {"scale_factor":x_scalefactor, "dtype": "i2", "zlib": True, "complevel":4, "_FillValue": nc_missing_value}})
			

	return(None)