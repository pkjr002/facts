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


import tempfile



def FindFiles(indir):

	pattern = os.path.join(indir, "**", "*.nc")
	files = glob.iglob(pattern, recursive=True)

	return(files)

'''
def ConvertSamplesToDist(infile, outfile, chunksize):

	# Quantiles to extract from the data
	q = np.unique(np.append(np.linspace(0,1,101), (0.001, 0.005, 0.01, 0.05, 0.167, 0.5, 0.833, 0.95, 0.99, 0.995, 0.999)))

	# Open the file
	with xr.open_dataset(infile, chunks={'locations':chunksize}) as nc:

		# Calculate the quantiles
		try:
			x = np.nanquantile(nc['sea_level_change'], q, axis=0)
		except:
			x = np.nanquantile(nc['sea_level_change_rate'], q, axis=0)
		x_lats = nc['lat']
		x_lons = nc['lon']
		x_years = nc['years']
		x_ids = nc['locations'].values

		# Transpose to fit the old format's shape [quantiles, sites, years]
		x = np.transpose(x, (0,2,1))

		# Extract the attributes for this data
		x_attrs = nc.attrs

	# Missing value
	nc_missing_value = np.iinfo(np.int16).min

	# Create the output array
	x_out = xr.Dataset({"localSL_quantiles": (("quantiles", "id", "years"), x, {"units":"mm", "missing_value":nc_missing_value}),
							"lat": (("id"), x_lats),
							"lon": (("id"), x_lons)},
		coords={"years": x_years, "id": x_ids, "quantiles": q}, attrs=x_attrs)

	x_out.to_netcdf(outfile, encoding={"localSL_quantiles": {"dtype": "i2", "zlib": True, "complevel":4, "_FillValue": nc_missing_value}})

	return(None)
'''



# Development - Do not match the old version of FACTS output
def ConvertSamplesToDist_dev(infile, outfile, chunksize):

	# Initialize valid variable names and units
	x_varname = None
	x_units = None
	x_scalefactor = None
	valid_varnames = ["sea_level_change", "sea_level_change_rate"]
	valid_varunits = {"sea_level_change": "mm", "sea_level_change_rate": "mm per year"}
	valid_scalefactors = {"sea_level_change": 1.0, "sea_level_change_rate": 0.1}

	# Quantiles to extract from the data
	q = np.unique(np.append(np.round(np.linspace(0,1,101), 3), (0.001, 0.005, 0.01, 0.05, 0.167, 0.5, 0.833, 0.95, 0.99, 0.995, 0.999)))

	# Open the file
	with xr.open_dataset(infile, chunks={'locations':chunksize}) as nc:

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
		x = np.nanquantile(nc[x_varname], q, axis=0)
		x_lats = nc['lat']
		x_lons = nc['lon']
		x_years = nc['years']
		x_ids = nc['locations'].values

		# Extract the attributes for this data
		x_attrs = nc.attrs

	# Missing value
	nc_missing_value = np.iinfo(np.int16).min

	# Create the output array
	x_out = xr.Dataset({x_varname: (("quantiles", "years", "locations"), x, {"units":x_units, "missing_value":nc_missing_value}),
							"lat": (("locations"), x_lats),
							"lon": (("locations"), x_lons)},
		coords={"years": x_years, "locations": x_ids, "quantiles": q}, attrs=x_attrs)

	x_out.to_netcdf(outfile, encoding={x_varname: {"scale_factor":x_scalefactor, "dtype": "i2", "zlib": True, "complevel":4, "_FillValue": nc_missing_value}})

	return(None)







# For merging multiple data sets with non-monotonically increasing/decreasing values along concat dimension
def open_mfdataset_merge_only(paths, **kwargs):
	paths = sorted(glob.iglob(paths))
	#if len(paths) == 1:
		#path = paths[0]
		#return(xr.open_dataset(path, **kwargs))
	#return xr.merge([xr.open_dataset(path, **kwargs) for path in paths])
	return xr.concat([xr.open_dataset(path, **kwargs) for path in paths], dim="locations")


# Use new format of files.  Manually chunk, generate temp intermediate files, then put everything together.
def ConvertSamplesToDist_dev_dev(infile, outfile, tempdir, chunksize):

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
		for i in np.arange(0,nsites,chunksize):

			# This chunk's temporary file name
			temp_filename = os.path.join(tempdir, "temp_dist_file_{0:05d}.nc".format(int(i/chunksize)))

			# If this temporary file exists, move on to the next chunk
			#if os.path.isfile(temp_filename):
			#	print("{} found, skipping to next chunk".format(temp_filename))
			#	continue

			# Get the sample indices
			top_idx = np.amin([i+chunksize, nsites])
			site_idx = np.arange(i,top_idx)

			# Extract the data and make the quantile
			x = np.nanquantile(nc[x_varname].isel({"locations":site_idx}), q, axis=0)

			# Create the output array
			x_tempout = xr.Dataset({x_varname: (("quantiles", "years", "locations"), x.data, {"units":x_units, "missing_value":nc_missing_value}),
									"lat": (("locations"), x_lats[site_idx].data),
									"lon": (("locations"), x_lons[site_idx].data)},
				coords={"years": x_years.data, "locations": x_ids[site_idx].data, "quantiles": q.data}, attrs=x_attrs)

			# Write the temporary output file
			x_tempout.to_netcdf(temp_filename, encoding={x_varname: {"scale_factor":x_scalefactor, "dtype": "i2", "zlib": True, "complevel":4, "_FillValue": nc_missing_value}})


	# Work around for instances where only one temporary file is generated
	#n_tempfiles = len(fnmatch.filter(os.listdir(tempdir), "temp_dist_file_*.nc"))
	#if n_tempfiles == 1:
	#	shutil.copy2(os.path.join(tempdir, "temp_dist_file_00000.nc"), outfile)
	#	os.remove(os.path.join(tempdir, "temp_dist_file_00000.nc"))
	#	return(None)

	# Open the temporary data sets
	#combined = xr.open_mfdataset(os.path.join(tempdir, "temp_dist_file_*.nc"), concat_dim="locations", chunks={"locations":chunksize})
	globstring = os.path.join(tempdir, "temp_dist_file_*.nc")
	combined = open_mfdataset_merge_only(globstring, chunks={"locations":chunksize})
	combined.attrs = x_attrs

	# Write the combined data out to the final netcdf file
	combined.to_netcdf(outfile, encoding={x_varname: {"scale_factor":x_scalefactor, "dtype": "i2", "zlib": True, "complevel":4, "_FillValue": nc_missing_value}})

	# Remove the temporary files
	for i in np.arange(0,nsites,chunksize):
		os.remove(os.path.join(tempdir, "temp_dist_file_{0:05d}.nc".format(int(i/chunksize))))


	return(None)





if __name__ == "__main__":

	# Initialize the argument parser
	parser = argparse.ArgumentParser(description="Convert FACTS full-sample output into distribution quantiles")

	# Add arguments for the resource and experiment configuration files
	parser.add_argument('--indir', help="Input directory with projection files", required=True)
	parser.add_argument('--outdir', help="Output directory for distribution files", required=True)
	parser.add_argument('--tempdir', help="Directory to hold temporary files", default=None)
	parser.add_argument('--chunksize', help="Number of locations to process per chunk [default=50]", type=int, default=50)


	# Parse the arguments
	args = parser.parse_args()
	indir = args.indir
	outdir = args.outdir
	tempdir = args.tempdir
	chunksize = args.chunksize

	# Find files to convert
	files = FindFiles(indir)

	i = 0

	# Loop through the files
	for this_file in files:

		print(this_file)

		# Extract the parent directories of this file
		inpath = str(Path(this_file).parent)

		# Generate in the outdir the same directory structure as indir
		outpath = Path(re.sub(indir,outdir,inpath))
		Path.mkdir(outpath, parents=True, exist_ok=True)

		# Make the output file name for this_file
		outfile = re.sub(indir,outdir,this_file)

		# Make a temporary directory for this file
		with tempfile.TemporaryDirectory() as file_tempdir:

			# Skip this file if the output file already exists
			if os.path.isfile(outfile):
				print("{} already exists. Moving on to next file.".format(outfile))
				continue

			# Convert the file
			ConvertSamplesToDist_dev_dev(this_file, outfile, file_tempdir, chunksize)


	# Done
	sys.exit()
