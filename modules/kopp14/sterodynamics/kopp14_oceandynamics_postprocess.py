import numpy as np
import pickle
import os
import sys
import time
import argparse
import re
from scipy.stats import norm
from scipy.stats import t
from netCDF4 import Dataset
import xarray as xr

''' kopp14_postprocess_oceandynamics.py

This runs the post-processing stage for the ocean dynamics component of the Kopp14
workflow. Projections generated from this stage are site-specific and include both the
ocean dynamics and thermal expansion contributions.

Parameters:
nsamps = Number of samples to draw
rng_seed = Seed value for the random number generator
pipeline_id = Unique identifier for the pipeline running this code

Note that the value of 'nsamps' and 'rng_seed' are shared between the projection stage
and the post-processing stage when run within FACTS.

'''

def kopp14_postprocess_oceandynamics(nsamps, rng_seed, pipeline_id):

	# Read in the configuration -----------------------------------
	infile = "{}_config.pkl".format(pipeline_id)
	try:
		f = open(infile, 'rb')
	except:
		print("Cannot open infile\n")
		raise

	# Extract the data from the file
	my_data = pickle.load(f)

	# Extract the relevant data
	targyears = my_data['targyears']
	rcp_scenario = my_data['rcp_scenario']
	GCMprobscale = my_data['GCMprobscale']
	maxDOF = my_data['maxDOF']
	mergeZOSZOSTOGA = my_data["mergeZOSZOSTOGA"]
	no_correlation = my_data["no_correlation"]

	# Read in the ZOS data file ------------------------------------
	infile = "{}_ZOS.pkl".format(pipeline_id)
	try:
		f = open(infile, 'rb')
	except:
		print("Cannot open infile\n")
		raise

	# Extract the data from the file
	my_data = pickle.load(f)

	focus_site_ids = my_data['focus_site_ids']
	focus_site_lats = my_data['focus_site_lats']
	focus_site_lons = my_data['focus_site_lons']
	modellist = my_data['comb_modellist']

	# Read in the TE fit data file --------------------------------
	infile = "{}_thermalexp_fit.pkl".format(pipeline_id)
	try:
		f = open(infile, 'rb')
	except:
		print("Cannot open infile\n")
		raise

	# Extract the data from the file
	my_data = pickle.load(f)

	ThermExpMean = my_data['ThermExpMean']
	ThermExpStd = my_data['ThermExpStd']
	ThermExpYears = my_data['ThermExpYears']

	# Read in the OD fit data file --------------------------------
	infile = "{}_oceandynamics_fit.pkl".format(pipeline_id)
	try:
		f = open(infile, 'rb')
	except:
		print("Cannot open infile\n")
		raise

	# Extract the data from the file
	my_data = pickle.load(f)

	OceanDynYears = my_data['OceanDynYears']
	OceanDynMean = my_data['OceanDynMean']
	OceanDynStd = my_data['OceanDynStd']
	OceanDynN = my_data['OceanDynN']
	OceanDynTECorr = my_data['OceanDynTECorr']
	OceanDynDOF = my_data['OceanDynDOF']

	# Read in the TE projections data file ------------------------
	infile = "{}_projections.pkl".format(pipeline_id)
	try:
		f = open(infile, 'rb')
	except:
		print("Cannot open infile\n")
		raise

	# Extract the data from the file
	my_data = pickle.load(f)

	tesamps = my_data['thermsamps']

	# Evenly sample an inverse normal distribution
	x = np.linspace(0,1,nsamps+2)[1:(nsamps+1)]
	norm_inv = norm.ppf(x)

	# Initialize variable to hold the samples
	local_sl = np.empty((nsamps, len(targyears), len(focus_site_ids)))

	# Determine the thermal expansion scale coefficient
	ThermExpScale = norm.ppf(0.95)/norm.ppf(GCMprobscale)

	# Loop over the sites
	nsites = len(focus_site_ids)
	for i in np.arange(0,nsites):

		# This site index
		this_site_ind = i

		# Reset the RNG seed
		rng = np.random.default_rng(rng_seed)

		# Create a permutation of the inverse normal distribution
		norm_inv_perm = rng.permutation(norm_inv)

		# Loop over the target years
		ntimes = len(targyears)
		for j in np.arange(0,ntimes):

			# This target year
			targyear = targyears[j]

			# Find which index targyear is located in the OD and TE years
			od_year_ind = np.argmin(np.abs(OceanDynYears - targyear))
			te_year_ind = np.argmin(np.abs(ThermExpYears - targyear))

			# Calculate the conditional mean and sd
			if no_correlation:
				condmean = OceanDynMean[od_year_ind,this_site_ind]
				condstd = (ThermExpScale * OceanDynStd[od_year_ind,this_site_ind])
			else:
				condmean = OceanDynMean[od_year_ind,this_site_ind] +  OceanDynStd[od_year_ind,this_site_ind] * OceanDynTECorr[od_year_ind,this_site_ind] * (tesamps[:,j]-ThermExpMean[te_year_ind])/ThermExpStd[te_year_ind]
				condstd = (ThermExpScale * OceanDynStd[od_year_ind,this_site_ind]) * np.sqrt(1 - OceanDynTECorr[od_year_ind,this_site_ind]**2)

			# Make the projection
			local_sl[:,j,i] = t.ppf(norm.cdf(norm_inv_perm),OceanDynDOF[od_year_ind,this_site_ind]) * condstd + condmean
			
			if not (mergeZOSZOSTOGA):
				local_sl[:,j,i] += tesamps[:,j]


	# Write the localized projections to a netcdf file
	dset = Dataset(os.path.join(os.path.dirname(__file__), "{}_localsl.nc".format(pipeline_id)), "w", format="NETCDF4")
	
	# Define Dimensions
	year_dim = dset.createDimension("years", ntimes)
	samp_dim = dset.createDimension("samples", nsamps)
	loc_dim  = dset.createDimension("locations", nsites)

	# Populate dimension variables
	year_var = dset.createVariable("years", "i4", ("years",))
	samp_var = dset.createVariable("samples", "i8", ("samples",))
	loc_var  = dset.createVariable("locations", "i8", ("locations",))
	lat_var  = dset.createVariable("lat", "f4", ("locations",))
	lon_var  = dset.createVariable("lon", "f4", ("locations",))

	# Create a data variable
	samps = dset.createVariable("sea_level_change", "f4", ("samples", "years", "locations"), zlib=True, least_significant_digit=2)

	# Assign attributes
	dset.description = "Local SLR contributions from ocean dynamics according to K14 workflow"
	dset.history = "Created " + time.ctime(time.time())
	dset.source = "FACTS: {0} - {1}. ".format(pipeline_id, rcp_scenario)
	lat_var.units = "Degrees North"
	lon_var.units = "Degrees East"
	samps.units = "mm"

	# Put the data into the netcdf variables
	lat_var[:]  = focus_site_lats
	lon_var[:]  = focus_site_lons
	loc_var[:]  = focus_site_ids
	year_var[:]  = targyears
	samp_var[:]  = np.arange(nsamps)
	samps[:,:,:] = local_sl

	# Close the netcdf
	dset.close()

	# Produce the intermediate data output netCDF file =========================

	# Produce the included model string
	model_string_pieces = ["{0}-{1}".format(modellist[x], rcp_scenario) for x in np.arange(len(modellist))]
	model_string = "Models and scenarios included: " + ", ".join(model_string_pieces)

	# Write the localized projections to a netcdf file
	rootgrp = Dataset(os.path.join(os.path.dirname(__file__), "{}_OceanDyn.nc".format(pipeline_id)), "w", format="NETCDF4")
	
	# Define Dimensions
	site_dim = rootgrp.createDimension("nsites", nsites)
	odyear_dim = rootgrp.createDimension("OceanDynYears", len(OceanDynYears))
	
	# Populate dimension variables
	lat_var = rootgrp.createVariable("lat", "f4", ("nsites",))
	lon_var = rootgrp.createVariable("lon", "f4", ("nsites",))
	id_var = rootgrp.createVariable("id", "i4", ("nsites",))
	odyear_var = rootgrp.createVariable("OceanDynYears", "i4", ("OceanDynYears",))
	
	# Create a data variable
	oceandynmean = rootgrp.createVariable("OceanDynMean", "f4", ("nsites", "OceanDynYears"))
	oceandynstd = rootgrp.createVariable("OceanDynStd", "f4", ("nsites", "OceanDynYears"))
	oceandynn = rootgrp.createVariable("OceanDynN", "i4", ("nsites", "OceanDynYears"))
	oceandyntecorr = rootgrp.createVariable("OceanDynTECorr", "f4", ("nsites", "OceanDynYears"))
	oceandyndof = rootgrp.createVariable("OceanDynDOF", "i4", ("nsites", "OceanDynYears"))
	
	# Assign attributes
	rootgrp.description = "Ocean Dynamics intermediate data for the K14 workflow"
	rootgrp.history = "Created " + time.ctime(time.time())
	rootgrp.source = "FACTS: {0} - {1}. ".format(pipeline_id, rcp_scenario) + model_string
	lat_var.units = "Degrees North"
	lon_var.units = "Degrees East"
	
	# Put the data into the netcdf variables
	lat_var[:] = focus_site_lats
	lon_var[:] = focus_site_lons
	id_var[:] = focus_site_ids
	odyear_var[:] = OceanDynYears
	oceandynmean[:,:] = OceanDynMean.T
	oceandynstd[:,:] = OceanDynStd.T
	oceandynn[:,:] = OceanDynN.T
	oceandyntecorr[:,:] = OceanDynTECorr.T
	oceandyndof[:,:] = OceanDynDOF.T
	
	# Close the netcdf
	rootgrp.close()

	# Produce the intermediate zos/zostoga output netCDF file =========================

	# Read interim ZOS and ZOSTOGA data file ------------------------------------
	infile = "{}_zos_fit_combined.pkl".format(pipeline_id)
	try:
		f = open(infile, 'rb')
	except:
		print("Cannot open infile\n")
		raise
	# Extract the data from the file
	interim_data = pickle.load(f)
	f.close()
	# ------------------------------------

	# Save to xarray dataset
	ds_out = xr.Dataset({'zos': (['year', 'model', 'location'], interim_data['sZOS']),
			     'zostoga': (['year', 'model'], interim_data['sZOSTOGA'])},
			    coords={'year': interim_data['datayears'],
				    'model': interim_data['comb_modellist'],
				    'location': interim_data['focus_site_ids'],
				    'latitude': (('location'), interim_data['focus_site_lats']),
				    'longitude': (('location'), interim_data['focus_site_lons'])})
	
	# Assign attributes
	ds_out.attrs['description'] = "ZOS and ZOSTOGA intermediate dataset"
	ds_out.attrs['history'] = "Created " + time.ctime(time.time())
	ds_out.attrs['source'] = "FACTS: {0} - {1}. ".format(pipeline_id, rcp_scenario)
	ds_out['latitude'].attrs['units'] = "Degrees North"
	ds_out['longitude'].attrs['units'] = "Degrees East"
	
	# Write the dataset to a netCDF file
	ds_out.to_netcdf("{}_CMIP.nc".format(pipeline_id), format="NETCDF4")
	
	return(None)


if __name__ == '__main__':

	# Initialize the command-line argument parser
	parser = argparse.ArgumentParser(description="Run the post-processing stage for the Kopp14 ocean dynamics workflow",\
	epilog="Note: This is meant to be run as part of the Kopp14 module within the Framework for the Assessment of Changes To Sea-level (FACTS)")

	# Define the command line arguments to be expected
	parser.add_argument('--nsamps', help="Number of samples to generate", default=20000, type=int)
	parser.add_argument('--seed', help="Seed value for random number generator", default=1234, type=int)
	parser.add_argument('--pipeline_id', help="Unique identifier for this instance of the module")

	# Parse the arguments
	args = parser.parse_args()

	# Run the postprocessing stage
	kopp14_postprocess_oceandynamics(args.nsamps, args.seed, args.pipeline_id)

	# Done
	exit()
