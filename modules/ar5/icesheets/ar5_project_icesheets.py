# GMSLR projection program used for IPCC WG1 AR5
# Translated from IDL to Python 2.7 by Jonathan Gregory 23.10.19

import os
import numpy as np
import pickle
import argparse
import time
import re
from netCDF4 import Dataset


class ProjectionError(Exception):
	pass
	

def project_greensmb(zt, fit_dict, nr):
	
	# Extract relevant parameters from the fit dictionary
	dtgreen = fit_dict['dtgreen']	 
	fnlogsd = fit_dict['fnlogsd']
	febound = fit_dict['febound']
	fgreendyn = fit_dict['fgreendyn']
	dgreen = fit_dict['dgreen']
	mSLEoGt = fit_dict['mSLEoGt']
	
	# random log-normal factor
	fn = np.exp(np.random.standard_normal(nr)*fnlogsd)
	
	# elevation feedback factor
	fe = np.random.sample(nr)*(febound[1]-febound[0])+febound[0]
	ff = fn*fe
	
	
	ztgreen = zt - dtgreen
	greensmbrate = fettweis(ztgreen, mSLEoGt)[:,np.newaxis,:] * ff[np.newaxis, :, np.newaxis]

	greensmb = np.cumsum(greensmbrate, axis=2)[:]
	greensmb += (1 - fgreendyn) * dgreen

	return greensmb.transpose((1,0,2))


def fettweis(ztgreen, mSLEoGt):
	# Greenland SMB in m yr-1 SLE from global mean temperature anomaly
	# using Eq 2 of Fettweis et al. (2013)
	return (71.5 * ztgreen + 20.4 * (ztgreen**2) + 2.8 * (ztgreen**3)) * mSLEoGt


def project_antsmb(zit, fit_dict, nr, nt, fraction=None):
	# Return projection of Antarctic SMB contribution as a cf.Field
	# zit -- cf.Field, ensemble of time-integral temperature anomaly timeseries
	# template -- cf.Field with the required shape of the output
	# fraction -- array-like, random numbers for the SMB-dynamic feedback
	
	# Extract relevant parameters from the fit dictionary
	pcoK = fit_dict['pcoK']
	KoKg = fit_dict['KoKg']
	mSLEoGt = fit_dict['mSLEoGt']
	smax = fit_dict['smax']

	# Generate a distribution of products of the above two factors
	pcoKg = (pcoK[0]+np.random.standard_normal([nr,nt,1])*pcoK[1])*\
		(KoKg[0]+np.random.standard_normal([nr,nt,1])*KoKg[1])
	meansmb = 1923 # model-mean time-mean 1979-2010 Gt yr-1 from 13.3.3.2
	moaoKg = -pcoKg * 1e-2 * meansmb * mSLEoGt # m yr-1 of SLE per K of global warming

	if fraction is None:
		fraction=np.random.rand(nr,nt,1)
	elif fraction.size!=nr*nt:
		raise ProjectionError('fraction is the wrong size')
	else:
		fraction.shape=(nr,nt,1)


	ainterfactor = 1 - fraction * smax	
	
	antsmb = moaoKg * ainterfactor * zit.reshape(1,nt,-1)
	
	return antsmb


def project_greendyn(fit_dict, nm, nt, data_years):
	
	# Extract relevant parameters from the fit dictionary
	fgreendyn = fit_dict['fgreendyn']
	dgreen = fit_dict['dgreen']
	gdyn_finalrange = fit_dict['gdyn_finalrange']
	
	gdyn_startratemean = 0.63 * fgreendyn
	gdyn_startratepm = 0.17 * fgreendyn
	
	gdyn_timeprojection = time_projection(gdyn_startratemean, gdyn_startratepm, gdyn_finalrange, nm, nt, data_years)

	return(gdyn_timeprojection + fgreendyn * dgreen)


def project_antdyn(fit_dict, nm, nt, data_years, fraction=None):
	
	# Extract relevant parameters from the fit dictionary
	dant = fit_dict['dant']
	adyn_startratemean = fit_dict['adyn_startratemean']
	adyn_startratepm = fit_dict['adyn_startratepm']
	adyn_finalrange = fit_dict['adyn_finalrange']
	
	adyn_timeprojection = time_projection(adyn_startratemean, adyn_startratepm, adyn_finalrange, nm, nt, data_years, fraction=fraction)

	return adyn_timeprojection + dant


def time_projection(startratemean, startratepm, finalrange, nr, nt, data_years, nfinal=1, fraction=None):
	# Return projection of a quantity which is a quadratic function of time
	# startratemean, startratepm -- rate of GMSLR at the start in mm yr-1, whose
	#		likely range is startratemean +- startratepm
	# finalrange -- two-element list giving likely range in m for GMSLR at the end
	# nfinal -- int, optional, number of years at the end over which finalrange is
	#		a time-mean; by default 1 => finalrange is the value for the last year
	# fraction -- array-like, optional, random numbers in the range 0 to 1,
	#		by default uniformly distributed

	if fraction is None:
		fraction=np.random.rand(nr,nt,1)
	elif fraction.size!=nr*nt:
		raise ProjectionError('fraction is the wrong size')
	fraction = fraction.reshape(nr,nt,1)
	
	# Number of years
	nyr = len(data_years)

	# For terms where the rate increases linearly in time t, we can write GMSLR as
	#		S(t) = a*t**2 + b*t
	# where a is 0.5*acceleration and b is start rate. Hence
	#		a = S/t**2-b/t
	momm = 1e-3 # convert mm yr-1 to m yr-1
	startrate = (startratemean + startratepm * np.array([-1,1], dtype=np.float)) * momm
	finalyr = np.arange(nfinal) - nfinal + nyr + 1 # last element ==nyr
	# If nfinal=1, the following is equivalent to
	# np.array(finalrange,dtype=np.float)/nyr**2-startrate/nyr
	acceleration = (np.array(finalrange, dtype=np.float) - startrate * finalyr.mean()) / (finalyr**2).mean()

	# Create a field of elapsed time in years
	time = data_years
	time=time-time[0]+1 # years since start

	# Calculate two-element list containing fields of the minimum and maximum
	# timeseries of projections, then calculate random ensemble within envelope
	range = [float(acceleration[i]) * (time**2) + float(startrate[i]) * time for i in [0,1]]
	
	projection = (range[0] * (1 - fraction) + range[1] * fraction)

	return projection



def ar5_project_icesheets(rng_seed, nmsamps, ntsamps, pipeline_id):
	
	# Define the target years
	targyears = np.arange(2010,2101,10)
	
	# Load the preprocessed data
	data_file = "{}_data.pkl".format(pipeline_id)
	try:
		f = open(data_file, 'rb')
	except:
		print("Cannot open data file\n")
	
	# Extract the data variables
	my_data = pickle.load(f)
	f.close()
	
	temp_mean = my_data['temp_mean']
	temp_sd = my_data['temp_sd']
	inttemp_mean = my_data['inttemp_mean']
	inttemp_sd = my_data['inttemp_sd']
	data_years = my_data['data_years']
	startyr = my_data["startyr"]
	scenario = my_data['scenario']
	
	# Load the fit data
	data_file = "{}_fit.pkl".format(pipeline_id)
	try:
		f = open(data_file, 'rb')
	except:
		print("Cannot open fit file\n")
	
	# Extract the fit variables
	my_fit = pickle.load(f)
	f.close()
	
	# Set the seed for the random number generator
	np.random.seed(rng_seed)

	# Generate perfectly correlated samples
	z=np.random.standard_normal(ntsamps)[:,np.newaxis]
	
	# For each quantity, mean + standard deviation * normal random number
	zt = temp_mean + (temp_sd * z)
	zit = inttemp_mean + (inttemp_sd * z)

	# Number of realizations
	nr = nmsamps
	
	# Number of years in the data record
	nyr = len(data_years)
	
	# correlation between antsmb and antdyn
	fraction=np.random.rand(nmsamps * ntsamps) 
	
	# Project the SMB and Dynamics portions of each ice sheet
	greensmb=project_greensmb(zt, my_fit, nmsamps)
	greendyn=project_greendyn(my_fit, nmsamps, ntsamps, data_years)
	antsmb=project_antsmb(zit, my_fit, nmsamps, ntsamps, fraction=fraction)	
	antdyn=project_antdyn(my_fit, nmsamps, ntsamps, data_years, fraction=fraction)
	
	
	# Reduce the years to just the target years
	year_idx = np.isin(data_years, targyears)
	data_years = data_years[year_idx]
	greensmb = greensmb[:,:,year_idx]
	greendyn = greendyn[:,:,year_idx]
	antsmb = antsmb[:,:,year_idx]
	antdyn = antdyn[:,:,year_idx]
	
	# Total up components
	greennet = (greensmb + greendyn) * 1000   # Convert to mm
	antnet = (antdyn + antsmb) * 1000    # Convert to mm
	totalnet = greennet + antnet
	
	# Flatten the ice sheet samples
	greennet = greennet.reshape(-1, greennet.shape[-1])
	antnet = antnet.reshape(-1, antnet.shape[-1])
	totalnet = totalnet.reshape(-1, totalnet.shape[-1])
	
	# Write the netCDF output
	WriteNetCDF(greennet, "GIS", data_years, scenario, pipeline_id)
	WriteNetCDF(antnet, "AIS", data_years, scenario, pipeline_id)
	
	# Write out a total icesheets netcdf file
	WriteNetCDF(totalnet, "TIS", data_years, scenario, pipeline_id)
	
	# Load in the icesheet fraction data-------------------------------------------
	# Note: There's were derived from the Kopp14 workflow.  
	
	# Initialize the data structures
	ice_frac = []
	ice_region_names = []
	
	# Open the icesheet fraction file
	ice_frac_file = os.path.join(os.path.dirname(__file__), "icesheet_fraction.txt")
	with open(ice_frac_file, 'r') as fp:
		
		# Get the fraction years from the header line
		header_items = re.split(",\s*", fp.readline())
		ice_frac_years = np.array([int(x) for x in header_items[1:]])
		
		# Read in the rest of the files
		for line in fp:
			line = line.rstrip()
			
			# Split the line into the region name and the fractions then append to data structures
			line_parts = re.split(",\s*", line)
			ice_region_names.append(line_parts[0])
			ice_frac.append([float(x) for x in line_parts[1:]])
	
	# Convert the fraction data structure into a numpy array
	ice_frac = np.array(ice_frac)
	
	# Subset the fraction data to the years of interest
	year_idx = np.isin(ice_frac_years, data_years)
	ice_frac = ice_frac[:,year_idx]
	
	# Reshape the samples and fraction data structures for broadcasting
	antnet = antnet[:,np.newaxis,:]
	ice_frac = ice_frac[np.newaxis,:,:]
	
	# Apply the regional fractions to the global projections
	aissamps = antnet * ice_frac
	
	# Save the global glacier and ice caps projections to a pickle
	output = {"gissamps": greennet, "aissamps": antnet, "totsamps": totalnet, \
		"waissamps": aissamps[:,0,:], "eaissamps": aissamps[:,1,:], "data_years": data_years}
	outfile = open(os.path.join(os.path.dirname(__file__), "{}_projections.pkl".format(pipeline_id)), 'wb')
	pickle.dump(output, outfile)
	outfile.close()
	
	# Write the estimated east and west AIS contributions to netCDF files
	WriteNetCDF(aissamps[:,0,:], "WAIS", data_years, scenario, pipeline_id)
	WriteNetCDF(aissamps[:,1,:], "EAIS", data_years, scenario, pipeline_id)
	
	return(0)
	


def WriteNetCDF(icesamps, icetype, data_years, scenario, pipeline_id):
	
	# Flatten the samples
	icesamps = icesamps.T

	# Write the total global projections to a netcdf file
	nc_filename = os.path.join(os.path.dirname(__file__), "{0}_{1}_globalsl.nc".format(pipeline_id, icetype))
	rootgrp = Dataset(nc_filename, "w", format="NETCDF4")

	# Define Dimensions
	nyr = len(data_years)
	nsamps = icesamps.shape[1]
	year_dim = rootgrp.createDimension("years", nyr)
	samp_dim = rootgrp.createDimension("samples", nsamps)

	# Populate dimension variables
	year_var = rootgrp.createVariable("year", "i4", ("years",))
	samp_var = rootgrp.createVariable("sample", "i8", ("samples",))

	# Create a data variable
	samps = rootgrp.createVariable("samps", "f4", ("years", "samples"), zlib=True, least_significant_digit=2)
	
	# Assign attributes
	rootgrp.description = "Global SLR contribution from {} according to AR5 workflow"
	rootgrp.history = "Created " + time.ctime(time.time())
	rootgrp.source = "FACTS: {0} - {1}".format(pipeline_id, scenario)
	year_var.units = "[-]"
	samp_var.units = "[-]"
	samps.units = "mm"

	# Put the data into the netcdf variables
	year_var[:] = data_years
	samp_var[:] = np.arange(nsamps)
	samps[:,:] = icesamps

	# Close the netcdf
	rootgrp.close()	

	return(0)

	

if __name__ == '__main__':
	
	# Initialize the command-line argument parser
	parser = argparse.ArgumentParser(description="Run the icesheets projection stage for the AR5 SLR projection workflow",\
	epilog="Note: This is meant to be run as part of the Framework for the Assessment of Changes To Sea-level (FACTS)")
	
	# Define the command line arguments to be expected
	parser.add_argument('--nmsamps', help="Number of method samples to generate [default=1000]", default=1000, type=int)
	parser.add_argument('--ntsamps', help="Number of climate samples to generate [default=450]", default=450, type=int)
	parser.add_argument('--seed', help="Seed value for random number generator [default=1234]", default=1234, type=int)
	parser.add_argument('--pipeline_id', help="Unique identifier for this instance of the module")
	
	# Parse the arguments
	args = parser.parse_args()
	
	# Run the projection process on the files specified from the command line argument
	ar5_project_icesheets(args.seed, args.nmsamps, args.ntsamps, args.pipeline_id)
	
	exit()
