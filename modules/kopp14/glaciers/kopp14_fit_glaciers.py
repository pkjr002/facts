import numpy as np
import pickle
import os
import argparse
from cholcov import cholcov

''' kopp14_fit_glaciers.py

This script runs the glaciers fitting task for the Kopp 2014 workflow. 
This task generates the t-distributions for projecting glacier contributions.

Parameters: 
data_file = The glacier data/configuration file

Output: 

'''

def kopp14_fit_glaciers(data_file):
	
	# Load the data file
	try:
		f = open(data_file, 'rb')
	except:
		print("Cannot open data file\n")
	
	# Extract the data variables
	my_config = pickle.load(f)
	f.close()
	
	#rcp_scenario = my_config["rcp_scenario"]
	projGIC = my_config["projGIC"]
	projGICse = my_config["projGICse"]
	projGICyrs = my_config["projGICyrs"]
	projGICmodel = my_config["projGICmodel"]
	
	# Define the target years
	targyears = np.arange(2010, 2101, 10)
	
	# Big Assumption: Years available across all models are the same... 1851 - 2300
	GICyears = projGICyrs[:,0]	
	
	# Initialize variables to hold the distribution information
	meanGIC = np.full((projGIC.shape[1], len(targyears)), np.nan)
	NGIC = np.full(len(targyears), np.nan)
	T = np.full((projGIC.shape[1], projGIC.shape[1], len(targyears)), np.nan)
	
	for i in np.arange(0,len(targyears)):
		
		# Which target year are we working with
		targyear = targyears[i]
		
		# Find the temporal indices that span targyear by 4 years either side
		theseGICyearInds = np.flatnonzero((GICyears >= np.max((2000,targyear-4))) * (GICyears <= np.min((np.max(targyears), targyear+4))))
		
		# Take the mean of the GIC data over this time window
		targ = np.mean(projGIC[theseGICyearInds,:,:], 0).T
		targse = np.mean(projGICse[theseGICyearInds,:,:], 0).T
		
		# Find the number of good GIC models available (i.e. data in all regions over the time window)
		inds = np.flatnonzero(~np.isnan(np.sum(targ, 1)))
		NGIC[i] = len(inds)
		
		# Find the multi-model mean GIC
		meanGIC[:,i] = np.mean(targ[inds,:], 0)
		
		# Combine the multi-model variance with the model standard error
		covGIC = np.cov(targ[inds,:], rowvar=False)
		meanGICse = np.mean(targse[inds,:], 0)
		covGIC = covGIC + (np.diag(meanGICse)**2)
		T[:,:,i] = cholcov(covGIC)
	
	# Store the thermal expansion variables in a pickle
	output = {'meanGIC': meanGIC, 'T': T, 'NGIC': NGIC}
	outfile = open(os.path.join(os.path.dirname(__file__), "kopp14_glaciers_fit.pkl"), 'wb')
	pickle.dump(output, outfile)
	outfile.close()


if __name__ == '__main__':
	
	# Initialize the command-line argument parser
	parser = argparse.ArgumentParser(description="Run the glaciers fitting stage for the Kopp14 SLR projection workflow",\
	epilog="Note: This is meant to be run as part of the Kopp14 module within the Framework for the Assessment of Changes To Sea-level (FACTS)")
	
	# Define the command line arguments to be expected
	parser.add_argument('--data_file', help="Pickle file containing Glacier data produced from preprocessing stage [default=\'kopp14_glaciers_data.pkl\']", default='kopp14_glaciers_data.pkl')
	
	# Parse the arguments
	args = parser.parse_args()
	
	# Run the fitting process with the provided command line arguments
	kopp14_fit_glaciers(args.data_file)
	
	exit()