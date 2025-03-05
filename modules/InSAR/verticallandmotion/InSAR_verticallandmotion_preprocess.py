import numpy as np
import pickle as p
import os
import sys
import re
import argparse

''' InSAR_preprocess_verticallandmotion.py

This runs the preprocessing stage for the vertical land motion component of the InSAR
workflow.

Parameters:
pipeline_id = Unique identifier for the pipeline running this code
inputtype = Type of input for the module (grid_insar, grid_gps, hires)

'''

def DataCheck(x):
	if("NaN" in x.strip()):
		return(np.nan)
	else:
		return(float(x))


def ReadVLM(infile):

	# Initialize lists to hold data
	lats = []
	lons = []
	vlm_rates = []
	sigmas = []

	# Name of the VLM data file
	with open(infile, 'r') as f:

		# Skip the header line
		line = f.readline()

		# Loop through the lines
		for line in f:

			# Strip the white space off the end of the line
			line.rstrip()

			# Get the pieces of data from the line
			lp = re.split("\s+", line)

			# Assign the data to their respective lists
			lons.append(float(lp[0]))
			lats.append(float(lp[1]))
			vlm_rates.append(float(lp[2]))
			sigmas.append(float(lp[3]))



	# Convert lists to numpy arrays
	lons = np.array(lons)
	lats = np.array(lats)
	vlm_rates = np.array(vlm_rates)
	sigmas = np.array(sigmas)
	
	# Return the arrays
	return(lats, lons, vlm_rates, sigmas)


def InSAR_preprocess_verticallandmotion(pipeline_id):

	# Define the input file based on input type
	#type2file = {"grid_insar": "VLM-grid_insar.dat", "grid_gps": "VLM-grid_gps.dat", "hires": "coast_ins.dat"}
	#if inputtype not in type2file.keys():
	#	raise Exception("Input type not recognized: {}".format(inputtype))
	inputfile = os.path.join(os.path.dirname(__file__), "InSAR_rate_File.txt")

	# Read input file
	(lats, lons, vlm_rates, sigmas) = ReadVLM(inputfile)
	rates = vlm_rates
	
	
	outdata = {'lats': lats, 'lons': lons, 'rates': -1*rates, 'sds': sigmas}

	# Define the data directory
	outdir = os.path.dirname(__file__)

	# Write the rates data to a pickle file
	outfile = open(os.path.join(outdir, "{}_data.pkl".format(pipeline_id)), 'wb')
	p.dump(outdata, outfile)
	outfile.close()


if __name__ == '__main__':

	# Initialize the command-line argument parser
	parser = argparse.ArgumentParser(description="Run the pre-processing stage for the InSAR vertical land motion workflow",\
	epilog="Note: This is meant to be run as part of the InSAR module within the Framework for the Assessment of Changes To Sea-level (FACTS)")

	# Define the command line arguments to be expected
	parser.add_argument('--pipeline_id', help="Unique identifier for this instance of the module")
	#parser.add_argument('--min_qf', help="Minimum value of data quality to use (default = 5)", type=float, default=5.0)


	# Parse the arguments
	args = parser.parse_args()

	# Run the preprocessing stage with the user defined RCP scenario
	InSAR_preprocess_verticallandmotion(args.pipeline_id)

	# Done
	exit()
