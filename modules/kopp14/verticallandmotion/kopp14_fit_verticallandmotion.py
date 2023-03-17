import argparse

''' kopp14_fit_verticallandmotion.py

This runs the fitting stage for the vertical land motion component of the Kopp 14
workflow.

Parameters:
pipeline_id = Unique identifier for the pipeline running this code

Note: This is currently a NULL process. Rates are already read in during the preprocess
stage.

'''

def kopp14_fit_verticallandmotion(pipeline_id):

	return(0)

	
if __name__ == '__main__':	
	
	# Initialize the command-line argument parser
	parser = argparse.ArgumentParser(description="Run the fitting stage for the Kopp 14 vertical land motion workflow",\
	epilog="Note: This is meant to be run as part of the Kopp 14 module within the Framework for the Assessment of Changes To Sea-level (FACTS)")
	
	# Define the command line arguments to be expected
	parser.add_argument('--pipeline_id', help="Unique identifier for this instance of the module")
	
	# Parse the arguments
	args = parser.parse_args()
	
	# Run the preprocessing stage with the user defined RCP scenario
	kopp14_fit_verticallandmotion(args.pipeline_id)

	import psutil as ps
	peak_mem = ps.Process().memory_info().rss * 1e-9
	module_set = 'kopp14'
	mod_name = 'verticallandmotion'
	task_name = ['preprocess','fit','project','postprocess']
	f = open(f'{module_set}_{mod_name}_{task_name[1]}_memory_diagnostic.txt','w')
	f.write(f'This Task Used: {peak_mem} GB')
	f.close()
	
	# Done
	exit()