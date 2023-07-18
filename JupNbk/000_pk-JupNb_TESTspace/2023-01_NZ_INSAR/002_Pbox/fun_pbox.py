import numpy as np
import pandas as pd
import xarray as xr
from mpl_toolkits.basemap import Basemap
import glob
import os
import shutil
import re
import cartopy
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import fnmatch
#
PD=os.getcwd(); #PD




# Function block

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# create folder with JnB dir.
def FOcr8(folder_name,subfolder_name=None,subsubfolder_name=None):
    # folder_path, subfolder_path , subsubfolder_path = FOcr8("000_full_sample_workflows","wf_1e","ssp585")
    # I/P: folder_name = '000_full_sample_components'
    folder_path = os.path.join(PD, folder_name)
    if os.path.exists(folder_path): shutil.rmtree(folder_path)
    os.mkdir(folder_path)
    # SUB Folder
    if subfolder_name is not None:
        subfolder_path = os.path.join(folder_path, subfolder_name)
        if os.path.exists(subfolder_path): shutil.rmtree(subfolder_path)
        os.mkdir(subfolder_path)
        # Sub Sub Folder
        if subsubfolder_name is not None:
            subsubfolder_path = os.path.join(subfolder_path, subsubfolder_name)
            if os.path.exists(subsubfolder_path): shutil.rmtree(subsubfolder_path)
            os.mkdir(subsubfolder_path)
            #
            return folder_path, subfolder_path, subsubfolder_path   
            #
        return folder_path, subfolder_path
    else:
        return folder_path
# ^^^

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def FOcr8_1(path, folder_name):
    folder_path = os.path.join(path, folder_name)
    if os.path.exists(folder_path): shutil.rmtree(folder_path)
    os.mkdir(folder_path)
    # Return the folder path
    return folder_path
# ^^^


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# deletes all files in a folder that match a particular pattern, except those that contain both pattern and pattern2:
def delete_files_with_pattern(folder, pattern, pattern2):
    for filename in os.listdir(folder):
        if fnmatch.fnmatch(filename, pattern):
            if fnmatch.fnmatch(filename, pattern2):
                continue  # Skip files matching the exclusion pattern
            file_path = os.path.join(folder, filename)
            os.remove(file_path)
            # print(f"Deleted file: {file_path}")
# ^^^

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Copy all files to the FOLDER that match a pattern
def cp(source_dir,destination_dir,pattern):
    # source_dir = expF
    # destination_dir = folder_path
    source_file_pattern = os.path.join(source_dir, pattern)
    matching_files = glob.glob(source_file_pattern)
    for file_path in matching_files:
        shutil.copy2(file_path, destination_dir)
# ^^^   
  

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Copy all files in a folder to another folder
def cp_dir2dir(srcDIR,dstnDIR):
    file_list = os.listdir(srcDIR)
    for file_name in file_list:
        source_file = os.path.join(srcDIR, file_name)
        destination_file = os.path.join(dstnDIR, file_name)
        shutil.copy2(source_file, destination_file)
# ^^^   
 
 
 
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def fileNAME0(patH,name):
    folder_path = patH
    search_term = name   # replace with the word you want to search for
    file_pattern = f"{folder_path}/*{search_term}*"  # create a pattern to match files containing the search term
    matching_files = glob.glob(file_pattern)
    if len(matching_files)>1: 
        raise ValueError("There are 2 files with same keyword")
    fnme = os.path.basename(matching_files[0])
    return fnme
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