import os
import xarray as xr
import numpy as np
import glob
#====================
import numpy as np
import pandas as pd
import xarray as xr
import glob
import os
import shutil
import re
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import statsmodels.api as sm
from matplotlib.colors import ListedColormap, BoundaryNorm


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def plot(search_terms1,path20k,search_terms2,path2k,loc,yrST):
    #
    # 1k location and 20k samples.
    data20k = fileNAME(path20k, search_terms1)
    # OG 7k-loop location and 2k samples
    data2k = fileNAME(path2k, search_terms2)
    #
    # Extract SL variables for specific time
    d20k = xtract_data_4m_nc((path20k + data20k), 'sea_level_change', loc, yrST,yrEN)
    slc20k, time20k, lat_lon20k = d20k['slc'], d20k['time'], [item.item() for item in [d20k['lat'], d20k['lon']]]
    #
    d2k = xtract_data_4m_nc((path2k + data2k), 'sea_level_change', loc, yrST,yrEN)
    slc2k, time2k, lat_lon2k = d2k['slc'], d2k['time'], [item.item() for item in [d2k['lat'], d2k['lon']]]
    #
    plot_qqplot(time20k,slc20k, data20k, lat_lon20k, time2k, slc2k, data2k, lat_lon2k, yrST, yrEN)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def plot_qqplot(time20k, slc20k, data20k, lat_lon20k, time2k, slc2k, data2k, lat_lon2k, yrST,ax):
    fig, ax = plt.subplots(1, 1, figsize=(5, 2.25))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    #
    xx20k = slc20k[:, np.where(time20k == yrST)[0][0]]
    xx2k = slc2k[:, np.where(time2k == yrST)[0][0]]
    #
    sm.qqplot_2samples(xx20k, xx2k, line='45', ax=ax)
    ax.lines[0].set(marker='o', markersize=4, markerfacecolor='black', markeredgecolor='blue', markeredgewidth=0.25)
    #
    ax.set_xlabel('20k-samples (mm)', fontsize=8)
    ax.set_ylabel('2k-samples (mm)', fontsize=8)
    ax.set_title(f'Year {yrST}', fontsize=8)
    ax.tick_params(axis='both', labelsize=7)
    ax.grid(True)
    #
    location = [
        f"lat/lon20k = {str(lat_lon20k)}",
        f"lat/lon2k = {str(lat_lon2k)}"
    ]
    text = "\n".join(location)
    ax.text(0.65, 0.2, text, fontsize=5.5, fontweight='normal', ha='left', va='center', transform=ax.transAxes)
    #
    data = 'Data:: ' + data20k.split('/')[-1] + '\n' + data2k.split('/')[-1]
    fig.text(0.5, 1.15, data, fontsize=6, ha='center', va='center', color='white',
             bbox={'facecolor': 'green', 'edgecolor': 'white', 'pad': 10})
    #
    plt.show()
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def mark_quantiles(ax, data, label):
    quantiles = np.percentile(data, [17, 50, 83])
    colors = ['red', 'blue', 'green']
    linestyles = ['dashed', 'solid', 'dotted']
    
    for i, quantile in enumerate(quantiles):
        ax.axhline(quantile, color=colors[i], linestyle=linestyles[i], label=f'{int(quantile)}th Quantile ({label})')
        ax.axvline(quantile, color=colors[i], linestyle=linestyles[i])
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def xtract_data_4m_nc(dataNC,var,loc,yrST,yrEN=None):    
    data = xr.open_dataset(dataNC)
    # index for years.
    time_data = data['years'].values    
    #
    if yrEN is None:
        idx_year = np.where(time_data == yrST)[0]
    else:
        idx_year = np.where((time_data >= yrST) & (time_data <= yrEN))[0] # range of years
    #
    time=time_data[idx_year]
    slc = data[var][:,idx_year,loc].values
    lat = np.around(data['lat'][loc].values, decimals=2)
    lon = np.around(data['lon'][loc].values, decimals=2)
    #
    output = {
        'slc': slc, 'time': time,
        'lat': lat, 'lon': lon
    }
    return output
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# search filename in Folder, multiple search terms    (need o update to match multiple)
def fileNAME(folder_path, search_terms):
    # _________________________________________________________
    #use::  search_terms = ["term1", "term2", "term3",.....] 
    #       file = fileNAME(folder_path, search_terms)
    #_________________________________________________________
    matching_files = []
    for file_path in glob.glob(f"{folder_path}/*"):
        file_name = os.path.basename(file_path)
        if all(term in file_name for term in search_terms):
            matching_files.append(file_path)
    if len(matching_files) > 1:
        raise ValueError("There are 2 or more files with the same keyword")
    elif len(matching_files) == 0:
        raise ValueError("No files found with the specified search terms")
    fnme = os.path.basename(matching_files[0])
    return fnme
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^