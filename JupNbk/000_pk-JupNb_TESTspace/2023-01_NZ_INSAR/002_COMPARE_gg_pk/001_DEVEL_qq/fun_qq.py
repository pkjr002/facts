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
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.colors import ListedColormap, BoundaryNorm


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def plot(search_terms1,path20k,search_terms2,path2k):

    # 1k location and 20k samples.
    # path20k = '/scratch/pk695/FACTS/002_fork/facts/experimentsNZ.230608/nzOG/nzOG.ssp585.1kloc/output/'
    data20k = fileNAME(path20k, search_terms1)

    # OG 7k-loop location and 2k samples
    # path2k = '/scratch/pk695/FACTS/002_fork/facts/experimentsNZ/nzOG/nzOG.ssp585/output_local_ssp585/'
    data2k = fileNAME(path2k, search_terms2)

    # Extract SL variables for specific time
    d20k = xtract_data_4m_nc((path20k + data20k), 'sea_level_change', 0, 2020, 2100)
    slc20k = d20k['slc']
    time20k = d20k['time']

    d2k = xtract_data_4m_nc((path2k + data2k), 'sea_level_change', 0, 2020, 2100)
    slc2k = d2k['slc']
    time2k = d2k['time']

    # Plot the QQ Plot
    plot_qqplot(time20k, slc20k, slc2k, data20k, data2k)



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def mark_quantiles(ax, data, label):
    quantiles = np.percentile(data, [17, 50, 83])
    colors = ['red', 'blue', 'green']
    linestyles = ['dashed', 'solid', 'dotted']
    
    for i, quantile in enumerate(quantiles):
        ax.axhline(quantile, color=colors[i], linestyle=linestyles[i], label=f'{int(quantile)}th Quantile ({label})')
        ax.axvline(quantile, color=colors[i], linestyle=linestyles[i])


# ........................................................................................................
def plot_qqplot(time20k, slc20k, slc2k, data20k, data2k):
    num_cols = 2
    num_rows = 1
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    axno = 0
    
    # Loop over each year.
    for yr0, yr1 in enumerate(time20k):
        # Select var
        if yr1 == 2020 or yr1 == 2100:
            xx20k = slc20k[:, yr0]
            xx2k = slc2k[:, yr0]
            
            ax = axes[axno]
            sm.qqplot_2samples(xx20k, xx2k, line='45', ax=ax)
            ax.set_xlabel('20k-samples (mm)')
            ax.set_ylabel('2k-samples (mm)')
            ax.set_title(f'Year {yr1}')
            
            # mark_quantiles(ax, xx20k, '20k')
            # mark_quantiles(ax, xx2k, '2k')
            
            axno += 1
    
    # Plot Title
    # fig.text(0.5, 1.05, 'QQ-Plot', fontsize=14, fontweight='bold', ha='center', va='center')
    
    # Data
    data = 'Data:: ' + data20k.split('/')[-1] + '\n' + data2k.split('/')[-1]
    fig.text(0.5, 1.05, data, fontsize=12, ha='center', va='center', color='white',
             bbox={'facecolor': 'green', 'edgecolor': 'white', 'pad': 10})
    
    # plt.tight_layout()
    # plt.legend()
    plt.show()
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

# # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def xtract_data_4m_nc(dataNC,var,loc,yrST,yrEN):    
    data = xr.open_dataset(dataNC)
    # index for years.
    time_data = data['years'].values    
    idx_year=np.where((time_data >= yrST) & (time_data <= yrEN))[0]
    #
    time=time_data[idx_year]
    #
    slc = data[var][:,idx_year,loc].values
    lat=data['lat'][loc].values
    lon=data['lon'][loc].values
    
    output = {
        'slc': slc, 'time': time,
        'lat': lat, 'lon': lon
    }
    return output
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^