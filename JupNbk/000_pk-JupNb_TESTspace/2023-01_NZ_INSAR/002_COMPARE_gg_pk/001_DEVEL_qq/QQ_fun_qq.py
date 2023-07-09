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
def plot(search_terms1,path20k,search_terms2,path2k,loc=0,worldmap=0):

    # 1k location and 20k samples.
    data20k = fileNAME(path20k, search_terms1)

    # OG 7k-loop location and 2k samples
    data2k = fileNAME(path2k, search_terms2)

    # Extract SL variables for specific time
    d20k = xtract_data_4m_nc((path20k + data20k), 'sea_level_change', loc, 2020, 2100)
    slc20k = d20k['slc']
    time20k = d20k['time']
    lat_lon20k = [item.item() for item in [d20k['lat'], d20k['lon']]]

    d2k = xtract_data_4m_nc((path2k + data2k), 'sea_level_change', loc, 2020, 2100)
    slc2k = d2k['slc']
    time2k = d2k['time']
    lat_lon2k = [item.item() for item in [d2k['lat'], d2k['lon']]]
    
    # .................................................................................
    # PLOT
    # ....
    if worldmap==1:
        plot_wm(lat_lon20k[0],lat_lon20k[1])
    else:
        # Plot the QQ Plot
        plot_qqplot(time20k, slc20k, slc2k, data20k, data2k,lat_lon20k,lat_lon2k)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def plot_wm(lat,lon):
    #
    m = Basemap(projection='merc', llcrnrlat=-50, urcrnrlat=-30, llcrnrlon=160, urcrnrlon=180, resolution='l')
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries(linewidth=0.5)
    x, y = m(lon, lat)
    m.plot(x, y, 'ro', markersize=4)
    parallels = np.arange(-90, 90, 5) 
    meridians = np.arange(-180, 180, 5) 
    m.drawparallels(parallels, labels=[False, True, False, False], linewidth=1, color='black', fontsize=6)
    m.drawmeridians(meridians, labels=[False, False, False, True], linewidth=1, color='black', fontsize=6)
    #
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    #
    plt.show()
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# ........................................................................................................
def plot_qqplot(time20k, slc20k, slc2k, data20k, data2k,lat_lon20k,lat_lon2k):
    num_cols = 2
    num_rows = 1
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(11.25, 2.25))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    axno = 0
    #
    # Loop over each year.
    for yr0, yr1 in enumerate(time20k):
        # Select var
        if yr1 == 2020 or yr1 == 2100:
            xx20k = slc20k[:, yr0]
            xx2k = slc2k[:, yr0]
            #            
            ax = axes[axno]
            sm.qqplot_2samples(xx20k, xx2k, line='45', ax=ax)
            ax.lines[0].set(marker='o', markersize=4, markerfacecolor='black', markeredgecolor='blue', markeredgewidth=0.25)
            #
            # Get the quantiles.
            ptile=[5, 50, 95]
            quantiles1 = np.percentile(xx20k,ptile);    quantiles2 = np.percentile(xx2k, ptile)
            line_colors = [(1.0, 0.6, 0.2), (0.6, 0.2, 1.0), (0.2, 1.0, 0.2)]; # line_colors = ['red', 'blue', 'green']
            #
            # Mark percentiles.
            for i, (q1, q2) in enumerate(zip(quantiles1, quantiles2)):
                color = line_colors[i]
                ax.axvline(q2, color=color, linestyle='dashed')
                ax.axhline(q1, color=color, linestyle='dashed')
            #
            ax.set_xlabel('20k-samples (mm)', fontsize=8)
            ax.set_ylabel('2k-samples (mm)', fontsize=8)
            ax.set_title(f'Year {yr1}', fontsize=8)
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
            # add quant details.
            q1_round = [round(q, 2) for q in quantiles1]; q2_round = [round(q, 2) for q in quantiles2]
            table_text = f'Quantiles:{ptile}\nX: {q2_round}\nY: {q1_round}'
            ax.text(0.05, 0.95, table_text, transform=ax.transAxes, verticalalignment='top', fontsize='5', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            #
            axno += 1   
            #
    # TITLE Data
    data = 'Data:: ' + data20k.split('/')[-1] + '\n' + data2k.split('/')[-1]
    fig.text(0.5, 1.05, data, fontsize=8, ha='center', va='center', color='white', bbox={'facecolor': 'green', 'edgecolor': 'white', 'pad': 10})
    #
    # Create legend
    legend_labels = [f'{pt} Percentile' for pt in ptile]
    legend_lines = [plt.Line2D([0], [0], color=color, linestyle='dashed') for color in line_colors]
    fig.legend(legend_lines, legend_labels, loc='upper right', bbox_to_anchor=(0.9, 1.15),fontsize='xx-small')
    #
    # plt.tight_layout() #plt.legend()
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
    lat = np.around(data['lat'][loc].values, decimals=2)
    lon = np.around(data['lon'][loc].values, decimals=2)
    
    output = {
        'slc': slc, 'time': time,
        'lat': lat, 'lon': lon
    }
    return output
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^