import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import glob
import os
import shutil
import re
import cartopy
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
#
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap, BoundaryNorm
#
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
#
#
current_directory=os.getcwd(); current_directory
# ==================================================================================================



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# FUNCTION BLOCK.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
def list_files_with_names(path, names):    
    if not isinstance(names, list):
        raise ValueError("The 'names' argument should be a list of strings.")
    #
    file_list = os.listdir(path)
    file_list = sorted(file_list)
    #
    matching_files = []
    for filename in file_list:
        if all(name in filename for name in names):
            matching_files.append(filename)
    return matching_files





# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def idx_1yr(yr=None,Dray=None,Tray=None):
    var=Dray[:,np.where(Tray == yr)[0][0]]
    return var
# ^^^


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def index_year_range(yrST=None, yrEN=None, Darray=None):
    if yrST is not None and yrEN is not None:
        idx = np.where((Darray >= yrST) & (Darray <= yrEN))[0]
    elif yrST is not None:
        idx = np.where(Darray >= yrST)[0]
    elif yrEN is not None:
        idx = np.where(Darray <= yrEN)[0]
    else:
        raise ValueError("Provide:: yrST or yrEN")
    return idx
# ^^^


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def extract_nc_info(nc_path, stn, unit, yrST=None, yrEN=None):
    #
    nc_data = xr.open_dataset(nc_path, engine='netcdf4')
    #
    time = nc_data['years'].values
    idx = index_year_range(yrST, yrEN, time)
    time = time[idx]
    #
    UNIT_CONVERSION = {'m': 1000, 'cm': 10, 'mm': 1}
    slc = nc_data['sea_level_change'].values[:, idx, stn] / UNIT_CONVERSION[unit]
    #
    lat = np.around(nc_data['lat'][stn].values, decimals=2)
    lon = np.around(nc_data['lon'][stn].values, decimals=2)
    #
    return nc_data, slc, time, lat, lon
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def sub_plot(ax, x1, y1, x2 , y2, plot_info):
    # ........................................
    sspcolors = {
    'ssp245': np.array([247, 148, 32]) / 255,
    'ssp370': np.array([231, 29, 37]) / 255,
    'ssp585': np.array([149, 27, 30]) / 255
    }
    # ........................................
    ax.scatter(x1, y1, marker='s', edgecolor='red', linestyle='None', s=1, facecolor='none',label=plot_info['label1'])
    ax.scatter(x2, y2, marker='o', edgecolor='blue', linestyle='None', s=1, facecolor='black',label=plot_info['label2'])
    #
    ax.set_xlabel(plot_info['x_label'], fontsize=6)
    ax.set_ylabel(plot_info['y_label'], fontsize=6)
    ax.set_title(plot_info['title'], fontsize=6)
    #
    ax.set_xlim(plot_info['x_lim'])
    ax.set_ylim(plot_info['y_lim'])
    #
    ax.set_xticks(plot_info['x_ticks']);  ax.set_xticklabels(plot_info['x_ticks'],fontsize=20, rotation=45)
    ax.set_yticks(plot_info['y_ticks']);  ax.set_yticklabels(plot_info['y_ticks'],fontsize=20) 
    #
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='both', direction='in', right=True, top=True)
    ax.legend()




# ^^^


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def log_plot(VAR1,VAR2,VAR_name,TVAR1,TVAR2,
             xgrid_min, xgrid_int, xgrid_max, ygrid_min, ygrid_int, ygrid_max,
             kde_min_tolerance,CMAP, cbar_num_ticks, 
            xlim_min, xlim_max_plus, xlim_max ,xlim_increment, 
            ylim_min, ylim_max_plus, ylim_max ,ylim_increment,
             COMPONENT,ax,fig,
            kde_min,kde_max):
    # ........................................
    #
    # Compute the KDE
    kde  = gaussian_kde([VAR1, VAR2])
    #
    # Adjusted to match axis limits
    xgrid = np.linspace(xgrid_min, xgrid_int, xgrid_max)  
    ygrid = np.linspace(ygrid_min, ygrid_int, ygrid_max)  
    X, Y  = np.meshgrid(xgrid, ygrid)
    #
    # Evaluate the KDE on this grid
    Z = kde([X.flatten(), Y.flatten()]).reshape(X.shape)
    #
    #
#     kde_min = Z.min()
#     kde_max = Z.max()
#     kde_min = max(kde_min, kde_min_tolerance)
#     kde_max = max(kde_max, kde_min * 10)
    #
    # Use logarithmic norm
    norm = LogNorm(vmin=kde_min, vmax=kde_max)
    #
    # Plot the KDE
    cax = ax.pcolormesh(X, Y, Z, shading='auto', norm=norm, cmap=CMAP)
    #
    #
    # Create the color bar
    cbar = fig.colorbar(cax, ax=ax)
    cbar_num_ticks=cbar_num_ticks
#     tick_values = np.logspace(np.log10(kde_min), np.log10(kde_max), num=cbar_num_ticks)
    tick_values = np.logspace(np.log10(kde_min), np.log10(kde_max), num=cbar_num_ticks)
    cbar.set_ticks(tick_values)
    #
    # Adjust label values.
    cbar.set_ticklabels(['{:.3f}'.format(tick) for tick in tick_values])
#     cbar.set_ticklabels(['{:.4e}'.format(tick) if tick < 0.0001 else '{:.4f}'.format(tick) for tick in tick_values])
    #
    #
    # Set titles and labels
    ax.set_title(f"{COMPONENT} contribution to GMSL in {TVAR1} \n and that in {TVAR2}", fontsize=6)
    ax.set_xlabel(f"{COMPONENT} contribution in {TVAR1} (cm)", fontsize=6)
    ax.set_ylabel(f"{COMPONENT} contribution in {TVAR2} (cm)", fontsize=6)
    # Set axis limits and ticks
    ax.set_xlim(xlim_min, xlim_max)
    ax.set_ylim(ylim_min, ylim_max)
    ax.set_xticks(np.arange(xlim_min, xlim_max+xlim_max_plus, xlim_increment))
    ax.set_xticklabels(np.arange(xlim_min, xlim_max+xlim_max_plus, xlim_increment), fontsize=6, rotation=45)
    ax.set_yticks(np.arange(ylim_min, ylim_max+ylim_max_plus, ylim_increment))
    ax.set_yticklabels(np.arange(ylim_min, ylim_max+ylim_max_plus, ylim_increment), fontsize=6)
    # Add text
    ax.text(0.95, 0.95, VAR_name, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=10)
    #
#     plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
