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
    ax.scatter(x1, y1, marker='s', edgecolor='red', linestyle='None', s=10, facecolor='none',label=plot_info['label1'])
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
            xgrid_min, xgrid_max, ygrid_min, ygrid_max, linspace_int,
            kde_min_tolerance,CMAP, cbar_num_ticks, 
            xlim_min, xlim_max ,xlim_increment, ylim_min, ylim_max ,ylim_increment,
            COMPONENT,ax,fig,font,
            kde_cbar_min=None,kde_cbar_max=None):
    # ........................................
    # Compute the KDE
    kde  = gaussian_kde([VAR1, VAR2])
    #
    if xgrid_min is None: xgrid_min = min(VAR1)
    if xgrid_max is None: xgrid_max = max(VAR1) 
    if ygrid_min is None: ygrid_min = min(VAR2)
    if ygrid_max is None: ygrid_max = max(VAR2) 
    #
    xgrid = np.linspace(xgrid_min, xgrid_max, linspace_int)  
    ygrid = np.linspace(ygrid_min, ygrid_max, linspace_int)  
    X, Y  = np.meshgrid(xgrid, ygrid)
    #
    # Evaluate the KDE on this grid
    Z = kde([X.flatten(), Y.flatten()]).reshape(X.shape)
    #
    kde_cbar_min = max(Z.min(), kde_min_tolerance) if kde_cbar_min is None else kde_cbar_min
    kde_cbar_max = Z.max() if kde_cbar_max is None else kde_cbar_max
#     kde_cbar_max = max(Z.max(), kde_cbar_min * 100) if Z.max() is not None else kde_cbar_min * 100

    #
    # Use logarithmic norm
    norm = LogNorm(vmin=kde_cbar_min, vmax=kde_cbar_max)
    #
    # Plot the KDE
    cax = ax.pcolormesh(X, Y, Z, shading='auto', norm=norm, cmap=CMAP)
    #
    #
    # Create the color bar
    cbar = fig.colorbar(cax, ax=ax)
    cbar_num_ticks=cbar_num_ticks
#     tick_values = np.logspace(np.log10(kde_cbar_min), np.log10(kde_cbar_max), num=cbar_num_ticks)
    tick_values = np.logspace(np.log10(kde_cbar_min), np.log10(kde_cbar_max), num=cbar_num_ticks)
    cbar.set_ticks(tick_values)
    #
    # Adjust label values.
#     cbar.set_ticklabels(['{:.5f}'.format(tick) for tick in tick_values])
    cbar.set_ticklabels(['{:.1e}'.format(tick) if tick < 0.0001 else '{:.4f}'.format(tick) for tick in tick_values])
    for label in cbar.ax.get_yticklabels():
        label.set_rotation(-45)
    #
    #
    # Set titles and labels
    ax.set_title(f"{COMPONENT} contribution to GMSL in {TVAR1} \n and that in {TVAR2}", fontsize=font)
    ax.set_xlabel(f"{COMPONENT} contribution in {TVAR1} (cm)", fontsize=font)
    ax.set_ylabel(f"{COMPONENT} contribution in {TVAR2} (cm)", fontsize=font)
    # Set axis limits and ticks
    if xlim_min is None: xlim_min=xgrid_min 
    if xlim_max is None: xlim_max=xgrid_max
    ax.set_xlim(xlim_min, xlim_max)
    x_ticks = np.arange(xlim_min, xlim_max + 1, xlim_increment)
    ax.set_xticks(x_ticks) 
    ax.set_xticklabels(['{:.2f}'.format(tick) for tick in x_ticks], fontsize=font, rotation=45)
    if ylim_min is None: ylim_min=ygrid_min 
    if ylim_max is None: ylim_max=ygrid_max
    ax.set_ylim(ylim_min, ylim_max)
    ax.set_yticks(np.arange(ylim_min, ylim_max+1, ylim_increment))
    ax.set_yticklabels(np.arange(ylim_min, ylim_max+1, ylim_increment), fontsize=font)
    # Add text
    ax.text(0.95, 0.95, VAR_name, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=font+1.5)
    #
#     plt.show()
# ^^^
    
    
      
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
# PLOT :: 1 component for multiple years.
#.............................................................
def plot_1file(component,VAR1_T1,VAR1_T2, VAR1_T3, VAR1_T4, T1,T2,T3,T4,
               xgrid_min, xgrid_max, ygrid_min, ygrid_max, linspace_int,
               kde_min_tolerance,CMAP, cbar_num_ticks,
               xlim_min, xlim_max ,xlim_increment, ylim_min, ylim_max ,ylim_increment,
               COMPONENT,font):
    data = [
        {"VAR1": VAR1_T1, "VAR2": VAR1_T4, "TVAR1": T1},
        {"VAR1": VAR1_T2, "VAR2": VAR1_T4, "TVAR1": T2},
        {"VAR1": VAR1_T3, "VAR2": VAR1_T4, "TVAR1": T3}
    ]
    # Set up the figure and grid
    fig = plt.figure(figsize=(15, 4))
    gs = fig.add_gridspec(1, 3)
    fig.subplots_adjust(wspace=0.4, hspace=0.4)

    # Loop to create subplots
    for i, item in enumerate(data):
        ax = fig.add_subplot(gs[0, i])
        log_plot(item["VAR1"], item["VAR2"], component, item["TVAR1"], T4, 
                xgrid_min, xgrid_max, ygrid_min, ygrid_max, linspace_int,
                kde_min_tolerance,CMAP, cbar_num_ticks, 
                xlim_min, xlim_max ,xlim_increment, ylim_min, ylim_max ,ylim_increment,
                COMPONENT,ax,fig,font)
    plt.show()
