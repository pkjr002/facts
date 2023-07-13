import numpy as np
import pandas as pd
import xarray as xr
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
PD=os.getcwd(); PD
# ==================================================================================================

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def idx_1yr(yr=None,Dray=None,Tray=None):
    var=Dray[:,np.where(Tray == yr)[0][0]]
    return var
# ^^^


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def idx_yr(yrST=None, yrEN=None, Darray=None):
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
def nc_Xtrct(nc_path,stn,unit,yrST=None, yrEN=None):
    nc_nme = str(nc_path).split("/")[10].split("coupling")[1]
    nc_dat = xr.open_dataset(nc_path)
    # ..........................................
    time = nc_dat['years'].values
    idx = idx_yr(yrST, yrEN, time)
    def_unit = {'m': 1000, 'cm': 10, 'mm': 1}
    # ..........................................
    time = time[idx]
    slc = nc_dat['sea_level_change'].values[:,idx,stn] / def_unit[unit]
    lat = np.around(nc_dat['lat'][stn].values, decimals=2)
    lon = np.around(nc_dat['lon'][stn].values, decimals=2)
    #
    return nc_nme,nc_dat,slc,time,lat,lon
# ^^^


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def LSfile(path, name):
    nclist=os.listdir(path+'/');  nclist=sorted(nclist)
    nclist_local=[]
    for ncname in nclist:
        if name in ncname:
            nclist_local.append(ncname) 
    return(nclist_local)
# ^^^


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def sub_plot(ax, x1, x2, y1, y2, plot_info):
    # ........................................
    sspcolors = {
    'ssp245': np.array([247, 148, 32]) / 255,
    'ssp370': np.array([231, 29, 37]) / 255,
    'ssp585': np.array([149, 27, 30]) / 255
    }
    # ........................................
    ax.scatter(x1, y1, marker='s', edgecolor=sspcolors['ssp585'], linestyle='None', s=10, facecolor='none')
    ax.scatter(x2, y2, marker='o', edgecolor='black', linestyle='None', s=10, facecolor='black')
    #
    ax.set_xlabel(plot_info['x_label'], fontsize=6)
    ax.set_ylabel(plot_info['y_label'], fontsize=6)
    ax.set_title(plot_info['title'])
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




# ^^^
