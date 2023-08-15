import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def plot_line(path,station,lineColor,ax=None):
    if ax is None:
        ax = plt.gca()
    #        
    d_nc=xr.open_dataset(path)
    #
    percentile  = d_nc['quantiles'].values * 100
    idx         = np.where(percentile == 50)[0]
    idx1        = np.where(percentile == 17)[0]
    idx2        = np.where(percentile == 83)[0] 
    #
    slc         = d_nc['sea_level_change'].values 
    time        = d_nc['years'].values
    idx_yr      = np.where((time >= 2020) & (time <= 2300))[0]
    #
    lat = d_nc['lat'][station].values
    lon = d_nc['lon'][station].values
    #
    xdata=time[idx_yr]
    ydata=slc[idx, idx_yr, station].reshape(-1)
    #
    line, = ax.plot(xdata,ydata, color=lineColor)
    #
    ax.set_title(f'datafile:: {os.path.basename(path)}',fontsize=6)
    ax.set_xlabel('Year', fontsize=6)
    ax.set_ylabel('SLC(mm)', fontsize=6)
    #
    ax.set_xlim(xdata.min()-10, xdata.max()+10)
    xticks=np.arange(xdata[0],xdata[-1]+1,20)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks,fontsize=6, rotation=45)
    #
    ax.tick_params(axis='y', which='both', left=True, right=True)
    #
    txt=split_path(os.path.relpath(path),4)
    ax.text(0.015, 0.8, txt, fontsize=6, fontweight='normal', ha='left', va='center', transform=ax.transAxes)
    ax.text(0.015, 0.5, f'station={station}', fontsize=6, fontweight='normal', ha='left', va='center', transform=ax.transAxes)
    #
    ax.text(0.015, 0.45, f'lat={lat}', fontsize=6, fontweight='normal', ha='left', va='center', transform=ax.transAxes)
    ax.text(0.015, 0.4, f'lon={lon}', fontsize=6, fontweight='normal', ha='left', va='center', transform=ax.transAxes)
    #
    ax.grid(True)
    ax.grid(color='gray', linestyle='-', linewidth=0.25,alpha=0.25)
    return None





# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Split a path into desired lines.
def split_path(path, numSplit):
    segLen = len(path) // numSplit
    # Split the path into segments
    seg = [path[i * segLen: (i + 1) * segLen] for i in range(numSplit)]
    # if the path length is not evenly divisible by num_lines
    if len(path) % numSplit != 0:
        seg[-1] += path[numSplit * segLen:]
    #
    printable_path = '\n'.join(seg)
    return printable_path




# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# find the Station using lat/lon
def StnLoc(pathIP,pathTarg):
    #
    latIP,lonIP=LoadNC(pathIP)
    latTarg,lonTarg=LoadNC(pathTarg)
    #
    close_indices = find_close_indices(latIP, latTarg)
    return close_indices



def LoadNC(path):
    d_nc=xr.open_dataset(path)
    lat = d_nc['lat'].values
    lon = d_nc['lon'].values
    return lat,lon
    

def find_close_indices(latIP, latTarg, rtol=1e-6):
    close_indices = []
    for i, val1 in enumerate(latIP):
        if any(np.isclose(val1, latTarg, rtol=rtol)):
            close_indices.append(i)
    return np.array(close_indices)


