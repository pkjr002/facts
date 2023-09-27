import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def showFIG(ax,ncpath,stnIDX,region,ssp,options):
    # OPTIONS
    xlab    = options.get('xlab', None)
    ylab    = options.get('ylab', None)
    x_min   = options.get('x_min', None)
    x_max   = options.get('x_max', None)
    x_int   = options.get('x_int', None)
    y_min   = options.get('y_min', None)
    y_max   = options.get('y_max', None)
    y_int   = options.get('y_int', None)
    decimal = options.get('decimal', None)
    #
    yrST    = options.get('yrST', None)
    yrEN    = options.get('yrEN', None)
    latIN   = options.get('latIN',None)
    lonIN   = options.get('lonIN',None)
    #
    grid        = options.get('grid',None)     # yes
    tick_pos    = options.get('tick_pos',None) #regular | all_axis
    font        = options.get('font', None)
    #
    plotFIG(ax,ncpath,stnIDX,region,ssp,xlab,ylab,x_min, x_max, y_min, y_max, x_int, y_int,decimal,yrST,yrEN,latIN,lonIN,grid,tick_pos,font)
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def plotFIG(ax,ncpath,stnIDX,region,ssp,xlab,ylab,x_min, x_max, y_min, y_max, x_int, y_int,decimal,yrST,yrEN,latIN,lonIN,grid,tick_pos,font):
    #::Get nc data
    slc, time, lat, lon, loc, samp, _ = ncREAD(ncpath)
#     slc_ptile, time, lat, lon, loc, qtile, ptile, _ = ncREAD(ncpath)
    #::Get percentile
    ptile, slc_ptile = getQUANT(slc)
    #::Get Index 
    idx_yr, idx_ptile, idx_ID = getINDEX(time,yrST,yrEN,ptile,stnIDX,lat,lon,latIN,lonIN)
    #________________________________________
    # PLOT:: 
    lines=[]; labels=[]
    #
    line, = ax.plot(time[idx_yr], slc_ptile[idx_ptile[0], idx_yr, idx_ID].reshape(-1), color=getCOLOR(ssp))
    lines.append(line); labels.append(f'{ssp} M')
    #
    ax.fill_between(time[idx_yr], slc_ptile[idx_ptile[1], idx_yr, idx_ID].reshape(-1), slc_ptile[idx_ptile[2], idx_yr, idx_ID].reshape(-1),color=getCOLOR(ssp), alpha=0.2)
    # 
    #________________________________________
    # Ax-Properties 
    ax_properties(ax,ssp,lines,labels,xlab,ylab, x_min, x_max, x_int, y_min, y_max,y_int,decimal,grid,tick_pos,font)
    #________________________________________
    # Mark Text on plot ::
    # Slc (y-values) based on x-axis
    for tt in [2100,yrEN]:
        for idx in idx_ptile:
            txt_value = slc_ptile[idx, np.where(time == tt)[0], stnIDX][0]
            txt0 = f'x'
            txt1 = f'  {txt_value:.2f} m'
            ax.text(tt, txt_value, txt0, fontsize=font-(0.15*font), ha='center', va='center', color=getCOLOR(ssp))
            ax.text(tt, txt_value, txt1, fontweight='bold', fontsize=font+(0.15*font), ha='left', va='center', color=getCOLOR(ssp))
    # TITLE
    txt0 = os.path.basename(ncpath)
    txt1=split_path(txt0, 2)
    ax.text(0.5, 1.06, txt1, transform=ax.transAxes, fontsize=font+(0.35*font), ha='center', va='center', color='black')
    # Location
    txt1=f'lat= {lat[stnIDX]}, \nlon= {lon[stnIDX]} \nstation = {loc[stnIDX]} \nstnIDX = {stnIDX}'
    ax.text(0.06, 0.6, txt1, transform=ax.transAxes, fontsize=font+(0.1*font), ha='left', va='center', color='black')
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def ax_properties(ax,ssp,lines,labels,xlab,ylab, x_min, x_max, x_int, y_min, y_max,y_int,decimal,grid,tick_pos,font):
    #
    ax.set_xlabel(xlab, fontsize=font+(0.35*font))    
    ax.set_ylabel(ylab, fontsize=font+(0.35*font))    
    # 
    if x_min or x_max or x_int is not None:
        ax.set_xlim(x_min, x_max)  
        x_ticks= np.arange(x_min, x_max+0.1, x_int).astype(int)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks,fontsize=font, rotation=45)
    #
    if y_min or y_max or y_int is not None:
        ax.set_ylim(y_min, y_max)
        if decimal is not None: y_ticks = np.around(np.arange(y_min, y_max, y_int), decimals=decimal)
        else: y_ticks = np.arange(y_min, y_max, y_int)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks, fontsize=font, rotation=0) 
    #
    tick_len=3.5;  tick_wid=1
    if tick_pos == 'regular':
        ax.tick_params(direction='in', length=tick_len, width=tick_wid, axis='both')
    if tick_pos == 'all_axis':
        ax.tick_params(direction='in', length=tick_len, width=tick_wid, axis='both', top=True, right=True)
    #
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1)
    #
    # 0 Horizontal line. 
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    #
    if grid=='yes':
        ax.grid(True, color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
    #
    ax.legend(lines + [ax.fill_between([], [], [], color='gray', alpha=0.2)],
                 labels + ['Shading is 17-83 percentile'], loc='upper left', fontsize=font+(0.25*font))
    #                 
    #txt = f'{name}\n(Site {stnIDX})'
    #ax.text(0.03, 0.62, txt,color='blue',transform=ax.transAxes, verticalalignment='top', fontsize='20', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    #ax.set_title(,fontsize=25)
    #
    return(None)
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def getCOLOR(ssp_scenario):
    #print("Value passed to getCOLOR:", ssp_scenario)
    colors = {
        'ssp119': np.array([0, 173, 207]) / 255,
        'ssp126': np.array([23, 60, 102]) / 255,
        'ssp245': np.array([247, 148, 32]) / 255,
        'ssp370': np.array([231, 29, 37]) / 255,
        'ssp585': np.array([149, 27, 30]) / 255
    }
    return colors.get(ssp_scenario)
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def ncREAD(ncpath):
    with xr.open_dataset(ncpath) as d_nc:
        valid_varnames = ["sea_level_change", "sea_level_change_rate"]
        d_nc_varname = None
        for this_varname in valid_varnames:
            if this_varname in d_nc.keys(): 
                d_nc_varname = this_varname
                break
        if d_nc_varname is None:
            print(f'No valid variable name exists in {os.path.basename(ncpath)}')
            return None
        #
        slc  = d_nc[d_nc_varname].values / 1000  #(meter)
        #
        time    = d_nc['years'].values
        lat     = d_nc['lat'].values
        lon     = d_nc['lon'].values
        loc     = d_nc['locations'].values
        samp    = d_nc['samples'].values
#         qtile   = d_nc['quantiles'].values 
#         ptile   = qtile*100
        #     
#     return slc, time, lat, lon, loc, qtile, ptile, d_nc
    return slc, time, lat, lon, loc, samp, d_nc
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def getQUANT(slc):
    qtile     = np.unique(np.append(np.round(np.linspace(0,1,101), 3), (0.001, 0.005, 0.01, 0.05, 0.167, 0.5, 0.833, 0.95, 0.99, 0.995, 0.999)))
    slc_ptile = np.nanquantile(slc, qtile, axis=0)
    ptile     = qtile*100
    return ptile, slc_ptile
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def getINDEX(time,yrST,yrEN,ptile,stnIDX,lat,lon,latIN,lonIN):
    #________
    yrST    = yrST if yrST is not None else 2020
    yrEN    = yrEN if yrEN is not None else 2150
    idx_yr  = np.where((time >= yrST) & (time <= yrEN))[0]
    #________
    idx_ptile = np.vstack([np.where(ptile == val)[0] for val in [50, 17, 83]])
    #________ 
    if latIN is not None and lonIN is not None: 
        idx_latIN, = np.where(lat == latIN)
        idx_lonIN, = np.where(lon == lonIN)
        #
        if len(idx_latIN) > 1 or len(idx_lonIN) > 1: 
            raise ValueError('Multiple Lat match')
        if idx_latIN != idx_lonIN: 
            raise ValueError('Indices for latitude and longitude do not match ...')
        #
        idx_ID = idx_latIN
    else:
        idx_ID = stnIDX
    #
    #
    return idx_yr, idx_ptile, idx_ID
#
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
#