import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
from scipy.stats import linregress
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
from sklearn.neighbors import KernelDensity
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
            kde_min_tolerance,CMAP, lOg, cbar_num_ticks, 
            xlim_min, xlim_max ,xlim_increment, ylim_min, ylim_max ,ylim_increment,
            COMPONENT,ax,fig,font,
            kde_cbar_min=None,kde_cbar_max=None,
            isFirstColumn=False,isLastColumn=False,isTopPlot=False):
    # ........................................
    # Compute the KDE
    kde  = gaussian_kde([VAR1, VAR2])
    
    if xgrid_min is None: xgrid_min = min(VAR1)
    if xgrid_max is None: xgrid_max = max(VAR1) 
    if ygrid_min is None: ygrid_min = min(VAR2)
    if ygrid_max is None: ygrid_max = max(VAR2) 
    
    xgrid = np.linspace(xgrid_min, xgrid_max, linspace_int)  
    ygrid = np.linspace(ygrid_min, ygrid_max, linspace_int)  
    X, Y  = np.meshgrid(xgrid, ygrid)
    
    # Evaluate the KDE on this grid
    Z = kde([X.flatten(), Y.flatten()]).reshape(X.shape)
    
    # Colorbar limits.
    # kde_cbar_min = Z.min() if kde_cbar_min is None else kde_cbar_min
    kde_cbar_min = max(Z.min(), kde_min_tolerance) if kde_cbar_min is None else kde_cbar_min
    kde_cbar_max = Z.max() if kde_cbar_max is None else kde_cbar_max
    
    # Use logarithmic norm for the pcolormesh
    if lOg == 'LOG':
        norm = LogNorm(vmin=kde_cbar_min, vmax=kde_cbar_max)
        ## Plot the Normalized KDE
        cax = ax.pcolormesh(X, Y, Z, shading='auto', norm=norm, cmap=CMAP);
        if isLastColumn:
            cbar = fig.colorbar(cax, ax=ax)
            tick_values = np.logspace(np.log10(kde_cbar_min), np.log10(kde_cbar_max), num=cbar_num_ticks)
            cbar.set_ticks(tick_values)
            cbar.set_ticklabels(['{:.1e}'.format(tick) if tick < 0.0001 else '{:.4f}'.format(tick) for tick in tick_values])
            for label in cbar.ax.get_yticklabels():
                label.set_rotation(-45)

    elif lOg == 'LIN': 
        # Plot the KDE
        cax = ax.pcolormesh(X, Y, Z, shading='auto', cmap=CMAP, vmin=kde_cbar_min, vmax=kde_cbar_max);
        if isLastColumn:
            cbar = fig.colorbar(cax, ax=ax)
            tick_values = np.linspace(kde_cbar_min, kde_cbar_max, cbar_num_ticks)
            cbar.set_ticks(tick_values)
            cbar.set_ticklabels(['{:.1e}'.format(tick) if tick < 0.0001 else '{:.4f}'.format(tick) for tick in tick_values])
            for label in cbar.ax.get_yticklabels():
                label.set_rotation(-45)
       
    # Set titles and labels
    
    if isTopPlot:
        ax.set_title(f"{COMPONENT} contribution to GMSL in {TVAR1} \n and that in {TVAR2}", fontsize=font)
       
    #ax.set_xlabel(f"{COMPONENT} contribution in {TVAR1} (cm)", fontsize=font)
    if isFirstColumn: 
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
    if isFirstColumn:
        ax.text(0.1, 0.95, VAR_name, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=font+1.5)
    #
#     plt.show()
# ^^^
    
    
      
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
# PLOT :: 1 component for multiple years.
#.............................................................
def plot_1file(component, VAR1_T1, VAR1_T2, VAR1_T3, VAR1_T4, VAR1_T5, T1, T2, T3, T4,T5,
               xgrid_min, xgrid_max, ygrid_min, ygrid_max, linspace_int,
               kde_min_tolerance, CMAP, lOg,  cbar_num_ticks,
               COMPONENT, font, axis_limits, kde_cbar_min=None,kde_cbar_max=None,isTopPlot=None):
    data = [
        {"VAR1": VAR1_T1, "VAR2": VAR1_T5, "TVAR1": T1},
        {"VAR1": VAR1_T2, "VAR2": VAR1_T5, "TVAR1": T2},
        {"VAR1": VAR1_T3, "VAR2": VAR1_T5, "TVAR1": T3},
        {"VAR1": VAR1_T4, "VAR2": VAR1_T5, "TVAR1": T4}
    ]
    # Set up the figure and grid
    # fig = plt.figure(figsize=(15, 4)); gs = fig.add_gridspec(1, 3); 
    fig = plt.figure(figsize=(20, 4)); 
    gs = fig.add_gridspec(1, 4);
    fig.subplots_adjust(wspace=0.1, hspace=0.4);

    # Loop to create subplots
    for i, item in enumerate(data):
        ax = fig.add_subplot(gs[0, i]);
        xlim_min, xlim_max, xlim_increment = axis_limits[i]['xlim']
        ylim_min, ylim_max, ylim_increment = axis_limits[i]['ylim']
        isFirstColumn = (i == 0)
        isLastColumn = (i == len(data) - 1)
        if isTopPlot is None:
            isTopPlot = False 
        
        
        log_plot(item["VAR1"], item["VAR2"], component, item["TVAR1"], T5, 
                 xgrid_min, xgrid_max, ygrid_min, ygrid_max, linspace_int,
                 kde_min_tolerance, CMAP, lOg, cbar_num_ticks, 
                 xlim_min, xlim_max, xlim_increment, ylim_min, ylim_max, ylim_increment,
                 COMPONENT, ax, fig, font,kde_cbar_min,kde_cbar_max,
                 isFirstColumn=isFirstColumn,isLastColumn=isLastColumn,isTopPlot=isTopPlot)
    # plt.show()



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
# Function for plotting Conditional Probability.
#.............................................................
# Function for plotting
# def plot_ConditionalProb(ax, var1, var2, t1, t2, ssp, k, bw, linspace_int, scatter, cmap, plotOPT):
def plot_ConditionalProb(ax, var1, var2, t1, t2,var1_lab,var2_lab):
    # Common parameters
    ssp='ssp245'
    #
    k = 'gaussian'; bw = 1; linspace_int = 100; scatter = 'NO'; cmap = 'Reds'
    plotOPT = {'y_ax_min':-10, 'y_ax_max': 75, 'c_bar_min': 0.001, 'c_bar_max': 0.252, 'plotCBAR' : 'YES'}
    #
    #-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    # AXIS variables

    var1=var1[ssp]
    var2=var2[ssp]

    xaxVAR = var1['slc'][:, np.where(var1['time']==t1)[0][0]] 
    yaxVAR = var2['slc'][:, np.where(var1['time']==t2)[0][0]] 
    #
    # LABELS
    xaxLAB = f'{var1_lab}_{t1}'     #xaxLAB = f"{var1['filename'].split('.')[2]}_{t1}";  
    yaxLAB = f'{var2_lab}_{t2}'     #yaxLAB = f"{var2['filename'].split('.')[2]}_{t2}"
    #
    # title = f"{var2['filename'].split('.')[2]} ({var2['filename'].split('.')[-2]}) contribution in {t2} \n as a function of {t1} {var1['filename'].split('.')[2]} ({var1['filename'].split('.')[-2]}) contribution"
    title = f'{t2} {var2_lab}  \n as a function of \n {t1} {var1_lab} '
    #
    # PLOT
    gilford(ax, xaxVAR, yaxVAR, k, bw, linspace_int, 'density_values_Normalized', xaxLAB, yaxLAB, title, ssp, scatter, cmap, t1, plotOPT)





# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
# Plot contour that sums to 1 along columns.
#.............................................................
def gilford(ax, xaxVAR, yaxVAR,K,BW,linspace_int, val, xaxLAB,yaxLAB,title,ssp,scatter,CMAP,T1, plotOPT=None):

    # create 2D matrix.
    INdata = np.column_stack((xaxVAR, yaxVAR))

    # KDE
    kde = KernelDensity(kernel=K, bandwidth=BW).fit(INdata)

    # Create a grid of test points
    xgrid = np.linspace(INdata[:,0].min()-1, INdata[:,0].max()+1, linspace_int)  
    ygrid = np.linspace(INdata[:,1].min()-1, INdata[:,1].max()+1, linspace_int)  
    Xgrid, Ygrid  = np.meshgrid(xgrid, ygrid)
    grid_samples = np.vstack([Xgrid.ravel(), Ygrid.ravel()]).T

    # Eval density model on the grid (log likelihoods)
    log_density_values = kde.score_samples(grid_samples)
    #Reshape 
    log_density_values = log_density_values.reshape(Xgrid.shape)

    
# ==================================================================================== 
    # make the max matrix value 0.
    adjusted_log_density_values = log_density_values - np.max(log_density_values)
    
    #calculate adjusted density.
    adjusted_density_values = np.exp(adjusted_log_density_values)
    # Normalize
    normalized_density_values = adjusted_density_values / np.sum(adjusted_density_values, axis=0)
    
    # Get the TRUE Density values back
    density_values = adjusted_density_values*np.exp(np.max(log_density_values))
    
    # Normalize log_density_values 
    # do all of this in log scales (Log-Sum-Exp trick)
    exp_sum = np.sum(np.exp(adjusted_log_density_values), axis=0)
    # Readjust by adding back the adjustment factor
    log_normalization_constant = np.log(exp_sum) + np.max(log_density_values)
    # Normalize within the log space 
    normalized_log_density_values = log_density_values - log_normalization_constant



    if val == 'log_density_values':
        PLOT_VAR=log_density_values

    elif val == 'log_density_values_Normalized':
#         # Convert from log values
#         density_values = np.exp(log_density_values)
#         # Normalize
#         density_values_Normalized = density_values/density_values.sum(axis=0)
#         # Convert back to log values
#         log_density_values_Normalized = np.log(density_values_Normalized)
#         #
#         PLOT_VAR=log_density_values_Normalized    
        PLOT_VAR=normalized_log_density_values

    elif val == 'density_values':
#         density_values = np.exp(log_density_values)
        PLOT_VAR=density_values

    elif val == 'density_values_Normalized':
#         # Convert from log values
#         density_values = np.exp(log_density_values)
#         # Normalize
#         density_values_Normalized = density_values/density_values.sum(axis=0)
#         PLOT_VAR=density_values_Normalized    
        PLOT_VAR=normalized_density_values
# ====================================================================================

    #_-_-_-_-_-_- FILL contour and COLORbar limits
    if val in ['density_values' , 'density_values_Normalized']:
        if plotOPT is not None and 'c_bar_min' in plotOPT:
            clevels=np.linspace(plotOPT['c_bar_min'],plotOPT['c_bar_max'],10)    
        else:
            clevels=np.linspace(1e-3,PLOT_VAR.max(),10)
    else: 
        clevels=np.linspace(PLOT_VAR.min(),PLOT_VAR.max(),10)
        #
    clabels=np.round(clevels,decimals=3).astype('str')
    #
    contour=ax.contourf(Xgrid, Ygrid, PLOT_VAR,levels=clevels,cmap=CMAP)
    #
    #_-_-_-_-_-_- Overlay a SCATTER of the original data
    if scatter == 'YES':
        ax.scatter(INdata[:, 0], INdata[:, 1], s=.5, facecolor='red')
    #
    #_-_-_-_-_-_- AXIS properties
    ax.set_title(title,fontsize=8)
    ax.set_xlabel(xaxLAB)
    ax.set_ylabel(yaxLAB)
    #
    if plotOPT['plotCBAR'] == 'YES':
        cbar=plt.colorbar(contour,ax=ax,label=val,ticks=clevels,orientation='horizontal',pad=0.2)
        # cbar=plt.colorbar(contour,ax=ax,label=val,ticks=clevels,orientation='vertical',pad=0.1)
        cbar.set_label(label=val, size=10, weight='bold', color='blue')
        cbar.set_ticklabels(clabels)
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=45)
    #
    # ax.text(0.9, 0.1, f'{ssp}\n{T1}', fontsize=7, color='black', weight='bold', ha='center', va='center', transform=ax.transAxes)
    ax.text(0.9, 0.1, f'{ssp}', fontsize=7, color='black', weight='bold', ha='center', va='center', transform=ax.transAxes)
    #
    if plotOPT is not None and 'y_ax_min' in plotOPT:
        ax.set_ylim(plotOPT['y_ax_min'],plotOPT['y_ax_max'])
    

    # return PLOT_VAR, Xgrid, Ygrid, INdata
    # return PLOT_VAR # make sure to uncoment output{}
# ^^^




# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
# Plot the samples of a single year.
#.............................................................
def PLOT_samps(ssps,comp,data_,years_,color):
    fig, ax = plt.subplots(5, 5, figsize=(15, 15))
    for s1,ssp in enumerate(ssps):
        for y1,year in enumerate(years_):
            IDXtime = np.where(data_[comp][ssp]['time'] == year)[0][0]
            Y_= data_[comp][ssp]['slc'][:,IDXtime]
            ax[s1,y1].plot(Y_, color=color, marker='.', linestyle='none')
            ax[s1, y1].set_ylim([-10,125])
            # STAT----------------------------------------------------
            p17_, p50_, p83_ = np.quantile(Y_, [0.17, 0.5, 0.83])
            std_dev = np.std(Y_)
            variance = np.var(Y_)
            slope, intercept, r_value, p_value, std_err = linregress(range(len(Y_)), Y_)
            #
            # put PERCENTILES on plot 
            percentiles = [(p50_, 'p50')]   #[(p17_, 'p17'), (p50_, 'p50'), (p95_, 'p95')]
            for percentile, label in percentiles:
                # Draw a horizontal line for each percentile
                ax[s1, y1].axhline(y=percentile, color='blue', linestyle='--')
                # Annotate each line
                if percentile == p50_:
                    ax[s1, y1].annotate(f'{label}: {percentile:.2f}', xy=(1, percentile), xycoords=('axes fraction', 'data'), 
                                        xytext=(-60, -10),textcoords='offset points', horizontalalignment='center', verticalalignment='center',   color='black')
            #
            # STaT text Box----------------------------
            STAT_textstr = '\n'.join((
                f'STD: {std_dev:.2f}', f'Variance: {variance:.2f}',
                
                f'Slope: {slope:.2f}',
                f'p(17,83): {p17_:.2f},{p83_:.2f}'
            ))
            ax[s1, y1].text(0.95, 0.95, STAT_textstr, transform=ax[s1, y1].transAxes, fontsize=8,
                    verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black'))
            #
            plot_details = '\n'.join((f'{year}',  f'{ssp}'))
            ax[s1, y1].text(0.05, 0.95, plot_details, transform=ax[s1, y1].transAxes, fontsize=8,
                    verticalalignment='top', horizontalalignment='left', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black'))
            #
            # Label X_, Y_, Title_
            ax[4, y1].set_xlabel('Samples')
            ax[s1, 0].set_ylabel('SLC (cm)')




# # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
# # Read NC.
# #.............................................................
# ssps = ['ssp119', 'ssp126','ssp245','ssp370', 'ssp585']
# #
# station=0; region='global'
# #
# start_year=2020; end_year=2100; unit='cm'; 
# #


# # Dictionary of filenames
# file_names = {
#     'wf1f': 'total.workflow.wf1f.{}.nc'.format(region),
#     'wf2f': 'total.workflow.wf2f.{}.nc'.format(region),
#     'wf3f': 'total.workflow.wf3f.{}.nc'.format(region),
#     'wf4' : 'total.workflow.wf4.{}.nc'.format(region)
# }

# # Base path of data folder (all ssp).
# base_path = '/Users/dota/werk/2022_09_FACTS/0000_facts-OPdata/amarel/ar2208/factsv1.1.1'

# # Dictionary to store the results
# all_ssp = {key: {} for key in file_names}

# # Loop over each SSP scenario
# for ssp in ssps:
#     for key, filename in file_names.items():
#         file_path = f'{base_path}/coupling.{ssp}/output/coupling.{ssp}.{filename}'
#         dat, slc, time, lat, lon = extract_nc_info(file_path, station, unit, start_year, end_year)
#         all_ssp[key][ssp] = {'dat': dat, 'slc': slc, 'time': time, 'lat': lat, 'lon': lon, 'path': file_path, 'filename': filename}
# del dat, slc, time, lat, lon, ssp, file_path, key, file_names, filename