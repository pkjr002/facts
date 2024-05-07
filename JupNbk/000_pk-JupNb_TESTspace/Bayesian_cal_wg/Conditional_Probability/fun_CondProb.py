import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
from scipy.stats import linregress
import glob
import os
import shutil
import re
import sys
import cartopy
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.transforms as transforms
import seaborn as sns
from sklearn.model_selection import GridSearchCV
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
# Plot the samples of a single year.
#.............................................................
def PLOT_samps(ssps,comp,data_,years_,color,yaxis_limit):
    
    # How many subplots to plot
    if len(ssps) == 5:
        fig, ax = plt.subplots(5, 5, figsize=(15, 15))
    elif len(ssps) == 1:
        fig, ax = plt.subplots(1, 5, figsize=(15, 3))
    else:
        print("Error: ONLY 1 or 5 ssps permitted at the moment")
        sys.exit(1) 

    for s1,ssp in enumerate(ssps):
        for y1,year in enumerate(years_):
            IDXtime = np.where(data_[comp][ssp]['time'] == year)[0][0]
            Y_= data_[comp][ssp]['slc'][:,IDXtime]
            # PLOT------------------------------------------------------------
            current_ax = ax[y1] if len(ssps) == 1 else ax[s1,y1]
            current_ax.plot(Y_, color=color, marker='.', linestyle='none')
            current_ax.set_ylim(yaxis_limit)   
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
                current_ax.axhline(y=percentile, color='blue', linestyle='--')
                # Annotate each line
                if percentile == p50_:
                    current_ax.annotate(f'{label}: {percentile:.2f}', xy=(1, percentile), xycoords=('axes fraction', 'data'), 
                                        xytext=(-60, -10),textcoords='offset points', horizontalalignment='center', verticalalignment='center',   color='black')
            #
            # Trend................
            istrend=check_trend(Y_)
            #
            alpha=0.5
            # STaT text Box----------------------------
            STAT_textstr = '\n'.join((
                f'STD: {std_dev:.2f}', f'Variance: {variance:.2f}',
                
                f'Slope: {slope:.2f}',
                f'p(17,83): {p17_:.2f},{p83_:.2f}',
                f'Trend: {istrend}'
            ))
            current_ax.text(0.95, 0.95, STAT_textstr, transform=current_ax.transAxes, fontsize=8,
                    verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=alpha),alpha=alpha)
            #    
            plot_details = '\n'.join((f'{year}',  f'{ssp}'))
            current_ax.text(0.05, 0.95, plot_details, transform=current_ax.transAxes, fontsize=8,
                    verticalalignment='top', horizontalalignment='left', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=alpha),alpha=alpha)
            #
            data_dets = '\n'.join(( f'{data_[comp][ssp]["path"].split("/")[-1].split(".")[-3]}' , f'{data_[comp][ssp]["path"].split("/")[-1].split(".")[-2]}' ))
            current_ax.text(0.05, 0.6, data_dets, transform=current_ax.transAxes, fontsize=8,
                    verticalalignment='top', horizontalalignment='left', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=alpha),alpha=alpha)
            #
            # Label X_, Y_, Title_
            if len(ssps) == 5: ax[4, y1].set_xlabel('Samples')
            else: ax[y1].set_xlabel('Samples')
            
            
            if len(ssps) == 5: ax[s1, 0].set_ylabel('SLC (cm)')
            else: ax[s1].set_ylabel('SLC (cm)')



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
# Find Trend Based on auto-correlation.
#.............................................................

from statsmodels.stats.diagnostic import acorr_ljungbox

def check_trend(data):

    # Ljung-Box test to get auto-correlation coefficient.
    lb_test = acorr_ljungbox(data, lags=len(data)-1, return_df=True)  # OR lags=[1,2,3,4,5]

    # update trend based on p-value.
    lb_test['trend'] = lb_test['lb_pvalue'].apply(lambda p: 'Yes' if p < 0.05 else 'No')
    # print(lb_test)

    # Conditionals for trend ..........................................................
    # STRICT => Check if all values in 'trend' are 'Yes'
    # trend_in_data = 'Yes' if (lb_test['trend'] == 'Yes').all() else 'No'

    # PCENT => Check based on % of yes/no
    frequency = lb_test['trend'].value_counts(normalize=True)
    if frequency.get('Yes', 0) >= 0.80:
        trend_in_data = 'Yes'
    elif frequency.get('No', 0) >= 0.80:
        trend_in_data = 'No'
    else:
        trend_in_data = 'CHK'  # Use this or any other label you prefer
    # Output
    # print("trend_in_data:", trend_in_data)
    return trend_in_data
# ..........................................


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
# Dataframe to check for trend
#.............................................................
def df_samps_trend(ssps,data_,years_):
    #
    if len(ssps) >1: 
        print("Error: ONLY 1 ssp permitted at the moment")
        sys.exit(1) 
    #
    first_level_keys = list(data_.keys())
    data_dict = {first_level_key: {} for first_level_key in first_level_keys} 
    data_years_istrend = []
    for c1,comp in enumerate(first_level_keys):
        years_istrend=[]
        for s1,ssp in enumerate(ssps):
            year_istrend=[]
            for y1,year in enumerate(years_):
                IDXtime = np.where(data_[comp][ssp]['time'] == year)[0][0]
                Y_= data_[comp][ssp]['slc'][:,IDXtime]
                #
                # Trend................
                istrend=check_trend(Y_)
                #
                # populate the list
                data_dict[comp][year] = istrend
    return data_dict 



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
# Fun: Conditional Probability Plotting.
#.............................................................
def plot_ConditionalProb_panel(all_ssp_data,plot_params,plotOPT):
    #
    panel_no=len(plot_params)
    if panel_no == 4:     fig, ax = plt.subplots(1, 4, figsize=(20, 4)); 
    elif panel_no == 5:   fig, ax = plt.subplots(1, 5, figsize=(25, 4)); 
    else :                raise ValueError(f"Unexpected number of panels: {panel_no}. Expected 4 or 5.")
    #
    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    #
    # Loop through the dictionary and plot
    for i, params in plot_params.items():
        # AXIS-LABELS
        var1_lab = next(key for key, val in all_ssp_data.items() if np.array_equal(val, params['var1']))
        var2_lab = next(key for key, val in all_ssp_data.items() if np.array_equal(val, params['var2']))
        plotOPT['YaxLab_disp'] = 'YES' if i == 0 else 'NO' 
        #
        if plotOPT['plotCBAR'] == 'YES':
            plot_ConditionalProb(ax[i], params['var1'], params['var2'], params['t1'], params['t2'],var1_lab,var2_lab,plotOPT)
        if plotOPT['plotCBAR'] == 'YES_1':
            showCBAR = 1 if i == 4 or 5 else 0
            plotOPT['showCBAR'] = showCBAR
            plotOPT['cbar_ax'] = fig.add_axes([0.92, 0.15, 0.01, 0.7])  # controls the shape [left, bottom, width, height]
            plot_ConditionalProb(ax[i], params['var1'], params['var2'], params['t1'], params['t2'],var1_lab,var2_lab,plotOPT)      
    plt.show()        
#.............................................................
def plot_ConditionalProb(ax, var1, var2, t1, t2,var1_lab,var2_lab,plotOPT):
    #-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    # Unpack Plot Options
    ssp = plotOPT['ssps']
    kernel = plotOPT['kernel']; 
    bw_kde = plotOPT['bw_kde']; 
    kde_grid_int = plotOPT['kde_grid_int']
    val=plotOPT['val']
    plt_og = plotOPT['plt_og']; 
    plt_scatter = plotOPT['plt_scatter']; 
    cmap = plotOPT['cmap'] 
    #-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    # Define AXIS variables
    var1=var1[ssp]
    var2=var2[ssp]
    # Extract slc for a specified time slice.
    xaxVAR = var1['slc'][:, np.where(var1['time']==t1)[0][0]] 
    yaxVAR = var2['slc'][:, np.where(var1['time']==t2)[0][0]] 
    # LABELS
    xaxLAB = f'{var1_lab}_{t1} (cm)'     
    yaxLAB = f'{var2_lab}_{t2} (cm)'  
    title  = f'{t2} {var2_lab}  \n conditional upon \n {t1} {var1_lab} '
    #-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    # PLOT Conditional Probability figure.
    gilford(ax, xaxVAR, yaxVAR, kernel, bw_kde, kde_grid_int, val, xaxLAB, yaxLAB, title, ssp, plt_og,plt_scatter, cmap, t1, plotOPT)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
# Plot contour that sums to 1 along columns.
#.............................................................
def gilford(ax, xaxVAR, yaxVAR,kernel,bw_kde,kde_grid_int, val, xaxLAB,yaxLAB,title,ssp,plt_og,plt_scatter, CMAP,T1, plotOPT=None):
    #-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    # Extract Quantiles
    quantiles = [0.01, 0.05, 0.17, 0.5, 0.83, 0.95, 0.99]
    Xp01_, Xp05_, Xp17_, Xp50_, Xp83_, Xp95_, Xp99_ = np.quantile(xaxVAR, quantiles)
    Yp01_, Yp05_, Yp17_, Yp50_, Yp83_, Yp95_, Yp99_ = np.quantile(yaxVAR, quantiles)
    #-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    # Create 2D matrix.
    INdata = np.column_stack((xaxVAR, yaxVAR))
    #
    if bw_kde == 'auto':
        data = INdata
        #print("Data shape:", data.shape)
        # Grid of bandwidth values to test
        bandwidths = np.linspace(0.1, 10, 40)  
        # Setup the grid search with cross-validation
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                           {'bandwidth': bandwidths},
                           cv=5)  # 5-fold cross-validation
        # Fit grid search
        grid.fit(data)
        # Best bandwidth
        best_bw = np.round(grid.best_params_['bandwidth'],2)
        print("Optimal bandwidth:", best_bw)
        bw_kde=best_bw
    else:
        bw_kde=bw_kde
    #.............................................................................................................
    # KDE
    kde = KernelDensity(kernel=kernel, bandwidth=bw_kde).fit(INdata)
    #-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    # Create a grid of test points
    if plotOPT['grid'] == 'data_MaxMin':
        # simple.
        xgrid = np.linspace(INdata[:,0].min()-1, INdata[:,0].max()+1, kde_grid_int)  
        ygrid = np.linspace(INdata[:,1].min()-1, INdata[:,1].max()+1, kde_grid_int)   
    #--> below not recomended as it can dilute the plot.
    elif plotOPT['grid'] == 'Fixed':
        xgrid = np.linspace(-20, 100, kde_grid_int)  
        ygrid = np.linspace(-20, 100, kde_grid_int)  
    #
    # %age
    # bandwidth_extension_factor = bw_kde * 4  #2 is eg
    # # Create a grid of test points, extended by a function of the bandwidth
    # xgrid = np.linspace(INdata[:,0].min() - bandwidth_extension_factor, 
    #                 INdata[:,0].max() + bandwidth_extension_factor, kde_grid_int)
    # ygrid = np.linspace(INdata[:,1].min() - bandwidth_extension_factor, 
    #                 INdata[:,1].max() + bandwidth_extension_factor, kde_grid_int)
    #-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    Xgrid, Ygrid  = np.meshgrid(xgrid, ygrid)
    grid_samples = np.vstack([Xgrid.ravel(), Ygrid.ravel()]).T
    #
    # Eval density model on the grid (log likelihoods)
    log_density_values = kde.score_samples(grid_samples)
    #Reshape 
    log_density_values = log_density_values.reshape(Xgrid.shape)
    #
    #-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    # Density values
    density_values = np.exp(log_density_values)
    #.............................................................................................................
    # Normalize 
    normalized_density_values = density_values / np.sum(density_values, axis=0);
    #.............................................................................................................
    ## Normalize log_density_values |.| Do all of this in log scales (Log-Sum-Exp trick)
    #adjusted_log_density_values = log_density_values - np.max(log_density_values)
    #exp_sum = np.sum(np.exp(adjusted_log_density_values), axis=0)
    ## Readjust by adding back the adjustment factor
    #log_normalization_constant = np.log(exp_sum) + np.max(log_density_values);
    ## Normalize within the log space 
    #normalized_log_density_values = log_density_values - log_normalization_constant

    #-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    if val == 'log_density_values':
        PLOT_VAR=log_density_values
    #elif val == 'log_density_values_Normalized':
    #    PLOT_VAR=normalized_log_density_values
    elif val == 'density_values':
        PLOT_VAR=density_values
    elif val == 'density_values_Normalized':
        PLOT_VAR=normalized_density_values
    #
    #-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    # PLOT:: Choose PLOT_VAR and fill contour and COLORbar limits
    if val in ['density_values' , 'density_values_Normalized', 'log_density_values', 'log_density_values_Normalized']:
        if val in ['density_values' , 'density_values_Normalized']:
            if plotOPT is not None and 'c_bar_min' in plotOPT:
                clevels=np.linspace(plotOPT['c_bar_min'],plotOPT['c_bar_max'],plotOPT['c_bar_int'])    
            else: clevels=np.linspace(1e-3,PLOT_VAR.max(),10)
        else: clevels=np.linspace(PLOT_VAR.min(),PLOT_VAR.max(),10)
        # PLOT:: contour
        clabels=np.round(clevels,decimals=3).astype('str')
        contour=ax.contourf(Xgrid, Ygrid, PLOT_VAR,levels=clevels,cmap=CMAP)
        # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&    
        # print(PLOT_VAR)
        # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # # PRINT<==
        # # Calculate the intervals consecutive x/y values
        #dx = np.diff(xgrid)  ; print("X-axis interval:", dx[0])
        #dy = np.diff(ygrid)  ; print("Y-axis interval:", dy[0])
        #print("----")
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # COLORBAR::
        if plotOPT['plotCBAR'] is not None:
            if plotOPT['plotCBAR'] == 'YES':
                cbar=plt.colorbar(contour,ax=ax,label=val,ticks=clevels,orientation='horizontal',pad=0.2)
                cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=45)  
            if plotOPT['plotCBAR'] == 'YES_1' and plotOPT['showCBAR'] == 1:
                cbar=plt.colorbar(contour,cax=plotOPT['cbar_ax'],label=val,ticks=clevels,orientation='vertical',pad=0.01)    
            cbar.set_label(label=val, size=10, weight='bold', color='blue')
            cbar.set_ticklabels(clabels)
            cbar.ax.tick_params(labelsize=8)
            # cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=45)    
        ax.set_xlim(np.floor(np.min(INdata[:, 0])), np.ceil(np.max(INdata[:, 0])))
        ax.set_ylim(np.floor(np.min(INdata[:, 1])), np.ceil(np.max(INdata[:, 1])))    
    #-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    # PLOT:: SNS Kde  
    if plotOPT['sns_kde'] == 'YES':
        sns.kdeplot(x=INdata[:, 0], y=INdata[:, 1], fill=True, thresh=0, levels=100, cmap="mako", ax=ax)

    #-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    # PLOT:: RAW original data  
    if plt_og == 'YES':
        numbers = np.linspace(1, 2000, 2000)    
        x_min, x_max = np.floor(np.min(numbers)), np.ceil(np.max(numbers)) 
        #
        color = 'tab:blue'
        ax.set_xlabel('')
        ax.set_ylabel('', color=color)
        ax.plot(numbers,INdata[:, 0], 'o', linestyle='none', color=color, label=xaxLAB)
        ax.tick_params(axis='y', labelcolor=color)
        #
        ax2 = ax.twinx()  
        color = 'tab:orange'
        ax2.set_ylabel('', color=color)
        ax2.plot(numbers,INdata[:, 1], 'x', linestyle='none', color=color, label=yaxLAB)
        ax2.tick_params(axis='y', labelcolor=color)
        #
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, loc='upper left') 
        ax.legend(handles1 + handles2, labels1 + labels2, loc='upper left', bbox_to_anchor=(.275, 1.35), title='')

        if plotOPT['fix_ax_lim'] == 'NO':
            ax.set_xlim(x_min,x_max)
            ax.set_ylim(np.floor(np.min(INdata[:, 0])), np.ceil(np.max(INdata[:, 0])))
            ax2.set_ylim(np.floor(np.min(INdata[:, 1])), np.ceil(np.max(INdata[:, 1])))
        elif plotOPT['fix_ax_lim'] == 'YES': 
            # ax.set_ylim(np.floor(np.min(INdata[:, 1])), np.ceil(np.max(INdata[:, 1])))
            # ax2.set_ylim(np.floor(np.min(INdata[:, 1])), np.ceil(np.max(INdata[:, 1])))
            ax.set_ylim(-10, np.ceil(np.max(INdata[:, 1])))
            ax2.set_ylim(-10, np.ceil(np.max(INdata[:, 1])))
        xaxLAB = 'samples'; ax.set_xlabel(xaxLAB)
        yaxLAB = 'projected sl (cm) ';
        if plotOPT['YaxLab_disp'] == 'YES':     ax.set_ylabel(yaxLAB)
        title = 'raw data'
    #-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    # PLOT:: SCATTER of the original data
    if plt_scatter == 'YES':
        ax.scatter(INdata[:, 0], INdata[:, 1], s=.5, facecolor='red')
        ax.set_xlim(np.floor(np.min(INdata[:, 0])), np.ceil(np.max(INdata[:, 0])))
        ax.set_ylim(np.floor(np.min(INdata[:, 1])), np.ceil(np.max(INdata[:, 1])))
        #if plotOPT['mark_ptile'] == 'YES':

    #-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    # PLOT:: voilin/boxWhisk
    if plotOPT is not None and plotOPT['plt_overlay'] in ['violin', 'box']:
        num_violins = plotOPT['num_violins']
        x_min, x_max = np.floor(np.min(INdata[:, 0])), np.ceil(np.max(INdata[:, 0]))
        bin_edges = np.linspace(x_min, x_max, num=num_violins+1, endpoint=True)
        x_binned = np.digitize(INdata[:, 0], bin_edges, right=True)

        # Group data by bins
        binned_data = [INdata[:, 1][x_binned == i] for i in range(1, len(bin_edges))]

        positions = np.arange(1, num_violins + 1) * (x_max - x_min) / num_violins - ((x_max - x_min) / (2 * num_violins)) + x_min

        if plotOPT['plt_overlay'] == 'violin':
            ax.violinplot(binned_data, positions=positions, widths=(x_max - x_min) / num_violins * 0.8, showmeans=False, showextrema=True, showmedians=True)
        elif plotOPT['plt_overlay'] == 'box':
            for i, data in enumerate(binned_data):
                ax.boxplot(data, positions=[positions[i]], widths=(x_max - x_min) / num_violins * 0.8, vert=True, patch_artist=True)
        #
        ax.set_xlim(x_min,x_max)
        if plotOPT is not None and 'y_ax_min' in plotOPT:
            ax.set_ylim(plotOPT['y_ax_min'],plotOPT['y_ax_max'])
    #-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    # AXIS properties
    ax.set_title(title,fontsize=8)
    if plotOPT['YaxLab_disp'] == 'YES':     ax.set_ylabel(yaxLAB)
    ax.set_xlabel(xaxLAB)
    #
    ax.text(0.9, 0.1, f'{ssp}', fontsize=7, color='black', weight='bold', ha='center', va='center', transform=ax.transAxes)
    #
    # ax.set_xlim(Xp01_,Xp99_)
    #
    # ................................................................................................
    if plotOPT['mark_ax_ptile'] == 'YES':
        qq=['5','17','50','83','95']
        relativeY = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        # ax.text(Xp50_, 0, '|', fontsize=7, ha='center', va='top', transform=relativeY)
        for i,Xp in enumerate([Xp05_,Xp17_, Xp50_,Xp83_,Xp95_]):
            ax.text(Xp, 0, '|', fontsize=15, color='blue', ha='center', va='top', transform=relativeY)
            if i not in [1,3]:
                ax.text(Xp, 0.05, f'p{qq[i]}' , fontsize=10, color='blue', ha='center', va='top', transform=relativeY)
        #
        relativeX = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        # ax.text(0, Yp50_,'-', fontsize=14, ha='right', va='center', transform=relativeX)  #'\u2014'
        for i,Yp in enumerate([Yp05_,Yp17_, Yp50_,Yp83_,Yp95_]):
            ax.text(0.05, Yp, '--', fontsize=14, color='blue', ha='right', va='center', transform=relativeX)
            if i not in [1,3]:  
                ax.text(0.05, Yp, f'p{qq[i]}', fontsize=10, color='blue', ha='right', va='center', transform=relativeX)
    # ................................................................................................
    # Adjust tick settings to ensure correct display
    ax.tick_params(axis='y', which='both', labelleft=True, labelright=False)
    ax.tick_params(axis='x', which='both', top=False, bottom=True, labeltop=False)


    if plotOPT['ptile_table'] == 'YES':
        percentiles = ['5','17','50','83','95']

        # Store results in a pandas DataFrame and then transpose it
        table_data = pd.DataFrame({
            f'{xaxLAB}':np.round([Xp05_,Xp17_, Xp50_,Xp83_,Xp95_],1),
            f'{yaxLAB}': np.round([Yp05_,Yp17_, Yp50_,Yp83_,Yp95_],1)
        }, index=[f'p{p}' for p in percentiles])



        # Create table at bottom of each subplot
        table = ax.table(cellText=table_data.values, 
                         rowLabels=table_data.index, 
                         colLabels=table_data.columns,
                         cellLoc='center', 
                         loc='bottom',
                         bbox=[0, -0.5, 1, 0.3])  # Adjust bbox for table positioning within plot
        ax.set_adjustable('datalim')
        ax.autoscale()


# ^^cx^
