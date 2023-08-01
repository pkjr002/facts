import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def project_SL(station,name,ssp,path,options):
    ylab        = options['ylab']
    x_min       = options['x_min']
    x_max       = options['x_max']
    y_min       = options['y_min']
    y_max       = options['y_max']
    x_ticks     = options['x_ticks']
    y_ticks     = options['y_ticks']
    yrST        = options.get('yrST', None)
    yrEN        = options.get('yrEN', None)
    latIN       = options.get('latIN',None)
    lonIN       = options.get('lonIN',None)
    #    
    # ggg         = [f'{path[0]}/medium_confidence/{ssp_value}/total_{ssp_value}_medium_confidence_values.nc' for ssp_value in ssp]
    ggg         = [f'{path[0]}/4_confidence_level_files/medium_confidence/{ssp_value}/total_{ssp_value}_medium_confidence_values.nc' for ssp_value in ssp]
    #
    pk          = [f'{path[1]}/4_confidence_level_files/medium_confidence/{ssp_value}/total_{ssp_value}_medium_confidence_values.nc' for ssp_value in ssp]
    pk_update   = [f'{path[2]}/4_confidence_level_files/medium_confidence/{ssp_value}/total_{ssp_value}_medium_confidence_values.nc'for ssp_value in ssp]

    #
    fig, axes = plt.subplots(1, 3, figsize=(40, 10)); plt.subplots_adjust(wspace=0.4, hspace=0.2)
    # fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    #
    plot_subplot(axes[0],ggg,station,name,ssp,ylab,x_min, x_max, y_min, y_max, x_ticks, y_ticks,yrST,yrEN,latIN,lonIN)
    plot_subplot(axes[1],pk,station,name,ssp,ylab,x_min, x_max, y_min, y_max, x_ticks, y_ticks,yrST,yrEN,latIN,lonIN)
    plot_subplot(axes[2],pk_update,station,name,ssp,ylab,x_min, x_max, y_min, y_max, x_ticks, y_ticks,yrST,yrEN,latIN,lonIN)


# ...........................................................................................................
colors = {
    'ssp119': np.array([0, 173, 207]) / 255,
    'ssp126': np.array([23, 60, 102]) / 255,
    'ssp245': np.array([247, 148, 32]) / 255,
    'ssp370': np.array([231, 29, 37]) / 255,
    'ssp585': np.array([149, 27, 30]) / 255
}
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def plot_subplot(ax,file_paths,station,name,ssp,ylab,x_min, x_max, y_min, y_max, x_ticks, y_ticks,yrST,yrEN,latIN,lonIN):
    lines = []
    labels = []
    for i, (f_p, ssp_value) in enumerate(zip(file_paths, ssp)):
        d0          = xr.open_dataset(f_p, engine='netcdf4')
        percentile  = d0['quantiles'].values * 100
        idx         = np.where(percentile == 50)[0]
        idx1        = np.where(percentile == 17)[0]
        idx2        = np.where(percentile == 83)[0] 
        slc         = d0['sea_level_change'].values / 1000
        # ........................................................
        # Index time.
        time        = d0['years'].values
        # if ('yrST' in locals() or 'yrST' in globals()) and ('yrEN' in locals() or 'yrEN' in globals()):
        if (yrST is not None) and (yrEN is not None):
            idx_yr=np.where((time >= yrST) & (time <= yrEN))[0]
        else: idx_yr=np.where((time >= 2020) & (time <= 2150))[0]
        # ........................................................
        # get the idx based on user lat lon.
        if latIN is not None and latIN is not None: 
            idx_latIN = np.where(d0['lat'].values == latIN)
            if len(idx_latIN) > 1: raise ValueError('Multiple Lat match')
            idx_lonIN = np.where(d0['lon'].values == lonIN)
            if len(idx_lonIN) > 1: raise ValueError('Multiple Lon match')
            if idx_latIN != idx_lonIN: 
                raise ValueError('Indices for latitude and longitude do not match ...')
            #
            station=idx_latIN
        else:
            station=station
        #            
        lat = np.around(d0['lat'][station].values, decimals=2)
        lon = np.around(d0['lon'][station].values, decimals=2)
        #
        line,       = ax.plot(time[idx_yr], slc[idx, idx_yr, station].reshape(-1), color=colors[ssp_value])
        lines.append(line); labels.append(f'{ssp_value} M')
        ax.fill_between(time[idx_yr], slc[idx1, idx_yr, station].reshape(-1), slc[idx2, idx_yr, station].reshape(-1),
                         color=colors[ssp_value], alpha=0.2)
        # ........................................................................................................................................................................
        # Mark the Right values.
        if ssp_value == 'ssp585':
            if (yrST is not None) and (yrEN is not None):
                ax.text(1.01, slc[idx2, np.where(time == yrEN)[0], station][0] + 0.03, f'{slc[idx2, np.where(time == yrEN)[0], station][0]:.2f} m',transform=ax.get_yaxis_transform(), fontweight='bold', fontsize=23, ha='left', va='center', color=colors[ssp[-1]])
            else: ax.text(1.01, slc[idx2, -1, station][0] + 0.03, f'{slc[idx2, -1, station][0]:.2f} m',transform=ax.get_yaxis_transform(), fontweight='bold', fontsize=23, ha='left', va='center', color=colors[ssp[-1]])
        if ssp_value == 'ssp126':
            if (yrST is not None) and (yrEN is not None):
                ax.text(1.01, slc[idx1, np.where(time == yrEN)[0], station][0] - 0.03, f'{slc[idx1, np.where(time == yrEN)[0], station][0]:.2f} m',transform=ax.get_yaxis_transform(), fontweight='bold', fontsize=23, ha='left', va='center', color=colors[ssp[0]])
            else: ax.text(1.01, slc[idx1, -1, station][0] - 0.03, f'{slc[idx1, -1, station][0]:.2f} m',transform=ax.get_yaxis_transform(), fontweight='bold', fontsize=23, ha='left', va='center', color=colors[ssp[0]])
        # ........................................................................................................................................................................
        # Set x-axis label and limits
        ax.set_xlabel('Year', fontsize=25)
        ax.set_ylabel(ylab, fontsize=25)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(x_ticks);  ax.set_xticklabels(x_ticks,fontsize=20, rotation=45)
        ax.set_yticks(y_ticks);  ax.set_yticklabels(y_ticks,fontsize=20) 
        #ax.grid(True)
        ax.legend(lines + [ax.fill_between([], [], [], color='gray', alpha=0.2)],
                      labels + ['Shading is 17-83 percentile'], loc='upper left', fontsize=20)
        # if np.all(ax == ax[0].values):
        #     ax.legend(lines + [ax.fill_between([], [], [], color='gray', alpha=0.2)],
        #               labels + ['Shading is 17-83 percentile'], loc='upper left', fontsize=7)
        txt = f'{name}\n(Site {station})'
        ax.text(0.03, 0.62, txt,color='blue',transform=ax.transAxes, verticalalignment='top', fontsize='20', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        #
        location = [
            f"lat = {str(lat)}",
            f"lon = {str(lon)}"
        ]
        text = "\n".join(location)
        ax.text(0.015, 0.4, text, fontsize=20, fontweight='normal', ha='left', va='center', transform=ax.transAxes)
        #
    #
    set_subplot_titles(ax, file_paths)
    # plt.show()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def set_subplot_titles(ax, file_paths):
    path = file_paths[0]
    labels = path.split('/')
    facts_label = next((label for label in labels if 'FACTS' in label), None)
    if facts_label is not None:
        ax.set_title(facts_label,fontsize=25)