import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def project_SL(folder,station,name,ssp,options):
    ylab = options['ylab']
    x_min = options['x_min']
    x_max = options['x_max']
    y_min = options['y_min']
    y_max = options['y_max']
    x_ticks = options['x_ticks']
    y_ticks = options['y_ticks']
    yrST = options.get('yrST', None)
    yrEN = options.get('yrEN', None)
    #
    #
    if len(folder) ==1:
        folder=folder[0]
        fig, axes = plt.subplots(1, 1, figsize=(10, 5)); plt.subplots_adjust(wspace=0.4, hspace=0.2)
        #
        folderloop=[f'{folder}/{ssp_value}/total_{ssp_value}_medium_confidence_values.nc' for ssp_value in ssp]
        FS=10
        plot_subplot(axes,folderloop,station,name,ssp,ylab,x_min, x_max, y_min, y_max, x_ticks, y_ticks,FS,yrST,yrEN)
    else:
    #
        fig, axes = plt.subplots(1, 3, figsize=(40, 10)); plt.subplots_adjust(wspace=0.4, hspace=0.2)
        #
        FS=20
        #
        folderloop=[f'{folder[0]}/{ssp_value}/total_{ssp_value}_medium_confidence_values.nc' for ssp_value in ssp]
        plot_subplot(axes[0],folderloop,station,name,ssp,ylab,x_min, x_max, y_min, y_max, x_ticks, y_ticks,FS,yrST,yrEN)
        #
        folderloop=[f'{folder[1]}/{ssp_value}/total_{ssp_value}_medium_confidence_values.nc' for ssp_value in ssp]
        plot_subplot(axes[1],folderloop,station,name,ssp,ylab,x_min, x_max, y_min, y_max, x_ticks, y_ticks,FS,yrST,yrEN)
        # plot_subplot(axes[2],pk_update,station,name,ssp,ylab,x_min, x_max, y_min, y_max, x_ticks, y_ticks,yrST,yrEN)
        #
        axes[2].axis('off')


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
# Font
def plot_subplot(ax,file_paths,station,name,ssp,ylab,x_min, x_max, y_min, y_max, x_ticks, y_ticks,FS,yrST,yrEN):
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
        lat = np.around(d0['lat'][station].values, decimals=2)
        lon = np.around(d0['lon'][station].values, decimals=2)
        #
        if ssp_value == 'ssp126':
            line,       = ax.plot(time[idx_yr], slc[idx1, idx_yr, station].reshape(-1), color=colors[ssp_value], linestyle='--')
            lines.append(line); labels.append(f'{ssp_value} p17')
        line,       = ax.plot(time[idx_yr], slc[idx, idx_yr, station].reshape(-1), color=colors[ssp_value])
        lines.append(line); labels.append(f'{ssp_value} M')
        if ssp_value == 'ssp585':
            line,       = ax.plot(time[idx_yr], slc[idx2, idx_yr, station].reshape(-1), color=colors[ssp_value], linestyle='--')
            lines.append(line); labels.append(f'{ssp_value} p83')
        # ........................................................................................................................................................................
        ## Mark the Right values.
        if ssp_value == 'ssp585':
            if (yrST is not None) and (yrEN is not None):
                ax.text(1.01, slc[idx2, np.where(time == yrEN)[0], station][0] + 0.03, f'{slc[idx2, np.where(time == yrEN)[0], station][0]:.2f} m',transform=ax.get_yaxis_transform(), fontweight='bold', fontsize=FS, ha='left', va='center', color=colors[ssp[-1]])
            else: ax.text(1.01, slc[idx2, -1, station][0] + 0.03, f'{slc[idx2, -1, station][0]:.2f} m',transform=ax.get_yaxis_transform(), fontweight='bold', fontsize=FS, ha='left', va='center', color=colors[ssp[-1]])
        if ssp_value == 'ssp126':
            if (yrST is not None) and (yrEN is not None):
                ax.text(1.01, slc[idx1, np.where(time == yrEN)[0], station][0] - 0.03, f'{slc[idx1, np.where(time == yrEN)[0], station][0]:.2f} m',transform=ax.get_yaxis_transform(), fontweight='bold', fontsize=FS, ha='left', va='center', color=colors[ssp[0]])
            else: ax.text(1.01, slc[idx1, -1, station][0] - 0.03, f'{slc[idx1, -1, station][0]:.2f} m',transform=ax.get_yaxis_transform(), fontweight='bold', fontsize=FS, ha='left', va='center', color=colors[ssp[0]])
        #
        # Mark all at end points
        # print(slc[idx, -1, station][0])
        if (yrST is not None) and (yrEN is not None):
            ax.text(1.01, slc[idx, np.where(time == yrEN)[0], station][0] + 0.03, f'{slc[idx, np.where(time == yrEN)[0], station][0]:.2f} m',transform=ax.get_yaxis_transform(), fontweight='bold', fontsize=FS, ha='left', va='center', color=colors[ssp_value])
        else: ax.text(1.01, slc[idx, -1, station][0] + 0.03, f'{slc[idx, -1, station][0]:.2f} m',transform=ax.get_yaxis_transform(), fontweight='bold', fontsize=FS, ha='left', va='center', color=colors[ssp_value])
        # ........................................................................................................................................................................
        # Set x-axis label and limits
        ax.set_xlabel('Year', fontsize=FS+(0.25*FS))
        ax.set_ylabel(ylab, fontsize=FS+(0.25*FS))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(x_ticks);  ax.set_xticklabels(x_ticks,fontsize=FS, rotation=45)
        ax.set_yticks(y_ticks);  ax.set_yticklabels(y_ticks,fontsize=FS) 
        ax.legend(lines,labels , loc='upper left', fontsize=FS)
        ax.grid(True)
        ax.grid(color='gray', linestyle='-', linewidth=0.5,alpha=0.5)
        #
        txt = f'{name}\n(Site {station})'
        ax.text(0.03, 0.62, txt,color='blue',transform=ax.transAxes, verticalalignment='top', fontsize=FS, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        #
        location = [
            f"lat = {str(lat)}",
            f"lon = {str(lon)}"
        ]
        text = "\n".join(location)
        ax.text(0.015, 0.4, text, fontsize=FS, fontweight='normal', ha='left', va='center', transform=ax.transAxes)
        #
    #
    ax.set_title('GMSL projections (medium confidence)',fontsize=FS+(0.5*FS))
    # plt.show()