import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap
import glob
import os
import shutil
import re
import cartopy
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import fnmatch
#
PD=os.getcwd(); #PD
# ==========================================================================================
#

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
colors = {
    'ssp119' : np.array([0, 173, 207]) / 255,
    'ssp126' : np.array([23, 60, 102]) / 255,
    'ssp245' : np.array([247, 148, 32]) / 255,
    'ssp370' : np.array([231, 29, 37]) / 255,
    'ssp585' : np.array([149, 27, 30]) / 255
}
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def plot(loc,ssp):
    #
    fig, axes = plt.subplots(3, 2, figsize=(10*2, 3*3)); 
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    #
    for ss0,ss1 in enumerate(ssp):
        #
        files=filePATH(ss1)   
        sls, Qvals, label, quant, qlevs, subq = nc2var(files,loc)
        #
        fnt=8
        lws = [2, 2, 1, 1] * 2
        color_list = list(colors.values())[:5]; 
        hp = []  
        binwidth = 50
        #
        for sss in range(4):
            # 
            current_color = color_list[sss]
            if sss < 2: linewidth = 3.5  
            else: linewidth = 1.5
            # ====================================================================================================================================
            # ====================================================================================================================================
            # PLOT  Pnl 1
            ax= axes[0,ss0]
            #
            # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
            # n, edges = np.histogram(sls[sss, :], np.arange(0, 4001, binwidth), density=True)
            bin_edges = np.arange(0, 4000 + binwidth, binwidth)
            n, edges = np.histogram(sls[sss, :], bins=bin_edges)
            #
            hist_sum = np.sum(n)
            scaling_factor = binwidth/1000
            n = n / hist_sum / scaling_factor
            #
            xx=.5*(edges[1:]+edges[:-1])/1000
            ax.plot(xx, n, color=current_color,label=label[sss, 0])

            # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
            #
            ax.set_xlabel('GMSL rise in 2100 (m)', fontsize=fnt+2)
            ax.set_ylabel('Normalized frequency', fontsize=fnt+2)
            ax.set_xlim(0, 3.5)
            # ax.set_ylim(-0.25, 1.025)
            ax.set_xticks(ax.get_xticks())
            x_ticks= np.around(np.arange(-0.5, 3.6, .5), decimals=1); ax.set_xticks(x_ticks);  ax.set_xticklabels(x_ticks,fontsize=fnt, rotation=0)
            # ax.set_yticks(ax.get_yticks())
            # ax.set_yticklabels(ax.get_yticks(), fontsize=fnt, rotation=0) 
            ax.tick_params(direction='in', length=7, width=2.5, axis='both')
            title=ss1[:4]+'-'+ss1[4]+'.'+ss1[5];  ax.set_title(title,fontsize=fnt+10)
            if ax==axes[0,0]: ax.legend(fontsize=fnt+2.5) 
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(2.5)
            # ====================================================================================================================================
            # ====================================================================================================================================
            # PLOT  Pnl 2
            ax= axes[1,ss0]
            #
            # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
            xx=sls[sss,:]/1000
            xx1=sls[sss,subq]/1000
            yy=quant
            yy1 = [-0.05 * sss-0.05] * 2
            ax.plot(xx, yy, color=current_color, linewidth=linewidth,label=label[sss, 0])
            ax.plot(xx1,yy1,color=current_color, linewidth=linewidth)
            # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
            # 
            # Set x-axis label and limits
            ax.set_xlabel('GMSL rise in 2100 (m)', fontsize=fnt+2)
            ax.set_ylabel('CDF quantile', fontsize=fnt+2)
            ax.set_xlim(-0.6, 3.5)
            ax.set_ylim(-0.25, 1.025)
            ax.set_xticks(ax.get_xticks())
            x_ticks= np.around(np.arange(-0.5, 3.6, .5), decimals=1); ax.set_xticks(x_ticks);  ax.set_xticklabels(x_ticks,fontsize=fnt, rotation=0)
            y_ticks= np.around(np.arange(0,1.1,0.2), decimals=1) ; ax.set_yticks(y_ticks);  ax.set_yticklabels(y_ticks,fontsize=fnt) 
            ax.tick_params(direction='in', length=7, width=2.5, axis='both')
            ax.axhline(0, color='black', linestyle='-', linewidth=1.5)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(2.5)
            # ====================================================================================================================================
            # ====================================================================================================================================
            # PLOT  Pnl 3
            ax= axes[2,ss0]
            #
            #
            # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
            # P-box 
            pbox1l = np.min(Qvals[:2, :], axis=0)
            pbox1h = np.max(Qvals[:2, :], axis=0)
            pbox2l = np.min(Qvals, axis=0)
            pbox2h = np.max(Qvals, axis=0)
            #
            ax.fill_betweenx(np.concatenate([qlevs,qlevs[::-1]]), np.concatenate([pbox2l, pbox2h[::-1]]) / 1000,  color=[0.8, 0.8, 1])
            ax.fill_betweenx(np.concatenate([qlevs,qlevs[::-1]]), np.concatenate([pbox1l, pbox1h[::-1]]) / 1000,  color=[0.4, 0.4, 1])
            #
            ax.plot([pbox2l[subq[0]] / 1000, pbox2h[subq[1]] / 1000], [-0.1, -0.1], color=[0.8, 0.8, 1], linewidth=6)
            ax.plot([pbox1l[subq[0]] / 1000, pbox1h[subq[1]] / 1000], [-0.1, -0.1], color=[0.4, 0.4, 1], linewidth=6)
            # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
            #
            ax.plot([0, 4], [0, 0], color='black', linestyle='-')
            ax.set_xlim([0, 3.5]); 
            ax.set_ylim([-0.25, 1])
            ax.set_xlabel('GMSL rise in 2100 (m)', fontsize=fnt+2)
            ax.set_ylabel('p-boxquantile', fontsize=fnt+2)
            ax.set_xticks(ax.get_xticks())
            x_ticks= np.around(np.arange(-0.5, 3.6, .5), decimals=1); ax.set_xticks(x_ticks);  ax.set_xticklabels(x_ticks,fontsize=fnt, rotation=0)
            y_ticks= np.around(np.arange(0,1.1,0.2), decimals=1) ; ax.set_yticks(y_ticks);  ax.set_yticklabels(y_ticks,fontsize=fnt) 
            ax.tick_params(direction='in', length=7, width=2.5, axis='both')
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(2.5)
    #
    plt.show()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def nc2var(files,loc):
    sls     = np.empty((0, 0))
    Qvals   = np.empty((0, 0))
    label   = []
    #
    for fi0,fi1 in enumerate(files):
        #
        # lab=fi1.split('/')[-1].split('_')[0]
        lab=fi1.split('/')[-1].split('.')[2]
        # ................................................................
        # Exract Data.
        dataset = xr.open_dataset(fi1)
        time    = dataset['years'].values
        subt    = np.where(time == 2100)[0][0]
        quant   = dataset['quantiles'].values
        qlevs = np.arange(0.01, 1, 0.01)
        subq = np.where((np.around(qlevs,decimals=2) == 0.17) | (np.around(qlevs,decimals=2) == 0.83))[0]
        #
        sl      = dataset['sea_level_change'].values[:,subt,loc].T
        qvals   = np.percentile(sl, qlevs * 100)
        if sls.size == 0: 
            sls = sl
            label=lab
            Qvals=qvals
        else: 
            sls=np.vstack([sls,sl])
            label=np.vstack([label,lab])
            Qvals=np.vstack([Qvals,qvals])
    
    return sls, Qvals, label, quant, qlevs, subq 


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def filePATH(ssp): # Creates a list of file paths.
    """"""""""""""""""
    # Make sure to have all the files in a single ssp folder
    """"""""""""""""""
    # path=f'/scratch/pk695/FACTS/002_fork/facts/JupNbk/000_pk-JupNb_TESTspace/2023-01_NZ_INSAR/002_Pbox/4_confidence_level_files/medium_confidence/{ssp}'
    # files = glob.glob(path + '/*.nc')
    #
    path='/scratch/pk695/FACTS/002_fork/facts/JupNbk/000_pk-JupNb_TESTspace/2023-01_NZ_INSAR/002_Pbox/2_workflow_quantiles/'
    files=[path+f'wf_1e/{ssp}/coupling.{ssp}.emuAIS.emulandice.AIS_globalsl.nc',
           path+f'wf_2e/{ssp}/coupling.{ssp}.larmip.larmip.AIS_globalsl.nc',
           path+f'wf_3e/{ssp}/coupling.{ssp}.deconto21.deconto21.AIS_AIS_globalsl.nc',
           path+f'wf_4/{ssp}/coupling.{ssp}.bamber19.bamber19.icesheets_AIS_globalsl.nc']

    return files