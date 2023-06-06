
import numpy as np
import pandas as pd
import xarray as xr
from mpl_toolkits.basemap import Basemap
import glob
import os
import shutil
import re
import cartopy
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
#
# %matplotlib inline
# %matplotlib notebook
#
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap, BoundaryNorm



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def fileNAME(patH,name):
    import glob
    folder_path = patH
    search_term = name   # "glaciers"  # replace with the word you want to search for
    file_pattern = f"{folder_path}/*{search_term}*"  # create a pattern to match files containing the search term
    matching_files = glob.glob(file_pattern)
    if len(matching_files)>1: 
        raise ValueError("There are 2 files with same keyword")
    fnme = os.path.basename(matching_files[0])
    return fnme


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def plot_full_samp_wf(ssp,folder,samp,component,loc):

    if ssp=="ssp126" or ssp=="ssp585":
        WF=['wf_1e','wf_1f','wf_2e','wf_2f','wf_3e','wf_3f','wf_4']
        fig, axs = plt.subplots(nrows=7, ncols=6, figsize=(12*1.5, 5*4))
    elif ssp=="ssp119" or ssp=="ssp245":    
        WF=['wf_1e','wf_1f','wf_2e','wf_2f','wf_3e','wf_3f']
        fig, axs = plt.subplots(nrows=6, ncols=6, figsize=(12*1.5, 5*4))
    elif ssp=="ssp370":
        WF=['wf_1e','wf_1f','wf_2e','wf_2f']
        fig, axs = plt.subplots(nrows=4, ncols=6, figsize=(12*1.5, 5*4))


    for i, wf in enumerate(WF):
        pathG='/projects/kopp/ggg46/nz_projections/regional/{arg0}/{arg1}/{arg2}/'.format(arg0=folder,arg1=wf,arg2=ssp)
        pathP='/scratch/pk695/FACTS/nzOG-pk695/local-pk695/001_{arg0}/{arg1}/{arg2}/'.format(arg0=folder,arg1=wf,arg2=ssp)
        #
        fileG=fileNAME(pathG,component)#'icesheets-ipccar6-ismipemuicesheet-{arg0}_GIS_localsl.nc'.format(arg0=ssp) 
        fileP=fileNAME(pathP,component)#'nzOG.{arg0}.emuGrIS.emulandice.GrIS_localsl.nc'.format(arg0=ssp)
        var='sea_level_change'
        # labels
        EXP_G=pathG.split('/')[3].split('_')[0];   EXP_P=pathP.split('/')[4].split('-')[1];   EXP_file=fileP; 
        #
        d_ncG, d_ncP = [xr.open_dataset(os.path.join(path, file)) for path, file in [(pathG, fileG), (pathP, fileP)]]
        #
        yrST=2020; yrEN=2100; yr=np.arange(yrST, yrEN+1, 10)
        yrIDX_G, yrIDX_P=[np.searchsorted(d['years'].values, yr) for d in [d_ncG, d_ncP]]
        #
        sampIDX_G, sampIDX_P=[np.searchsorted(d['samples'].values, samp) for d in [d_ncG, d_ncP]]
        #         
        latP, lonP = [d_ncP[d][loc].values for d in ['lat', 'lon']]

        # Compute the difference. 
        slcG=d_ncG[var][sampIDX_G, yrIDX_G, loc];       slcP=d_ncP[var][sampIDX_P, yrIDX_P, loc]
        slcG_reshaped = slcG.values.reshape(-1);        slcP_reshaped = slcP.values.reshape(-1)
        diff = slcG_reshaped - slcP_reshaped;           diff = np.reshape(diff, slcG.shape)

        # X-axis
        xx=d_ncG['years'][yrIDX_G].values

        for sam0,sam1 in enumerate(samp):
            row, col = i, sam0
            ax = axs[row, col]
            
            for lo0,lo1 in enumerate(loc):
                yyD=np.squeeze(diff[sam0,:,lo0])
                yy1=np.squeeze(slcG[sam0,:,lo0])
                yy2=np.squeeze(slcP[sam0,:,lo0])
                #
                ax.plot(xx, yy1, label='G')
                ax.plot(xx, yy2, linestyle='--', label='P')
                #ax.plot(xx, yy1, label=str(latP[lo0])+" , "+str(lonP[lo0]))
                #
                ax.grid(alpha=0.95)
                ax.text(.55, .1,ssp+' :: '+wf, fontsize=7, color='blue', transform=ax.transAxes, ha='left', va='center')
                ax.set_xlim(2020, 2100); ax.set_xticks(range(2020, 2100+1, 10))
                ax.tick_params(axis='x', labelsize=6)
                ax.tick_params(axis='y', labelsize=7)
                if lo0<1: 
                    ax.legend(fontsize='small')
                # Xlabels
                if row==6 and col in (0, 1, 2, 3, 4, 5, 6):
                    ax.tick_params(axis='x', labelrotation=45,labelsize=10)
                
                # Ylabels
                if col==0 and row in (0, 1, 2, 3, 4, 5, 6):
                    ax.set_ylabel('slc (mm)', fontsize=10); #ax.set_xlabel('Years')
                
                if row==0 and col in (0, 1, 2, 3, 4, 5):
                    SAMP=str(sam1)
                    ax.set_title(SAMP+' Sample' ,fontsize=10)
           
                
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    plt.show()

    # return fileG