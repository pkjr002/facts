import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
from netCDF4 import Dataset
from pandas.plotting import table 
import xarray as xr
from datetime import date
import time
import os


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def plotALL(ssp,data_nc,COMPONENT,unit):

    XAX1          = data_nc.years.values
    xlim, xlim1   = [2020,2100], [.1,2]

    color_AIS      = np.array([0, 173,207])/255
    color_GIS      = np.array([23  ,60 ,  102])/255
    color_WAIS     = np.array([231 ,29 ,  37])/255
    color_EAIS     = np.array([149 ,27 ,  30])/255
    color_PEN      = np.array([247 ,148,  32])/255

    plt.rcParams.update({'figure.figsize':(40,35), 'figure.dpi':100})
    grid = plt.GridSpec(4, 5, wspace=0.1, hspace=-.4)
    grid00 = grid[0].subgridspec(4, 5); grid01 = grid[1].subgridspec(4, 5)
    

    # Subplot Axis (Left plots)
    ax1_GIS,  ax2_GIS     = plt.subplot(grid00[0, :4]),    plt.subplot(grid00[0, 4]);
    ax1_WAIS, ax2_WAIS    = plt.subplot(grid00[1, :4]),    plt.subplot(grid00[1, 4]);
    ax1_PEN,  ax2_PEN     = plt.subplot(grid00[2, :4]),    plt.subplot(grid00[2, 4]);
    # (Right plots)
    ax1_AIS,  ax2_AIS     = plt.subplot(grid01[0, :4]),    plt.subplot(grid01[0, 4]);
    ax1_EAIS, ax2_EAIS    = plt.subplot(grid01[1, :4]),    plt.subplot(grid01[1, 4]);
    # ax1_,   ax2_        = plt.subplot(grid01[2, :4]),    plt.subplot(grid01[2, 4]);


    if unit == 'cm': convert = 10; axisfactor = 1;
    if unit == 'gt': convert = 1;  axisfactor = 3600;


    for component in COMPONENT: 
        ax1 = eval(f'ax1_{component}');    ax2 = eval(f'ax2_{component}')
        #
        # =======================
        # Plot Left (line plot)
        idxp50 = np.where(data_nc.quantiles.values == 50)[0]; 
        Yax50=np.squeeze(eval(f'(data_nc.{component}_sea_level_change.values[idxp50,:])/convert'))
        #
        idxp17 = np.where(data_nc.quantiles.values == 17)[0]
        Yax17=np.squeeze(eval(f'(data_nc.{component}_sea_level_change.values[idxp17,:]/convert)'))
        #
        idxp83 = np.where(data_nc.quantiles.values == 83)[0]
        Yax83=np.squeeze(eval(f'(data_nc.{component}_sea_level_change.values[idxp83,:]/convert)'))
        #
        ax1.plot(XAX1, Yax50, label = f'{component} (median, 17-83 shading)', color = eval(f'color_{component}'))
        # if component == 'GIS': ax1.text(0.1, 0.7, '(median solid line, 17-83 shading)', fontsize=7, color='black', transform=ax1.transAxes)
        
        ax1.fill_between(XAX1, Yax17, Yax83, color = eval(f'color_{component}'), alpha=0.2)
        ax1.axhline(y=0, color='black', linewidth=0.5)
        # =======================
        # Plot Right (whisker plot)
        XAX2    = np.array([[.2,.2]]) 
        idxt    = np.where(data_nc.years.values == 2100)[0]
        dumYax1 = eval(f'(data_nc.{component}_sea_level_change.values[idxp17,idxt])/convert')
        dumYax2 = eval(f'(data_nc.{component}_sea_level_change.values[idxp83,idxt])/convert')
        dumYax3 = eval(f'(data_nc.{component}_sea_level_change.values[idxp50,idxt])/convert')
        Yax2_1  = np.concatenate((dumYax1, dumYax2))
        Yax2_3=np.concatenate((dumYax3, dumYax3))
        #
        ax2.plot(XAX2.transpose(), Yax2_1.transpose(),color = eval(f'color_{component}'),linewidth=1)
        ax2.plot(XAX2.transpose(), Yax2_3.transpose(), color = eval(f'color_{component}'),marker = 'o',ms = 2,mfc='none')
        #
        #=======================
        # Use Common Yaxis limits across panels.
        if component == 'GIS':  ylim =[0, 18*axisfactor]
        if component == 'AIS':  ylim =[0, 18*axisfactor]
        if component == 'WAIS':  ylim =[-2*axisfactor, 13*axisfactor]
        if component == 'EAIS':  ylim =[-2*axisfactor, 13*axisfactor]
        if component == 'PEN':  ylim =[-.5*axisfactor, 3*axisfactor]
        #=======================
        # Ax properties.
        ax1.legend(loc='upper left',prop={'size': 10}); 
        ax1.tick_params(axis='both', labelsize=8)
        ax1.yaxis.set_ticks_position('both')
        ax1.set_xlim(xlim); #
        ax1.set_ylim(ylim); ax2.set_ylim(ylim); 
        ax2.axis('off'); ax2.set_xlim(xlim1);
        if unit == 'cm': ax1.set_ylabel("GMSL contribution (cm)", size=8);
        if unit == 'gt': ax1.set_ylabel("GMSL contribution (Gt)", size=8);
        # ax1.text(0.8, 0.9, 'ssp'+str(ssp), fontsize=12, color='black', transform=ax1.transAxes)
        if component == 'GIS': ax1.text(1.1, 1.2, 'ISMIP6:: ssp'+str(ssp)+':: FACTS 1.0 (module: emulandice)', fontsize=22, color='black', ha='center', va='center', transform=ax1.transAxes)
    plt.show()