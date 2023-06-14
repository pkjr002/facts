import numpy as np
import pandas as pd
import netCDF4 as nc
from pandas.plotting import table 
import xarray as xr
#
#
# ==================================================================================================================================
# Used to input module datasets for Figure 2 & 4
def module_Fig_nc(df,SCENARIO,EXPDATE, **kwargs):
    yrPTILE = kwargs.get('yrPTILE', None)
    #
    df_TEMP   = [];  df_TEMP_YR = []
    #
    # Loop over .nc files.
    for val in df.index.values:
        for scenario in SCENARIO: 
            module      = df.at[val, 'Module']     
            sub_module  = df.at[val, 'subModule'] 
            component   = df.at[val, 'Component']
            num         = df.at[val, 'Num']
            # Skip if data is absent.
            if df.at[val, 'DataFile'] in 'XXX':
                continue  
            # Open the .nc data file. 
            dataFOLDER  = '/projects/kopp/facts-experiments/{arg2}/coupling.{arg1}/output/'.format(arg1=scenario,arg2=EXPDATE)
            dataFILE    = 'coupling.{arg1}.'.format(arg1=scenario) + df["DataFile"][val]
            d_nc        = xr.open_dataset(dataFOLDER + dataFILE)
            #
            # Percentile calculation.
            percentList = [50, 5, 17, 83, 95]
            # Loop over years.
            for yy in d_nc["years"].values:
                if (yy > 2100) or (yy ==2005):
                    continue
                else:
                    # Find year index to pick SLC value
                    yind = np.where(d_nc["years"].values == yy)[0][0]
                    # .
                    GMSL = (d_nc["sea_level_change"][:,yind,:].values)/10
                    # Find Percentile ranges.
                    pcntle = np.percentile(GMSL[:], percentList );    pcntle = np.around(pcntle,5)
                    #
                    df_TEMP.append( [num,component,module, scenario, yy, ] + pcntle.tolist() )
                    #
                    if yy == yrPTILE: df_TEMP_YR.append( [num,component,module, scenario, ] + pcntle.tolist() )
                    #
    # Dataframe that contains all years percentiles.
    df_ptile = pd.DataFrame( df_TEMP, columns=['Num', 'Component','Module', 'SSP', 'Year', ] + [ f'col_{x}' for xi, x in enumerate( percentList )] )
    #
    # Dataframe that contains only percentile of requested year.
    df_ptile_YR = pd.DataFrame( df_TEMP_YR, columns=['Num', 'Component','Module', 'SSP', ] + [ f'col_{x}' for xi, x in enumerate( percentList )] )
    df_ptile_YR['median(17-83)'] = df_ptile_YR.apply(lambda x: f'{x.col_50:2.2f} ({x.col_17:2.2f} - {x.col_83:2.2f})', axis=1 )
    df_ptile_YR1    = pd.DataFrame( df_ptile_YR.set_index( ['Num','Component','Module','SSP'] )['median(17-83)'] ).unstack().swaplevel( 0,1, axis=1 )
    #
    return df_ptile, df_ptile_YR1
#