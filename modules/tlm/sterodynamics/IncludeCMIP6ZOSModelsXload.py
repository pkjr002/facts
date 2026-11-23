import numpy as np
import os
import sys
import pandas as pd
import xarray as xr
import momlevel
from netCDF4 import Dataset
import cftime

''' IncludeCMIP6ZOSModelsXload.py

This script parses through a directory of models and loads annual mean 'zos' data from each model.
A directory structure of 'variable'>'Model' is expected.
Historical and SSP files are expected for each model, if not the model is excluded from the ensemble.

Parameters:
model_dir       = Directory of model output. Each model is a subdirectory within this one.
varname        = Name of the variables of interest
years           = Years of interest.
scenario    = SSP of interest

Return:
model_list  = Vector of model names that are to be included (nmodels)
ZOS     = Sea-level change (years, nmodels)

'''

def IncludeCMIP6ZOSModelsXload(model_dir, varname, years, include_models, include_scenarios, focus_sites_lats, focus_sites_lons, focus_sites_ids, focus_sites_names):

    # Save to csv file to feed into momlevel function
    df = pd.DataFrame({'PSMSL_site': focus_sites_names,'PSMSLID': focus_sites_ids,'lat': focus_sites_lats,'lon': focus_sites_lons})
    df.to_csv('location.csv', index=False)

    # Initialize the model list and data matrix
    model_list = []
    scenario_list = []

    # Initialize variables to hold the IDW weights and indices
    ZOS = []

    # Loop through available models in model_dir
    for i in np.arange(len(include_models)):
        
        # Try this model/scenario pair
        model = include_models[i]
        scenario = include_scenarios[i]
        
        # Skip if the folder/file found in this directory is hidden or directory is a file with a . extension
        if '.' in model:
            continue
            
        # Skip if this model is not available
        if model not in os.listdir(model_dir):
            continue
                
        # Initialize lists to store data
        runtype_data = {'historical': [], scenario: []}
        runtype_datayrs = {'historical': [], scenario: []}
        runtype_fnames = {'historical': [], scenario: []}
        
        incorporate = True  # incorporate model or not
        
        # Read in historical and ssp data
        for runtype in (scenario, 'historical'):
            
            # Find the historical or ssp file you want to read in for this model (exact filename depends on the experiment years)
            filename = []
            for files_forModel in os.listdir(os.path.join(model_dir, model)):  # loop through files in model folder
                if (varname + '_Omon_' + model + '_' + runtype) in files_forModel or (varname + '_Oyr_' + model + '_' + runtype) in files_forModel:
                    if not any(np.any(runtype_data[item]) for item in runtype_data) or (files_forModel.split('_')[1] in runtype_fnames[scenario] and files_forModel.split('_')[4] in runtype_fnames[scenario]):
                        filename = files_forModel
                        break
                    
            if not filename:  # if the right filename cannot be found:
                incorporate = False
                
            if incorporate:
                # Open the netCDF file using xarray
                ds = xr.open_dataset(os.path.join(model_dir, model, filename), use_cftime=True)
                nc_fid = Dataset(os.path.join(model_dir, model, filename), 'r')
                
                # If this is historical, collect the model lats and lons and calculate
                #  the maximum allowed distance based on average horizontal resolution
                if runtype == 'historical':
                    # Raw model lats/lons
                    model_lats = nc_fid.variables['lat'][:]
                    model_lons = nc_fid.variables['lon'][:]
                    degd = max(np.mean(np.abs(np.diff(model_lats))), np.mean(np.abs(np.diff(model_lons))))
                    max_dist = (111 * degd) * 1.5
                    
                # Read out the data
                datatime = nc_fid.variables['time']
                fill_value = 9.9692100e+36
                dat = ds[varname].where(ds[varname] != fill_value, np.nan)
                dat = dat.fillna(np.nan)
                
                # Calculate the years
                nctime = cftime.num2date(datatime[:], datatime.units, datatime.calendar)
                datayrs = [int(x.strftime("%Y")) for x in nctime]
                
                # if monthly means, convert to annual
                if datayrs[0] == datayrs[1]:
                    deltat_mon = ds.time_bnds[:, 1].values - ds.time_bnds[:, 0].values
                    dt = xr.DataArray([deltat.total_seconds() for deltat in deltat_mon], dims=['time'], coords=[ds.time])
                    dat = (dat * dt).groupby('time.year').sum('time') / dt.groupby('time.year').sum('time')
                    dat = dat.where(dat != 0.0, np.nan)
                    datayrs = datayrs[0::12]
                else:
                    dat['year'] = dat.time.dt.year
                    dat = dat.swap_dims({'time':'year'})
                    
                #store into dict for each cmip6 runtype
                runtype_data[runtype] = dat
                runtype_datayrs[runtype] = np.array(datayrs)
                runtype_fnames[runtype] = filename
                
        if incorporate:
            # Check for overlap of historical and scenario datasets using data years
            # Set False if historical years are in scenario
            overlap = np.isin(runtype_datayrs['historical'], runtype_datayrs[scenario], invert=True)
            runtype_datayrs['historical'] = runtype_datayrs['historical'][overlap]
            runtype_data['historical'] = runtype_data['historical'].sel(year=runtype_datayrs['historical'])
            
            fullyrs = np.concatenate((runtype_datayrs['historical'], runtype_datayrs[scenario]))
            da = xr.concat([runtype_data['historical'], runtype_data[scenario]], dim='year')
            
            # Put the ZOS data onto the requested years
            da = da.interp(year=xr.DataArray(years, dims=['year']), method='linear', kwargs={'fill_value': np.nan})

            # Shift the longitude by rolling and then adjusting the coordinate
            da_shifted = da.roll(lon=180, roll_coords=True)
            da_shifted['lon'] = ((da_shifted['lon'] + 180) % 360) - 180
            
            # Create land mask
            mask = (~da_shifted.mean('year').isnull()).astype(int)
            
            # Extract time series from each location
            ds_tg = momlevel.extract_tidegauge(da_shifted,xcoord='lon',ycoord='lat',mask=mask,threshold=max_dist,csv='location.csv')
            
            # Add missing variables with NaN values
            for site in focus_sites_names:
                if site not in ds_tg.data_vars:
                    ds_tg[site] = (('year'), np.full(ds_tg.sizes['year'], np.nan)) 
            
            # Reorder the dataset to match the order of focus_sites_names
            ds_tg = ds_tg[focus_sites_names]

            # Convert the dataset to a DataArray
            da_tg = ds_tg.to_array(dim='location', name='zos')
            da_tg['model'] = model
            
            # Add this model to the overall data structure
            ZOS.append(da_tg)
            
            # Append the model to the model list
            model_list.append(model)
            scenario_list.append(scenario)
        
    # Convert ZOS to a numpy array and reshape (years, models, sites)
    ZOS = xr.concat(ZOS, dim='model').values
    ZOS = np.transpose(ZOS, axes=(2,0,1))
    
    return(model_list, scenario_list, ZOS)

if __name__ == '__main__':

    modeldir = "./data/cmip6/zos"
    varname = "zos"
    datayears = np.arange(1861,2300)
    include_models = os.listdir(modeldir)
    include_scenarios = np.repeat("ssp119", len(include_models))

    site_lats = [-7,-7,-7]
    site_lons = [-108,-107,-106]
    site_ids = ["001", "002", "003"]
    site_names = ["Site1", "Site2", "Site3"]

    (modellist, scenariolist, ZOS) = IncludeCMIP6ZOSModelsXload(modeldir, varname, datayears, include_models, include_scenarios, site_lats, site_lons, site_ids, site_names)

    print(modellist)
    sys.exit()
