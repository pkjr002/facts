import os
import xarray as xr


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def xtract_data_4m_nc(dataNC,var,loc,yrST,yrEN):    
    data = xr.open_dataset(dataNC)
    # index for years.
    time_data = data['years'].values    
    idx_year=np.where((time_data >= yrST) & (time_data <= yrEN))[0]
    #
    time=time_data[idx_year]
    #
    slc = data[var][:,idx_year,loc].values
    lat=data['lat'][loc].values
    lon=data['lon'][loc].values
    
    output = {
        'slc': slc, 'time': time,
        'lat': lat, 'lon': lon
    }
    return output


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
def ncNAME_dict(name)
    


