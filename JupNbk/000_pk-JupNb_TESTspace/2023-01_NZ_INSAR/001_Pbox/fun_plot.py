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
colors = {
    'ssp119' : np.array([0, 173, 207]) / 255,
    'ssp126' : np.array([23, 60, 102]) / 255,
    'ssp245' : np.array([247, 148, 32]) / 255,
    'ssp370' : np.array([231, 29, 37]) / 255,
    'ssp585' : np.array([149, 27, 30]) / 255
}
#
