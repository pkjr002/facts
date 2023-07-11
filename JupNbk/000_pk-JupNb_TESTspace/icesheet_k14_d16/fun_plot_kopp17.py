import numpy as np
import pandas as pd
import xarray as xr
import glob
import os
import shutil
import re
import cartopy
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
#
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap, BoundaryNorm
#
PD=os.getcwd(); PD
# ==================================================================================================


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def LSfile(path, name):
    #
    nclist=os.listdir(path+'/');  nclist=sorted(nclist)
    nclist_local=[]
    for ncname in nclist:
        if name in ncname:
            nclist_local.append(ncname) 
    return(nclist_local)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


