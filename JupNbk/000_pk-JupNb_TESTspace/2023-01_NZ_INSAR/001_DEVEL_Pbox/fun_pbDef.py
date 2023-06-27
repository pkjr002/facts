region="global"
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
common_files = [
    f'lws.ssp.landwaterstorage_{region}sl.nc',
    f'ocean.tlm.sterodynamics_{region}sl.nc',
    f'k14vlm.kopp14.verticallandmotion_localsl.nc'
]
#
pBox = {
    "pb_1e": [
        f'emuAIS.emulandice.AIS_{region}sl.nc',
        f'larmip.larmip.AIS_{region}sl.nc',
        f'emuGrIS.emulandice.GrIS_{region}sl.nc',
        f'emuglaciers.emulandice.glaciers_{region}sl.nc',
        *common_files   
    ],
    "pb_1f": [
        f'ar5AIS.ipccar5.icesheets_AIS_{region}sl.nc',
        f'larmip.larmip.AIS_{region}sl.nc',
        f'GrIS1f.FittedISMIP.GrIS_GIS_{region}sl.nc',
        f'ar5glaciers.ipccar5.glaciers_{region}sl.nc',
        *common_files
    ],
    "pb_2e": [
        f'emuAIS.emulandice.AIS_{region}sl.nc',
        f'larmip.larmip.AIS_{region}sl.nc',
        f'deconto21.deconto21.AIS_AIS_{region}sl.nc',
        f'bamber19.bamber19.icesheets_AIS_{region}sl.nc',
        #
        f'emuGrIS.emulandice.GrIS_{region}sl.nc',
        f'bamber19.bamber19.icesheets_GIS_{region}sl.nc',
        #
        f'emuglaciers.emulandice.glaciers_{region}sl.nc',
        f'ar5glaciers.ipccar5.glaciers_{region}sl.nc',
        *common_files
    ],
    "wf_2f": [
        f'ar5AIS.ipccar5.icesheets_AIS_{region}sl.nc', 
        f'larmip.larmip.AIS_{region}sl.nc',
        f'deconto21.deconto21.AIS_AIS_{region}sl.nc',
        f'bamber19.bamber19.icesheets_AIS_{region}sl.nc',
        #
        f'GrIS1f.FittedISMIP.GrIS_GIS_{region}sl.nc', 
        f'bamber19.bamber19.icesheets_GIS_{region}sl.nc',
        #
        f'emuglaciers.emulandice.glaciers_{region}sl.nc',
        f'ar5glaciers.ipccar5.glaciers_{region}sl.nc',
        *common_files
    ],
}
