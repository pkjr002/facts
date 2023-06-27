region="global"


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
total = {
    "wf_1e": [f'total.workflow.wf1e.{region}.nc'],
    "wf_1f": [f'total.workflow.wf1f.{region}.nc'],
    "wf_2e": [f'total.workflow.wf2e.{region}.nc'],
    "wf_2f": [f'total.workflow.wf2f.{region}.nc'],
    "wf_3e": [f'total.workflow.wf3e.{region}.nc'],
    "wf_3f": [f'total.workflow.wf3f.{region}.nc'],
    "wf_4":  [f'total.workflow.wf4.{region}.nc']
}




# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
common_files = [
    f'lws.ssp.landwaterstorage_{region}sl.nc',
    f'ocean.tlm.sterodynamics_{region}sl.nc',
    f'k14vlm.kopp14.verticallandmotion_localsl.nc'
]

component = {
    "wf_1e": [
        f'emuGrIS.emulandice.GrIS_{region}sl.nc',
        f'emuAIS.emulandice.AIS_{region}sl.nc',
        f'emuglaciers.emulandice.glaciers_{region}sl.nc',
        *common_files   
    ],
    "wf_1f": [
        f'GrIS1f.FittedISMIP.GrIS_GIS_{region}sl.nc',
        f'ar5AIS.ipccar5.icesheets_AIS_{region}sl.nc',
        f'ar5glaciers.ipccar5.glaciers_{region}sl.nc',
        *common_files
    ],
    "wf_2e": [
        f'emuGrIS.emulandice.GrIS_{region}sl.nc',
        f'larmip.larmip.AIS_{region}sl.nc',
        f'emuglaciers.emulandice.glaciers_{region}sl.nc',
        *common_files
    ],
    "wf_2f": [
        f'GrIS1f.FittedISMIP.GrIS_GIS_{region}sl.nc', 
        f'larmip.larmip.AIS_{region}sl.nc', 
        f'ar5glaciers.ipccar5.glaciers_{region}sl.nc', 
        *common_files
    ],
    "wf_3e": [
        f'emuGrIS.emulandice.GrIS_{region}sl.nc', 
        f'deconto21.deconto21.AIS_AIS_{region}sl.nc', 
        f'emuglaciers.emulandice.glaciers_{region}sl.nc', 
        *common_files
    ],
    "wf_3f": [
        f'GrIS1f.FittedISMIP.GrIS_GIS_{region}sl.nc', 
        f'deconto21.deconto21.AIS_AIS_{region}sl.nc', 
        f'ar5glaciers.ipccar5.glaciers_{region}sl.nc', 
        *common_files
    ],
    "wf_4": [
        f'bamber19.bamber19.icesheets_GIS_{region}sl.nc', 
        f'bamber19.bamber19.icesheets_AIS_{region}sl.nc', 
        f'ar5glaciers.ipccar5.glaciers_{region}sl.nc', 
        *common_files
    ],
}
