import os
import pickle
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import copy

# ---------------------------------------------------------------------------------------------------------------------------------------------------
""" Load Data """

def load_data_project(pathDICT):
    """
    Load the data produced by the fair project script.
    """
    file=f"{pathDICT['debug_Path']}/project.pkl"
    with open(file, "rb") as f:
        loaded_data     = pickle.load(f)
    # Access each variable
    rffemfull_array     = loaded_data["rffemfull_array"]
    rff_sp_array        = loaded_data["rff_sp_array"]
    #
    temps               = loaded_data["temps"]
    deeptemps           = loaded_data["deeptemps"]
    ohcs                = loaded_data["ohcs"]

    return rffemfull_array, rff_sp_array, temps, deeptemps, ohcs 



def load_data_preprocess(pathDICT):
    """
    Load pre-processed data from the fair module. 
    """
    file=f"{pathDICT['debug_Path']}/rff.LL.temperature.fair.rffLL_preprocess.pkl"
    with open(file, 'rb') as f:
        preprocess_data = pickle.load(f)
    emis	 			= preprocess_data["emis"]
    rffemissions	 	= preprocess_data["rffemissions"].copy()
    REFERENCE_YEAR 	    = preprocess_data["REFERENCE_YEAR"]
    t 					= np.arange(REFERENCE_YEAR, 2501)
    # pairds 			    = preprocess_data["pairds"]
    # scenario 		    = preprocess_data["scenario"]
    # rcmip_file 		    = preprocess_data["rcmip_file"]

    return emis, rffemissions, REFERENCE_YEAR, t


# ---------------------------------------------------------------------------------------------------------------------------------------------------
""" Dictionaries. """

rffemissionsGASIDX={
    "C": 0,
    "CH4": 1,
    "N2": 2 }

emisGASIDX={
    "C": 1,
    "CH4": 3,
    "N2O": 4,
    "N2": 4 }



# ---------------------------------------------------------------------------------------------------------------------------------------------------
""" PLOT """

def plot_grouped_dataset(emis, rffemissions, rffemfull_array, rff_sp_array, sample): 
    ''' These plots has only 1 track for each gas) '''

    ''' Figure Window spec'''
    # fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    # fig, axs = plt.subplots(1, 3, figsize=(15, 3.75))
    fig, axs = plt.subplots(1, 3, figsize=(35*.5, 4.75*.5))
    fig.subplots_adjust(wspace=0.5, hspace=0.5) 
    
    ''' ===========> PLOT emis  <=========== '''

    ax0=axs[0]
    title="Base Emission" # (40 gas species)
    # C -->
    ax0.plot(emis[:,0] , emis[:,emisGASIDX["C"]],'.b',label='C'); 
    # CH4-->
    ax00 = ax0.twinx()
    ax00.plot(emis[:,0] , emis[:,emisGASIDX["CH4"]],'.r',label='CH4'); 
    ax00.tick_params(axis='y', labelcolor='r'); ax00.spines['right'].set_color('r')
    # N2 --> 
    ax0.plot(emis[:,0] , emis[:,emisGASIDX["N2"]],'.g',label='"N2"'); 
    # Ax properties
    ax0.legend(loc="upper left"); ax00.legend(loc="upper right")
    ax0.set_title(f'Base Emissions'); ax0.grid(True)


    ''' ===========>
    For the next 2 plots:
    Since there are 10k track for each gas, we have to select a 
    > sample
    > corresponding draw of rff sp
    '''

    # Get the randomized RFF sp draw. 
    rffspIDX=rff_sp_array[sample]
    # print(f" \n rffspIDX = {rffspIDX} \n")

    

    ''' ===========> PLOT rff sps (3gas)   <=========== '''
    ax1=axs[1]
    # C
    ax1.plot(rffemissions['Year'],rffemissions['emissions'][rffemissionsGASIDX["C"],rffspIDX,:],'.b',label='C'); 
    # CH4
    ax11 = ax1.twinx()
    ax11.plot(rffemissions['Year'],rffemissions['emissions'][rffemissionsGASIDX["CH4"],rffspIDX,:],'.r',label='CH4'); 
    ax11.tick_params(axis='y', labelcolor='r'); ax11.spines['right'].set_color('r')
    # N2
    ax1.plot(rffemissions['Year'],rffemissions['emissions'][rffemissionsGASIDX["N2"],rffspIDX,:],'.g',label='N2'); 

    '''dummy time: Done because RFFSps range 2020-2100 '''
    ax1.plot(emis[:,0],emis[:,0],'.w')
    # Ax properties
    ax1.legend(loc="upper center"); ax11.legend(loc="upper right")
    ax1.set_title(f' Rff-SP 3gas.data| --> rffsp10k draw:: {rffspIDX} \n corresponding to Sample {sample}'); ax1.grid(True)


    ''' ===========> PLOT  New Emission   <=========== 

    RFF draw |replacing| base emis(40 gas species, but 3 indice used) 2020-2300
    '''
    ax2=axs[2]
    # C
    ax2.plot(rffemfull_array[sample,:,0],rffemfull_array[sample,:,emisGASIDX["C"]],'.b',label='C'); 
    # CH4
    ax22 = ax2.twinx()
    ax22.plot(rffemfull_array[sample,:,0],rffemfull_array[sample,:,emisGASIDX["CH4"]],'.r',label='CH4'); 
    ax22.tick_params(axis='y', labelcolor='r'); ax22.spines['right'].set_color('r')
    # N2
    ax2.plot(rffemfull_array[sample,:,0],rffemfull_array[sample,:,emisGASIDX["N2"]],'.g',label='N2'); 
    # Ax properties
    ax2.legend(loc="upper left"); ax22.legend(loc="upper right")
    ax2.set_title(f'rff em full - rff10k draw:: {rffspIDX}'); ax2.grid(True)
    ax2.set_title(f'emis wi Rffsp append| --> Sample {sample} \n corresponding to rff10k draw:: {rffspIDX}'); ax1.grid(True)

    for ii in range(3): axs[ii].axvline(x=2020, color='black', linestyle='-', linewidth=3)
    # for ii in range(3):  axs[ii].set_xlim(2020,2300); 
    for ii in range(3):  axs[ii].set_ylim(-10,50);
    ax00.set_ylim(0,800); ax11.set_ylim(0,800); ax22.set_ylim(0,800);


# ---------------------------------
def plot_grouped_gas(emis, rffemissions, rffemfull_array, rff_sp_array, sample): 


    ''' Since there are 10k track for each gas, we have to select a 
    > sample
    > corresponding draw of rff sp
    '''
    # Get the randomized RFF sp draw. 
    rffspIDX=rff_sp_array[sample]
    # print(f" \n rffspIDX = {rffspIDX} \n")


    ''' Figure Window spec'''
    # fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    # fig, axs = plt.subplots(1, 3, figsize=(15, 3.75))
    fig, axs = plt.subplots(1, 3, figsize=(35*.5, 4.75*.5))
    fig.subplots_adjust(wspace=0.5, hspace=0.5) 
    
    ''' ===========> PLOT GAS C <=========== '''
    ax0=axs[0]
    ''' emis -->'''
    ax0.plot(emis[:,0] , emis[:,emisGASIDX["C"]],'or',label='C- base', mfc='none', mew=.2, ms=5);

    ''' rffemissions '''
    #for idx in range(rffemissions['emissions'].shape[1]):  # Iterate over all rffspIDX
        #ax0.plot(rffemissions['Year'],rffemissions['emissions'][rffemissionsGASIDX["C"], idx, :], color='green', alpha=0.3,  lw=.1 )
    ax0.plot(rffemissions['Year'],rffemissions['emissions'][rffemissionsGASIDX["C"],rffspIDX,:],'--b',label='C - rff', mfc='none', mew=.2, lw=3);

    ''' rffemfull_array '''
    ax0.plot(rffemfull_array[sample,:,0],rffemfull_array[sample,:,emisGASIDX["C"]],'.k',label='C - rffemfull', mfc='none', mew=1, ms=1);
    # Ax properties
    ax0.legend(loc="upper left"); 
    ax0.set_title(f'C RFF Emissions | rffspIDX:{rffspIDX}'); ax0.grid(True)

    ''' ===========> PLOT GAS CH4 <=========== '''
    ax1=axs[1]
    ''' emis -->''' 
    ax1.plot(emis[:,0] , emis[:,emisGASIDX["CH4"]],'or',label='CH4- base', mfc='none', mew=.2, ms=5);
    
    ''' rffemissions '''
    #for idx in range(rffemissions['emissions'].shape[1]):  # Iterate over all rffspIDX
        #ax1.plot(rffemissions['Year'],rffemissions['emissions'][rffemissionsGASIDX["CH4"], idx, :], color='green', alpha=0.3,  lw=.1 )
    ax1.plot(rffemissions['Year'],rffemissions['emissions'][rffemissionsGASIDX["CH4"],rffspIDX,:],'--b',label='CH4 - rff', mfc='none', mew=.2, lw=3);
    
    ''' rffemfull_array '''
    ax1.plot(rffemfull_array[sample,:,0],rffemfull_array[sample,:,emisGASIDX["CH4"]],'.k',label='CH4 - rffemfull', mfc='none', mew=1, ms=1);
    
    # Ax properties
    ax1.legend(loc="upper left"); 
    ax1.set_title(f'CH4 RFF Emissions | rffspIDX:{rffspIDX}'); ax1.grid(True)

    ''' ===========> PLOT GAS N2 <=========== '''
    ax2=axs[2]
    ''' emis -->'''
    ax2.plot(emis[:,0] , emis[:,emisGASIDX["N2"]],'or',label='N2- base', mfc='none', mew=.2, ms=5);
   
    ''' rffemissions '''
    #for idx in range(rffemissions['emissions'].shape[1]):  # Iterate over all rffspIDX
        #ax2.plot(rffemissions['Year'],rffemissions['emissions'][rffemissionsGASIDX["N2"], idx, :], color='green', alpha=0.3,  lw=.1 )
    ax2.plot(rffemissions['Year'],rffemissions['emissions'][rffemissionsGASIDX["N2"],rffspIDX,:],'--b',label='N2 - rff', mfc='none', mew=.2, lw=3);

    ''' rffemfull_array '''
    ax2.plot(rffemfull_array[sample,:,0],rffemfull_array[sample,:,emisGASIDX["N2"]],'.k',label='N2 - rffemfull', mfc='none', mew=1, ms=1);
    
    # Ax properties
    ax2.legend(loc="upper left"); 
    ax2.set_title(f'N RFF Emissions | rffspIDX:{rffspIDX}'); ax2.grid(True)



# ---------------------------------------------------------------------------------------------------------------------------------------------------
''' Prep the RFF emissions '''

def prep_rff(baseem, rffemissions, rff_sp,REFERENCE_YEAR):
    # """
    #     takes the raw RFF-SP emissions (rffemissions) and combines with background 
    #     emissions from ssprcp (baseem) for a given rff_sp
    # """    
    # # maps GHG gas to index in emissions numpy array
    # idxdt = {
    #     "C": 1,
    #     "CH4": 3,
    #     "N2O": 4,
    #     "N2": 4,
    # }
    # styear = rffemissions.Year.values[0]
    # enyear = rffemissions.Year.values[-1]

    # # put the RFF-SP gases into the given background emissions 
    # rffemfull = baseem.copy()
    # # for gas in rffemissions.gas.values:
    # #     # rffemfull[styear-REFERENCE_YEAR:enyear-REFERENCE_YEAR+1,idxdt[gas]] = (rffemissions.sel(gas=gas,rff_sp=rff_sp).emissions.values)
    # #     # rffemfull[styear-REFERENCE_YEAR:enyear-REFERENCE_YEAR+1,idxdt[gas]] = rffemissions.sel(gas=gas,rff_sp=9873).emissions.values
    # #     # rffemfull[styear-REFERENCE_YEAR:enyear-REFERENCE_YEAR+1, idxdt[gas]] = 10
    # #     rffemfull[styear - REFERENCE_YEAR : enyear - REFERENCE_YEAR + 1, idxdt[gas]] = (rffemissions.sel(gas=gas, rff_sp=rff_sp).emissions.values)
    # for gas in rffemissions.gas.values:
    #     #rffemfull[styear - REFERENCE_YEAR : enyear - REFERENCE_YEAR + 1, idxdt[gas]] = (
    #     #    rffemissions.sel(gas=gas, rff_sp=rff_sp).emissions.values
    #     #)
    #     rffemfull[styear - REFERENCE_YEAR : enyear - REFERENCE_YEAR + 1, idxdt[gas]] = (
    #         10
    #     )

    # return rffemfull


    """
    takes the raw RFF-SP emissions (rffemissions) and combines with background
    emissions from ssprcp (baseem) for a given rff_sp
    """

    # maps GHG gas to index in emissions numpy array
    idxdt = {
        "C": 1,
        "CH4": 3,
        "N2O": 4,  # g
        "N2": 4,  # fair takes N2 units for N2O gas
    }
    styear = rffemissions.Year.values[0]
    enyear = rffemissions.Year.values[-1]

    val=[]
    # put the RFF-SP gases into the given background emissions 
    rffemfull = copy.copy(baseem)
    for gas in rffemissions.gas.values:
        rffemfull[styear - REFERENCE_YEAR : enyear - REFERENCE_YEAR + 1, idxdt[gas]] = (
            rffemissions.sel(gas=gas, rff_sp=rff_sp).emissions.values
        )
        val.append(rffemissions.sel(gas=gas, rff_sp=rff_sp).emissions.values)

    return rffemfull, val








def plot_grouped_gas_nbk(emis, rffemissions, rffemfull_array, rff_sp_array, REFERENCE_YEAR, sample):

    ''' Get the randomized RFF sp draw. '''
    rffspIDX=rff_sp_array[sample]; 
    #print(f" \n rffspIDX = {rffspIDX} \n")
    
    ''' Compute the new emissions file'''
    emis_rffsp_full1,val1 = prep_rff(emis, rffemissions, rffspIDX, REFERENCE_YEAR)

    ''' PLOT the emissions'''    
    fig, ax = plt.subplots(1, 3, figsize=(35*.5, 4.75*.5))
    fig.subplots_adjust(wspace=0.5, hspace=0.5) 

    ax[0].plot(emis[:,0] , emis[:,1],'or',label='C- base', mfc='none', mew=.2, ms=5);
    ax[0].plot(emis_rffsp_full1[:,0] , emis_rffsp_full1[:,1], '.k')
    ax[0].plot(rffemissions['Year'],rffemissions['emissions'][rffemissionsGASIDX["C"],rffspIDX,:],'--b',label='C - rff', mfc='none', mew=.2, lw=3);
    ax[0].grid(True)

    ax[1].plot(emis[:,0] , emis[:,3],'or',label='CH4- base', mfc='none', mew=.2, ms=5); 
    ax[1].plot(emis_rffsp_full1[:,0] , emis_rffsp_full1[:,3], '.k')
    ax[1].plot(rffemissions['Year'],rffemissions['emissions'][rffemissionsGASIDX["CH4"],rffspIDX,:],'--b',label='CH4 - rff', mfc='none', mew=.2, lw=3);
    ax[1].grid(True)

    ax[2].plot(emis[:,0] , emis[:,4],'or',label='N2- base', mfc='none', mew=.2, ms=5);
    ax[2].plot(emis_rffsp_full1[:,0] , emis_rffsp_full1[:,4], '.k')
    ax[2].plot(rffemissions['Year'],rffemissions['emissions'][rffemissionsGASIDX["N2"],rffspIDX,:],'--b',label='N2 - rff', mfc='none', mew=.2, lw=3);
    ax[2].grid(True)
    


def plot_3_gas(emis, rffemissions, rff_sp_array, sample):
    ''' Plot the 3gas emission for a particular draw '''


    ''' Get the randomized RFF sp draw. '''
    rffspIDX=rff_sp_array[sample]; 
    print(f" \n rffspIDX = {rffspIDX} \n")
    

    ''' PLOT the emissions'''    
    fig, ax = plt.subplots(1, 3, figsize=(35*.5, 4.75*.5))
    fig.subplots_adjust(wspace=0.5, hspace=0.5) 

    ax[0].plot(emis[:,0] , emis[:,1],'or',label='C- base', mfc='none', mew=.2, ms=5);
    ax[0].plot(rffemissions['Year'],rffemissions['emissions'][rffemissionsGASIDX["C"],rffspIDX,:],'--b',label='C - rff', mfc='none', mew=.2, lw=3);
    ax[0].grid(True)

    ax[1].plot(emis[:,0] , emis[:,3],'or',label='CH4- base', mfc='none', mew=.2, ms=5);
    ax[1].plot(rffemissions['Year'],rffemissions['emissions'][rffemissionsGASIDX["CH4"],rffspIDX,:],'--b',label='CH4 - rff', mfc='none', mew=.2, lw=3);
    ax[1].grid(True)

    ax[2].plot(emis[:,0] , emis[:,4],'or',label='N2- base', mfc='none', mew=.2, ms=5);
    ax[2].plot(rffemissions['Year'],rffemissions['emissions'][rffemissionsGASIDX["N2"],rffspIDX,:],'--b',label='N2 - rff', mfc='none', mew=.2, lw=3);
    ax[2].grid(True)