import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn


# Function for printing specific system messages to std out
def sys_msg(msg):
    print(f'[MESSAGE] {msg}')
    return None


# Function to obtain the GMSL projections for a specific module for a specified year as well as the quantiles
# dat (string): the Path to the the data file wished to be open
# year (int): The year to pull the data from
def get_module_data(filename, year=2100):
    data = (xr.open_dataset(filename).squeeze(drop=True).sea_level_change.sel(years=year, drop=True) / 1000).values
    module_data = []
    for i in range(len(data)):
        module_data.append(data[i])
    return module_data


# Function to obtain the GSAT surface air temperature from the FAIR Temperature Module
# dat (string): the Path to the the data file wished to be open
# year (int): The year to pull the data from
def get_gsat_data(filename, year=np.arange(2081, 2100)):
    data = (xr.open_dataset(filename).squeeze(drop=True).surface_temperature.sel(years=year, drop=True)).values
    gsat_data = []
    for i in range(len(data)):
        current_avg = np.average(data[i])
        gsat_data.append(current_avg)
    return gsat_data


# Bins go from 0.0 C to 8.0 C in increments of 0.5 C
def bin_data(data1, data2, bin_start=0.25, bin_stop=5, interval=0.5, cutoff=5, term_out=False, plot_ax=True):
    # data1 is the FAIR GSAT Data
    # data2 is the COMBINED GMSL data

    # bins the data and creates a pandas dataframe
    bins = np.arange(start=bin_start, stop=bin_stop, step=interval)
    bin_idxs = np.arange(start=0, stop=len(bins) - 1)

    # Creates a Pandas Dataframe and defines the values to go in the individual bins outlined above
    dataframe = pd.DataFrame({"gsat": data1, "gmsl": data2})
    dataframe['binID'] = pd.cut(dataframe['gsat'], bins, include_lowest=True, labels=bin_idxs)

    for bin_idx in range(len(bin_idxs)):
        counter = 0
        for gmsl_idx in range(len(dataframe['gsat'])):
            if dataframe['binID'][gmsl_idx] == bin_idx:
                counter += 1

    print('BIN DATA')

    for bin_idx in range(len(bin_idxs)):
        current_bin = []
        current_pos = bins[bin_idx] + (interval / 2)
        for gmsl_idx in range(len(dataframe['gmsl'])):
            if dataframe['binID'][gmsl_idx] == bin_idx:
                current_bin.append(dataframe['gmsl'][gmsl_idx])

        if len(current_bin) >= cutoff:
            # Gets the proper quantiles from np.quantile
            bin_quants = np.quantile(current_bin, [.05, .17, .5, .83, .95])
            print(f"&{current_pos} {np.round(bin_quants[2],2)} ({np.round(bin_quants[1],2)}-{np.round(bin_quants[3],2)})")
            box_color = 'black'
            median_color = 'white'

            for i in range(len(bin_quants)):
                plt.vlines(x=current_pos, ymin=bin_quants[0], ymax=bin_quants[4], color='black')

            # Plots a box whisker plot of the quantiles defined above
            plt.boxplot(bin_quants,
                        positions=[current_pos],
                        sym="",
                        whis=0,
                        manage_ticks=False,
                        patch_artist=True,
                        boxprops=dict(facecolor=box_color, color=box_color),
                        capprops=dict(color=box_color),
                        whiskerprops=dict(color=box_color),
                        flierprops=dict(color=box_color, markeredgecolor=box_color),
                        medianprops=dict(color=median_color)
                        )

        if plot_ax:
            for i in range(len(bin_idxs)):
                plt.axvline(bins[i] + (interval / 2),
                            linewidth=0.5,
                            linestyle="--",
                            color=(0, 0, 0, 0.01))

    return dataframe


def get_quants(data, ssp_label=''):
    quants = np.quantile(data, [.05, .17, .5, .83, .95])
    print(f'& {np.round(quants[2],2)} ({np.round(quants[1],2)}-{np.round(quants[3],2)})')
    return None


# ---------------------------------------------------------------------------------------------------------------------

# List of SSPS used in the temperature driven modules
ssps = ['coupling.ssp119', 'coupling.ssp126', 'coupling.ssp245', 'coupling.ssp370', 'coupling.ssp585']

mod_names = {'emulandice_ais': ['coupling.ssp119.emuAIS.emulandice.AIS_globalsl.nc', -0.05, .25, 'EMULANDICE/AIS'],
             'emulandice_gris': ['coupling.ssp119.emuGrIS.emulandice.GrIS_globalsl.nc', -0.05, .225, 'EMULANDICE/GrIS'],
             'emulandice_glaciers': ['coupling.ssp119.emuglaciers.emulandice.glaciers_globalsl.nc', .03, .23,
                                     'EMULANDICE/GLACIERS'],
             'ipccar5_glaciers': ['coupling.ssp119.ar5glaciers.ipccar5.glaciers_globalsl.nc', .02, .275,
                                  'IPCCAR5/GLACIERS'],
             'ipccar5_ais': ['coupling.ssp119.ar5AIS.ipccar5.icesheets_AIS_globalsl.nc', -.1, .2, 'IPCCAR5/AIS'],
             'ipccar5_gris': ['coupling.ssp119.ar5AIS.ipccar5.icesheets_GIS_globalsl.nc', .01, .35, 'IPCCAR5/GrIS'],
             'larmip_ais': ['coupling.ssp119.larmip.larmip.AIS_globalsl.nc', -.05, .65, 'LARMIP/AIS'],
             'fittedismip_gris': ['coupling.ssp119.GrIS1f.FittedISMIP.GrIS_GIS_globalsl.nc', 0, .25,
                                  'FITTEDISMIP/GRIS'],
             'tlm_sterodynamics': ['coupling.ssp119.ocean.tlm.sterodynamics_globalsl.nc', .07, .45, 'TLM/STERODYNAMICS'],
             'bamber19_ais': ['coupling.ssp119.bamber19.bamber19.icesheets_AIS_globalsl.nc', -.20, 1.50,
                              'BAMBER19/AIS'],
             'bamber19_gris': ['coupling.ssp119.bamber19.bamber19.icesheets_GIS_globalsl.nc', 0, 1.03, 'BAMBER19'
                                                                                                            '/GrIS'],
             'deconto21_ais': ['coupling.ssp119.deconto21.deconto21.AIS_AIS_globalsl.nc', .05, .65, 'DECONTO21/AIS']

             }

module = 'tlm_sterodynamics'
module_name = mod_names[module]
plot_title = module_name[3]

# Flag to tell if using the FACTSv1.1.0 (Modified) or FACTSv1.0.0 (Unmodified) data
use_mod = True
use_auto = False

fair_name = 'temperature.fair.temperature_gsat.nc'

xlim_range = [0.5, 5.5]

marker_s = 40  # Default is 40
alpha_val = 0.1  # Default is 0.1
figure_dim = [8, 5]

if use_mod is False:
    mod_flag = 'unmodded'
    mod_save_flag = 'unmodified'
    title_flag = '[FACTS V1.0.0]'
else:
    mod_flag = 'modded'
    mod_save_flag = 'modified'
    title_flag = '[FACTS V1.1.1]'

print('---------------------------------------------------------')
print(f'{module.upper()} {mod_save_flag.upper()} DATA')
print('---------------------------------------------------------')

# Set the Path to the data directory
dat_dir = os.path.join('data/', mod_flag)

# Pulls the FAIR Temperature GSAT data for the 19 year interval
gsat_119 = get_gsat_data(filename=f'{dat_dir}/{ssps[0]}/{ssps[0]}.{fair_name}')
gsat_126 = get_gsat_data(filename=f'{dat_dir}/{ssps[1]}/{ssps[1]}.{fair_name}')
gsat_245 = get_gsat_data(filename=f'{dat_dir}/{ssps[2]}/{ssps[2]}.{fair_name}')
gsat_370 = get_gsat_data(filename=f'{dat_dir}/{ssps[3]}/{ssps[3]}.{fair_name}')
gsat_585 = get_gsat_data(filename=f'{dat_dir}/{ssps[4]}/{ssps[4]}.{fair_name}')

# Pulls the GMSL data out of a single module output
gmsl_119 = get_module_data(filename=f'{dat_dir}/{ssps[0]}/{ssps[0]}.{module_name[0][16:]}')
gmsl_126 = get_module_data(filename=f'{dat_dir}/{ssps[1]}/{ssps[1]}.{module_name[0][16:]}')
gmsl_245 = get_module_data(filename=f'{dat_dir}/{ssps[2]}/{ssps[2]}.{module_name[0][16:]}')
gmsl_370 = get_module_data(filename=f'{dat_dir}/{ssps[3]}/{ssps[3]}.{module_name[0][16:]}')
gmsl_585 = get_module_data(filename=f'{dat_dir}/{ssps[4]}/{ssps[4]}.{module_name[0][16:]}')

# Pulls the quantiles for all the gmsl ssps:
print('SSP DATA')
quants_119 = get_quants(gmsl_119, ssp_label='ssp119')
quants_126 = get_quants(gmsl_126, ssp_label='ssp126')
quants_245 = get_quants(gmsl_245, ssp_label='ssp245')
quants_370 = get_quants(gmsl_370,ssp_label='ssp370')
quants_585 = get_quants(gmsl_585, ssp_label='ssp585')

gsat_combined = np.concatenate((gsat_119, gsat_126, gsat_245, gsat_370, gsat_585))
gmsl_combined = np.concatenate((gmsl_119, gmsl_126, gmsl_245, gmsl_370, gmsl_585))

plt.figure(figsize=(figure_dim[0], figure_dim[1]))

plt.scatter(gsat_119, gmsl_119, marker='o', s=marker_s, color='red', alpha=alpha_val, edgecolors='none',
            label=f'SSP1-1.9')

plt.scatter(gsat_126, gmsl_126, marker='o', s=marker_s, color='blue', alpha=alpha_val, edgecolors='none',
            label=f'SSP1-2.6')

plt.scatter(gsat_245, gmsl_245, marker="o", s=marker_s, color='green', alpha=alpha_val, edgecolors='none',
            label=f'SSP2-4.5')

plt.scatter(gsat_370, gmsl_370, marker="o", s=marker_s, color='orange', alpha=alpha_val, edgecolors='none',
            label=f'SSP3-7.0')

plt.scatter(gsat_585, gmsl_585, marker="o", s=marker_s, color='purple', alpha=alpha_val, edgecolors='none',
            label=f'SSP5-8.5')

ssp_comb = bin_data(gsat_combined,
                    gmsl_combined,
                    bin_start=0.25,
                    bin_stop=7,
                    interval=.5,
                    cutoff=200,
                    term_out=True,
                    plot_ax=False)

plt.xlim(xlim_range[0], xlim_range[1])
if not use_auto:
    plt.ylim(module_name[1], module_name[2])

plt.title(f'{plot_title} {title_flag}\n NSAMPS PER SCENARIO = 2000')
plt.xlabel('2081-2100 Average GSAT [$^\circ$C]')
plt.ylabel(f'2100 GMSL [m]')
plt.legend(bbox_to_anchor=(1,1), loc='upper left')
plt.tight_layout()
plt.savefig(f'data/SSP_Plots/{mod_save_flag}/{module}_{mod_save_flag}.png')
plt.show()