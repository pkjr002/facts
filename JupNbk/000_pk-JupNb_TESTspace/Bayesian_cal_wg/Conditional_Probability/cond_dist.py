import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def prepData(path, dat1, dat2, Yr1, Yr2):

    var1 = xr.open_dataset(f'{path}{dat1}', engine='netcdf4')
    var2 = xr.open_dataset(f'{path}{dat2}', engine='netcdf4')

    dat1NME = '-'.join(dat1.split(".")[3:])
    dat2NME = '-'.join(dat2.split(".")[3:])
    
    xaxVAR=var1['sea_level_change'][:, np.where(var1['years']==Yr1)[0][0]]
    yaxVAR=var2['sea_level_change'][:, np.where(var2['years']==Yr2)[0][0]] 
    
    #print('convert to cm')
    xaxVAR=xaxVAR/10
    yaxVAR=yaxVAR/10

    datNME = [f'{dat1NME}-{Yr1}-cm', f'{dat2NME}-{Yr2}-cm']
    
    INdata = np.column_stack((xaxVAR, yaxVAR))

    return INdata, datNME


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def PLOTraw(data_list, datNME_list):
    # Determine the number of datasets to plot
    num_data = len(data_list)
    num_plots = min(num_data, 5)  # Limit to 5 plots for the 1x5 grid

    
    fig, axs = plt.subplots(1, 5, figsize=(20, 4), gridspec_kw={'wspace': 0.9})
    
    # Iterate over the provided datasets and plot them
    for idx, data in enumerate(data_list[:num_plots]):

        datNME1 = datNME_list[idx][0]
        datNME2 = datNME_list[idx][1]

        ax = axs[idx]
        numbers = np.linspace(1, data.shape[0], data.shape[0])    
        x_min, x_max = np.floor(np.min(numbers)), np.ceil(np.max(numbers)) 

        # Plot the first column of the dataset
        color = 'tab:blue'
        ax.set_xlabel('Index')
        ax.set_ylabel(datNME1, color=color)
        ax.plot(numbers, data[:, 0], 'o', linestyle='none', color=color)
        ax.tick_params(axis='y', labelcolor=color)
        
        # Create a secondary y-axis for the second column
        ax2 = ax.twinx()  
        color = 'tab:orange'
        ax2.set_ylabel(datNME2, color=color)
        ax2.plot(numbers, data[:, 1], 'x', linestyle='none', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Adjust the y-axis limits based on the data range
        y_max = np.ceil(np.max(data[:, 1]))
        ax.set_ylim(-10, y_max)
        ax2.set_ylim(-10, y_max)

        # ax.set_title(f'Dataset {idx + 1}')

    # Hide any unused subplots
    for idx in range(num_plots, 5):
        axs[idx].set_visible(False)
    
    # plt.tight_layout()
    plt.show()


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def calculate_percentiles(data, percentiles=[0.05, 0.17, 0.5, 0.83, 0.95]):

    x_percentiles = np.percentile(data[:, 0], [p * 100 for p in percentiles])
    y_percentiles = np.percentile(data[:, 1], [p * 100 for p in percentiles])
    
    return x_percentiles, y_percentiles


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def PLOTscatter(data_list,datNME_list):
    # Determine the number of datasets to plot
    num_data = len(data_list)
    num_plots = min(num_data, 5)  # Limit to 5 plots for the 1x5 grid

    fig, axs = plt.subplots(1, 5, figsize=(20, 4), gridspec_kw={'wspace': 0.5})

    # Iterate over the provided datasets and plot them
    for idx, data in enumerate(data_list[:num_plots]):
        ax = axs[idx]

        datNME1 = datNME_list[idx][0]
        datNME2 = datNME_list[idx][1]
        
        #quantile_data = getQuantiles(data)
        #x_percentiles = [q[1] for q in quantile_data if q[0] in [0.05, 0.17, 0.5, 0.83, 0.95]]
        #y_percentiles = [q[2] for q in quantile_data if q[0] in [0.05, 0.17, 0.5, 0.83, 0.95]]
        x_percentiles, y_percentiles = calculate_percentiles(data)

        # Scatter plot of the first column vs. the second column
        ax.scatter(data[:, 0], data[:, 1], s=.5, facecolor='red')
        ax.set_xlabel(datNME1)
        if idx == 0:  ax.set_ylabel(datNME2)
        # ax.set_title(f'Dataset {idx + 1}')
        ax.grid(True)

        # Mark the percentiles on the scatter plot
        mark_percentiles(ax, x_percentiles, y_percentiles)
        
        # Plot the table of percentile values
        plot_table(ax, x_percentiles, y_percentiles, datNME1, datNME2)
        
    # Hide any unused subplots
    for idx in range(num_plots, 5):
        axs[idx].set_visible(False)
    
    # plt.tight_layout()
    plt.show()



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def automatic_bandwidth_range(INdata):

    # ---->number of bandwidth values to test

    # iqr = np.subtract(*np.percentile(INdata, [75, 25]))
    iqr = np.subtract(*np.percentile(INdata, [95, 5]))

    # Use IQR to determine the number of bandwidth values to test
    num_values = min(1000, max(10, int(iqr)))  # Ensures a number between 10 and 1000
    # <---

    # Find appropriate bandwidth array.
    # Flatten the data if it is multidimensional (consider all values)
    data = INdata.flatten() if INdata.ndim > 1 else INdata
    
    # Silverman's rule of thumb for initial bandwidth estimate
    std_dev = np.std(data)
    # iqr = np.subtract(*np.percentile(data, [75, 25]))  # Calculate IQR
    iqr = np.subtract(*np.percentile(data, [95, 5]))  # Calculate IQR
    n = len(data)
    
    # Silverman's bandwidth estimation
    silverman_bw = 0.9 * min(std_dev, iqr / 1.34) * n ** (-1/5)
    
    # Define a range of bandwidths around the estimated bandwidth
    bandwidths = np.linspace(silverman_bw / 2, silverman_bw * 2, num_values)
    
    return bandwidths, num_values



def generate_grids_full_data(INdata, base_points=100):
    # Get the min and max values of the x and y data
    x_min, x_max = INdata[:, 0].min() - 1, INdata[:, 0].max() + 1
    y_min, y_max = INdata[:, 1].min() - 1, INdata[:, 1].max() + 1

    # Calculate the range for x and y
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Calculate the number of points based on the proportion of the ranges
    total_range = x_range + y_range
    x_points = int(base_points * (x_range / total_range))
    y_points = base_points - x_points  # Ensure the total remains around base_points

    # Generate the grids using linspace
    xgrid = np.linspace(x_min, x_max, x_points)
    ygrid = np.linspace(y_min, y_max, y_points)
    
    return xgrid, ygrid




def COMPkde(INdata,print=0):
    #print("Data shape:", INdata.shape)

    #-----optimal BANDWIDTH
    # Grid of bandwidth values to test
    # bandwidths = np.linspace(0.1, 10, 40)  
    bandwidths,num_values = automatic_bandwidth_range(INdata) 

    # Setup the grid search with cross-validation
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                       {'bandwidth': bandwidths},
                       cv=5)  # 5-fold cross-validation
    
    # Fit grid search
    grid.fit(INdata)

    # Best bandwidth
    best_bw = np.round(grid.best_params_['bandwidth'],2)
    bw_kde=best_bw #-----BANDWIDTH
    

    kde = KernelDensity(kernel='gaussian', bandwidth=bw_kde).fit(INdata)
    
    # xgrid = np.linspace(INdata[:,0].min()-1, INdata[:,0].max()+1, 100)  
    # ygrid = np.linspace(INdata[:,1].min()-1, INdata[:,1].max()+1, 100)
    xgrid, ygrid = generate_grids_full_data(INdata, base_points=200)
    
    if print==1:
        print("Optimal bandwidth:", best_bw, " || ", "num_values",num_values, " || ",
            "KDE eval ","xgrid:",len(xgrid)," :: ","ygrid:",len(ygrid))
    
    Xgrid, Ygrid  = np.meshgrid(xgrid, ygrid)
    grid_samples = np.vstack([Xgrid.ravel(), Ygrid.ravel()]).T
    #
    # Eval density model on the grid (log likelihoods)
    log_density_values = kde.score_samples(grid_samples)
    #Reshape 
    log_density_values = log_density_values.reshape(Xgrid.shape)
    #
    #-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    # Density values
    density_values = np.exp(log_density_values)
    #.............................................................................................................
    # Normalize 
    # normalized_density_values = density_values / np.sum(density_values, axis=0);
    normalized_density_values = density_values / (np.sum(density_values, axis=0) + 1e-12)

    # OG percentiles
    xp, yp = calculate_percentiles(INdata)


    ## KDE Percentiles
    ## Flatten the density values to compute CDF
    #flat_density = density_values.ravel()
    #sorted_indices = np.argsort(flat_density)
    #sorted_density = flat_density[sorted_indices]
    ## Compute cumulative density (CDF)
    #cdf = np.cumsum(sorted_density)
    #cdf /= cdf[-1]  # Normalize CDF to range [0, 1]
    ## Calculate percentiles
    #percentiles = [0.05, 0.17, 0.5, 0.83, 0.95]
    #percentile_values = np.interp(percentiles, cdf, sorted_density)


    return normalized_density_values, density_values, xgrid, ygrid,xp, yp



def PLOTkde(density_values_list, xgrid_list, ygrid_list, xp, yp, datNME_list, clevels, table=None):
    # Determine the number of datasets to plot
    num_data = len(density_values_list)
    num_plots = min(num_data, 5)  # Limit to 5 plots for the 1x5 grid

    fig, axs = plt.subplots(1, 5, figsize=(20, 4), gridspec_kw={'wspace': 0.5})

    # Iterate over the provided datasets and plot them
    for idx in range(num_plots):
        normalized_density_values = density_values_list[idx]

        #percentiles = [0.05, 0.17, 0.5, 0.83, 0.95]  
        #percentile_values = np.interp(percentiles, np.cumsum(normalized_density_values.ravel()), 
        #                      np.sort(normalized_density_values.ravel()))
        #percentile_values = np.sort(percentile_values)
        ## Add debug print to verify the percentile values
        #print(f"Percentile Values (before sorting): {percentile_values}")
        ## Sort and ensure unique values for the contour levels
        #percentile_values = np.sort(np.unique(percentile_values))
        ## Ensure at least two distinct contour levels
        #if len(percentile_values) < 2:
        #    raise ValueError("Need at least two unique percentile values for contouring")


        xgrid = xgrid_list[idx]
        ygrid = ygrid_list[idx]

        datNME1 = datNME_list[idx][0]
        datNME2 = datNME_list[idx][1]

        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        ax = axs[idx]

        # Define contour levels and plot the normalized density
        clevels = clevels
        contour = ax.contourf(Xgrid, Ygrid, normalized_density_values, levels=clevels, cmap='Reds')
        # ax.contour(Xgrid, Ygrid, normalized_density_values, levels=percentile_values, colors='black', linestyles='dashed')


        ## Annotate percentile values at multiple representative locations
        #for idx, pval in enumerate(percentile_values):
        #    # Find a representative (x, y) point where the density equals the contour level
        #    # Use the middle of the grid for better positioning
        #    y_index = len(ygrid) // 2  # Middle row
        #    x_index = len(xgrid) // (idx + 1)  # Spread across the plot
        #
        #    ax.annotate(f'{pval:.2f}', 
        #                xy=(xgrid[x_index], ygrid[y_index]),  # New annotation positions
        #                xycoords='data', textcoords='offset points', 
        #                xytext=(5, 5), ha='center', fontsize=8, color='blue')
        
        
        
            

        # ax.set_title(f'Dataset {idx + 1}')
        if idx == num_plots - 1:
            fig.colorbar(contour, ax=ax, label='Probability Density')
        
        ax.set_xlabel(datNME1)
        ax.set_ylabel(datNME2)
        
        # Mark the percentiles on the scatter plot
        mark_percentiles(ax, xp[idx], yp[idx])
        
        # Plot the table of percentile values
        if table == 1:
            plot_table(ax, xp[idx], yp[idx], datNME1, datNME2) 
        
    # Hide any unused subplots
    for idx in range(num_plots, 4):
        axs[idx].set_visible(False)

    # plt.tight_layout()
    plt.show()



def PLOTcondProb(density_values_list, xgrid_list, ygrid_list, datNME_list, x_values, y_limits=None):
    # Determine the number of datasets to plot
    num_data = len(density_values_list)
    num_plots = min(num_data, 5)  # Limit to 5 plots for the 1x5 grid

    fig, axs = plt.subplots(1, 5, figsize=(30, 6))

    # Iterate over the provided datasets and plot them
    for idx in range(num_plots):
        density_values = density_values_list[idx]
        xgrid = xgrid_list[idx]
        ygrid = ygrid_list[idx]

        datNME1 = datNME_list[idx][0]
        datNME2 = datNME_list[idx][1]

        # Unpack the x-range for this plot
        start, stop, num_points = x_values[idx]
        x_range = np.linspace(start, stop, num_points)

        # Prepare an array to hold the conditional densities P(Y|X)
        conditional_densities = np.zeros((len(ygrid), len(x_range)))

        # Loop through each fixed X value
        for i, chosen_x in enumerate(x_range):
            # Find the closest index in xgrid to the chosen X value
            x_idx = np.abs(xgrid - chosen_x).argmin()

            # Extract the joint density values at this X value (slice along y-axis)
            joint_density_at_x = density_values[:, x_idx]

            # Calculate the marginal density P(X=x) (sum over Y)
            marginal_density_x = np.sum(joint_density_at_x)

            # Compute conditional density P(Y|X=x)
            if marginal_density_x > 0:
                conditional_density_y_given_x = joint_density_at_x / marginal_density_x
            else:
                conditional_density_y_given_x = np.zeros_like(joint_density_at_x)  # Handle division by 0

            # Store the conditional density for the chosen X value
            conditional_densities[:, i] = conditional_density_y_given_x

        # Create a plot for the conditional densities
        X, Y = np.meshgrid(x_range, ygrid)
        ax = axs[idx]
        
        # Define contour levels for visualization
        clevels = np.linspace(0.001, np.max(conditional_densities), 6)
        contour = ax.contourf(X, Y, conditional_densities, cmap='Reds', levels=clevels, alpha=0.8)

        # Set plot title and labels
        # ax.set_title(f'Dataset {idx + 1}')
        ax.set_xlabel(datNME1)
        ax.set_ylabel(datNME2)

        # Apply y-axis limits if provided
        if y_limits is not None:
            ax.set_ylim(y_limits[idx])

        # Add a grid for better readability
        ax.grid(True)

        # Add a color bar to the last plot
        if idx == num_plots - 1:
            fig.colorbar(contour, ax=ax, label='Conditional Probability Density')

    # Hide any unused subplots
    for idx in range(num_plots, 5):
        axs[idx].set_visible(False)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()




def getQuantiles(INdata):
    # Extract Quantiles
    quantiles = [0.01, 0.05, 0.17, 0.5, 0.83, 0.95, 0.99]
    Xp = np.quantile(INdata[:, 0], quantiles)
    Yp = np.quantile(INdata[:, 1], quantiles)

    # Combine quantiles, Xp, and Yp into a single list of tuples
    quantile_data = [(q, x, y) for q, x, y in zip(quantiles, Xp, Yp)]

    return quantile_data



def mark_percentiles(ax, x_percentiles, y_percentiles):
    qq = ['5', '17', '50', '83', '95']
    relativeY = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    relativeX = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    
    # Mark x-axis percentiles
    for i, Xp in enumerate(x_percentiles):
        ax.text(Xp, 0, '|', fontsize=10, color='blue', ha='center', va='top', transform=relativeY)
        if i not in [1, 3]:
            ax.text(Xp, 0.05, f'p{qq[i]}', fontsize=10, color='blue', ha='center', va='top', rotation=45, transform=relativeY)
    
    # Mark y-axis percentiles
    for i, Yp in enumerate(y_percentiles):
        ax.text(0.05, Yp, '--', fontsize=10, color='blue', ha='right', va='center', transform=relativeX)
        if i not in [1, 3]:
            ax.text(0.05, Yp, f'p{qq[i]}', fontsize=10, color='blue', ha='right', va='center', rotation=45, transform=relativeX)


def plot_table(ax, x_percentiles, y_percentiles, xaxLAB, yaxLAB):
    percentiles = ['5', '17', '50', '83', '95']
    
    # Store results in a pandas DataFrame and then transpose it
    table_data = pd.DataFrame({
    f'{"-".join(xaxLAB.split("-")[:3])}\n{"-".join(xaxLAB.split("-")[3:])}': np.round(x_percentiles, 1),
    f'{"-".join(yaxLAB.split("-")[:3])}\n{"-".join(yaxLAB.split("-")[3:])}': np.round(y_percentiles, 1)
}, index=[f'p{p}' for p in percentiles])

    # Create table at the bottom of each subplot
    table = ax.table(cellText=table_data.values, 
                     rowLabels=table_data.index, 
                     colLabels=table_data.columns,
                     cellLoc='center', 
                     loc='bottom',
                     bbox=[0, -0.8, 1, 0.5])  # Adjust bbox for table positioning within plot
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for (i, j), cell in table.get_celld().items():
        if i == 0:  # i == 0 corresponds to the header row
            cell.set_height(0.1)  # Adjust the value to make the header box larger
            cell.set_fontsize(6)
    
    ax.set_adjustable('datalim')
    ax.autoscale()