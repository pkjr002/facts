# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
# Fun: Conditional Probability Plotting.
#.............................................................
def plot_ConditionalProb_panel(all_ssp_data,plot_params,plotOPT):
    #
    panel_no=len(plot_params)
    if panel_no == 4:     fig, ax = plt.subplots(1, 4, figsize=(20, 4)); 
    elif panel_no == 5:   fig, ax = plt.subplots(1, 5, figsize=(25, 4)); 
    else :                raise ValueError(f"Unexpected number of panels: {panel_no}. Expected 4 or 5.")
    #
    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    #
    # Loop through the dictionary and plot
    for i, params in plot_params.items():
        # AXIS-LABELS
        var1_lab = next(key for key, val in all_ssp_data.items() if np.array_equal(val, params['var1']))
        var2_lab = next(key for key, val in all_ssp_data.items() if np.array_equal(val, params['var2']))
        plotOPT['YaxLab_disp'] = 'YES' if i == 0 else 'NO' 
        #
        if plotOPT['plotCBAR'] == 'YES':
            plot_ConditionalProb(ax[i], params['var1'], params['var2'], params['t1'], params['t2'],var1_lab,var2_lab,plotOPT)
        if plotOPT['plotCBAR'] == 'YES_1':
            showCBAR = 1 if i == 4 or 5 else 0
            plotOPT['showCBAR'] = showCBAR
            plotOPT['cbar_ax'] = fig.add_axes([0.92, 0.15, 0.01, 0.7])  # controls the shape [left, bottom, width, height]
            plot_var1=plot_ConditionalProb(ax[i], params['var1'], params['var2'], params['t1'], params['t2'],var1_lab,var2_lab,plotOPT)      
    plt.show()        
    return plot_var1
#.............................................................
def plot_ConditionalProb(ax, var1, var2, t1, t2,var1_lab,var2_lab,plotOPT):
    #-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    # Unpack Plot Options
    ssp = plotOPT['ssps']
    kernel = plotOPT['kernel']; 
    bw_kde = plotOPT['bw_kde']; 
    kde_grid_int = plotOPT['kde_grid_int']
    val=plotOPT['val']
    plt_og = plotOPT['plt_og']; 
    plt_scatter = plotOPT['plt_scatter']; 
    cmap = plotOPT['cmap'] 
    #-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    # Define AXIS variables
    var1=var1[ssp]
    var2=var2[ssp]
    # Extract slc for a specified time slice.
    xaxVAR = var1['slc'][:, np.where(var1['time']==t1)[0][0]] 
    yaxVAR = var2['slc'][:, np.where(var1['time']==t2)[0][0]] 
    # LABELS
    xaxLAB = f'{var1_lab}_{t1} (cm)'     
    yaxLAB = f'{var2_lab}_{t2} (cm)'  
    title  = f'{t2} {var2_lab}  \n conditional upon \n {t1} {var1_lab} '
    #-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    # PLOT Conditional Probability figure.
    plot_var = gilford(ax, xaxVAR, yaxVAR, kernel, bw_kde, kde_grid_int, val, xaxLAB, yaxLAB, title, ssp, plt_og,plt_scatter, cmap, t1, plotOPT)
    return plot_var


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
# Plot contour that sums to 1 along columns.
#.............................................................
def gilford(ax, xaxVAR, yaxVAR,kernel,bw_kde,kde_grid_int, val, xaxLAB,yaxLAB,title,ssp,plt_og,plt_scatter, CMAP,T1, plotOPT=None):
    #-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    # Extract Quantiles
    quantiles = [0.01, 0.05, 0.17, 0.5, 0.83, 0.95, 0.99]
    Xp01_, Xp05_, Xp17_, Xp50_, Xp83_, Xp95_, Xp99_ = np.quantile(xaxVAR, quantiles)
    Yp01_, Yp05_, Yp17_, Yp50_, Yp83_, Yp95_, Yp99_ = np.quantile(yaxVAR, quantiles)
    #-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    # Create 2D matrix.
    INdata = np.column_stack((xaxVAR, yaxVAR))
    #
    if bw_kde == 'auto':
        data = INdata
        #print("Data shape:", data.shape)
        # Grid of bandwidth values to test
        bandwidths = np.linspace(0.1, 10, 40)  
        # Setup the grid search with cross-validation
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                           {'bandwidth': bandwidths},
                           cv=5)  # 5-fold cross-validation
        # Fit grid search
        grid.fit(data)
        # Best bandwidth
        best_bw = np.round(grid.best_params_['bandwidth'],2)
        print("Optimal bandwidth:", best_bw)
        bw_kde=best_bw
    else:
        bw_kde=bw_kde
    #.............................................................................................................
    # KDE
    kde = KernelDensity(kernel=kernel, bandwidth=bw_kde).fit(INdata)
    #-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    # Create a grid of test points
    if plotOPT['grid'] == 'data_MaxMin':
        # simple.
        xgrid = np.linspace(INdata[:,0].min()-1, INdata[:,0].max()+1, kde_grid_int)  
        ygrid = np.linspace(INdata[:,1].min()-1, INdata[:,1].max()+1, kde_grid_int)   
    #--> below not recomended as it can dilute the plot.
    elif plotOPT['grid'] == 'Fixed':
        xgrid = np.linspace(-20, 100, kde_grid_int)  
        ygrid = np.linspace(-20, 100, kde_grid_int)  
    #
    # %age
    # bandwidth_extension_factor = bw_kde * 4  #2 is eg
    # # Create a grid of test points, extended by a function of the bandwidth
    # xgrid = np.linspace(INdata[:,0].min() - bandwidth_extension_factor, 
    #                 INdata[:,0].max() + bandwidth_extension_factor, kde_grid_int)
    # ygrid = np.linspace(INdata[:,1].min() - bandwidth_extension_factor, 
    #                 INdata[:,1].max() + bandwidth_extension_factor, kde_grid_int)
    #-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
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
    normalized_density_values = density_values / np.sum(density_values, axis=0);
    #.............................................................................................................
    ## Normalize log_density_values |.| Do all of this in log scales (Log-Sum-Exp trick)
    #adjusted_log_density_values = log_density_values - np.max(log_density_values)
    #exp_sum = np.sum(np.exp(adjusted_log_density_values), axis=0)
    ## Readjust by adding back the adjustment factor
    #log_normalization_constant = np.log(exp_sum) + np.max(log_density_values);
    ## Normalize within the log space 
    #normalized_log_density_values = log_density_values - log_normalization_constant

    #-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    if val == 'log_density_values':
        PLOT_VAR=log_density_values
    #elif val == 'log_density_values_Normalized':
    #    PLOT_VAR=normalized_log_density_values
    elif val == 'density_values':
        PLOT_VAR=density_values
    elif val == 'density_values_Normalized':
        PLOT_VAR=normalized_density_values
    #
    #-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    # PLOT:: Choose PLOT_VAR and fill contour and COLORbar limits
    if val in ['density_values' , 'density_values_Normalized', 'log_density_values', 'log_density_values_Normalized']:
        if val in ['density_values' , 'density_values_Normalized']:
            if plotOPT is not None and 'c_bar_min' in plotOPT:
                clevels=np.linspace(plotOPT['c_bar_min'],plotOPT['c_bar_max'],plotOPT['c_bar_int'])    
            else: clevels=np.linspace(1e-3,PLOT_VAR.max(),10)
        else: clevels=np.linspace(PLOT_VAR.min(),PLOT_VAR.max(),10)
        # PLOT:: contour
        clabels=np.round(clevels,decimals=3).astype('str')
        contour=ax.contourf(Xgrid, Ygrid, PLOT_VAR,levels=clevels,cmap=CMAP)
        # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&    
        print(PLOT_VAR)
        # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # # PRINT<==
        # # Calculate the intervals consecutive x/y values
        #dx = np.diff(xgrid)  ; print("X-axis interval:", dx[0])
        #dy = np.diff(ygrid)  ; print("Y-axis interval:", dy[0])
        #print("----")
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # COLORBAR::
        if plotOPT['plotCBAR'] is not None:
            if plotOPT['plotCBAR'] == 'YES':
                cbar=plt.colorbar(contour,ax=ax,label=val,ticks=clevels,orientation='horizontal',pad=0.2)
                cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=45)  
            if plotOPT['plotCBAR'] == 'YES_1' and plotOPT['showCBAR'] == 1:
                cbar=plt.colorbar(contour,cax=plotOPT['cbar_ax'],label=val,ticks=clevels,orientation='vertical',pad=0.01)    
            cbar.set_label(label=val, size=10, weight='bold', color='blue')
            cbar.set_ticklabels(clabels)
            cbar.ax.tick_params(labelsize=8)
            # cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=45)    
        # ax.set_xlim(np.floor(np.min(INdata[:, 0])), np.ceil(np.max(INdata[:, 0])))
        # ax.set_ylim(np.floor(np.min(INdata[:, 1])), np.ceil(np.max(INdata[:, 1])))    
    #-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    # PLOT:: SNS Kde  
    if plotOPT['sns_kde'] == 'YES':
        sns.kdeplot(x=INdata[:, 0], y=INdata[:, 1], fill=True, thresh=0, levels=100, cmap="mako", ax=ax)

    #-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    # PLOT:: RAW original data  
    if plt_og == 'YES':
        numbers = np.linspace(1, 2000, 2000)    
        x_min, x_max = np.floor(np.min(numbers)), np.ceil(np.max(numbers)) 
        #
        color = 'tab:blue'
        ax.set_xlabel('')
        ax.set_ylabel('', color=color)
        ax.plot(numbers,INdata[:, 0], 'o', linestyle='none', color=color, label=xaxLAB)
        ax.tick_params(axis='y', labelcolor=color)
        #
        ax2 = ax.twinx()  
        color = 'tab:orange'
        ax2.set_ylabel('', color=color)
        ax2.plot(numbers,INdata[:, 1], 'x', linestyle='none', color=color, label=yaxLAB)
        ax2.tick_params(axis='y', labelcolor=color)
        #
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, loc='upper left') 
        ax.legend(handles1 + handles2, labels1 + labels2, loc='upper left', bbox_to_anchor=(.275, 1.35), title='')

        if plotOPT['fix_ax_lim'] == 'NO':
            ax.set_xlim(x_min,x_max)
            ax.set_ylim(np.floor(np.min(INdata[:, 0])), np.ceil(np.max(INdata[:, 0])))
            ax2.set_ylim(np.floor(np.min(INdata[:, 1])), np.ceil(np.max(INdata[:, 1])))
        elif plotOPT['fix_ax_lim'] == 'YES': 
            # ax.set_ylim(np.floor(np.min(INdata[:, 1])), np.ceil(np.max(INdata[:, 1])))
            # ax2.set_ylim(np.floor(np.min(INdata[:, 1])), np.ceil(np.max(INdata[:, 1])))
            ax.set_ylim(-10, np.ceil(np.max(INdata[:, 1])))
            ax2.set_ylim(-10, np.ceil(np.max(INdata[:, 1])))
        xaxLAB = 'samples'; ax.set_xlabel(xaxLAB)
        yaxLAB = 'projected sl (cm) ';
        if plotOPT['YaxLab_disp'] == 'YES':     ax.set_ylabel(yaxLAB)
        title = 'raw data'
    #-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    # PLOT:: SCATTER of the original data
    if plt_scatter == 'YES':
        ax.scatter(INdata[:, 0], INdata[:, 1], s=.5, facecolor='red')
        ax.set_xlim(np.floor(np.min(INdata[:, 0])), np.ceil(np.max(INdata[:, 0])))
        ax.set_ylim(np.floor(np.min(INdata[:, 1])), np.ceil(np.max(INdata[:, 1])))
        #if plotOPT['mark_ptile'] == 'YES':

    #-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    # PLOT:: voilin/boxWhisk
    if plotOPT is not None and plotOPT['plt_overlay'] in ['violin', 'box']:
        num_violins = plotOPT['num_violins']
        x_min, x_max = np.floor(np.min(INdata[:, 0])), np.ceil(np.max(INdata[:, 0]))
        bin_edges = np.linspace(x_min, x_max, num=num_violins+1, endpoint=True)
        x_binned = np.digitize(INdata[:, 0], bin_edges, right=True)

        # Group data by bins
        binned_data = [INdata[:, 1][x_binned == i] for i in range(1, len(bin_edges))]

        positions = np.arange(1, num_violins + 1) * (x_max - x_min) / num_violins - ((x_max - x_min) / (2 * num_violins)) + x_min

        if plotOPT['plt_overlay'] == 'violin':
            ax.violinplot(binned_data, positions=positions, widths=(x_max - x_min) / num_violins * 0.8, showmeans=False, showextrema=True, showmedians=True)
        elif plotOPT['plt_overlay'] == 'box':
            for i, data in enumerate(binned_data):
                ax.boxplot(data, positions=[positions[i]], widths=(x_max - x_min) / num_violins * 0.8, vert=True, patch_artist=True)
        #
        ax.set_xlim(x_min,x_max)
        if plotOPT is not None and 'y_ax_min' in plotOPT:
            ax.set_ylim(plotOPT['y_ax_min'],plotOPT['y_ax_max'])
    #-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    # AXIS properties
    ax.set_title(title,fontsize=8)
    if plotOPT['YaxLab_disp'] == 'YES':     ax.set_ylabel(yaxLAB)
    ax.set_xlabel(xaxLAB)
    #
    ax.text(0.9, 0.1, f'{ssp}', fontsize=7, color='black', weight='bold', ha='center', va='center', transform=ax.transAxes)
    #
    # ax.set_xlim(Xp01_,Xp99_)
    #
    # ................................................................................................
    if plotOPT['mark_ax_ptile'] == 'YES':
        qq=['5','17','50','83','95']
        relativeY = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        # ax.text(Xp50_, 0, '|', fontsize=7, ha='center', va='top', transform=relativeY)
        for i,Xp in enumerate([Xp05_,Xp17_, Xp50_,Xp83_,Xp95_]):
            ax.text(Xp, 0, '|', fontsize=15, color='blue', ha='center', va='top', transform=relativeY)
            if i not in [1,3]:
                ax.text(Xp, 0.05, f'p{qq[i]}' , fontsize=10, color='blue', ha='center', va='top', transform=relativeY)
        #
        relativeX = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        # ax.text(0, Yp50_,'-', fontsize=14, ha='right', va='center', transform=relativeX)  #'\u2014'
        for i,Yp in enumerate([Yp05_,Yp17_, Yp50_,Yp83_,Yp95_]):
            ax.text(0.05, Yp, '--', fontsize=14, color='blue', ha='right', va='center', transform=relativeX)
            if i not in [1,3]:  
                ax.text(0.05, Yp, f'p{qq[i]}', fontsize=10, color='blue', ha='right', va='center', transform=relativeX)
    # ................................................................................................
    # Adjust tick settings to ensure correct display
    ax.tick_params(axis='y', which='both', labelleft=True, labelright=False)
    ax.tick_params(axis='x', which='both', top=False, bottom=True, labeltop=False)


    if plotOPT['ptile_table'] == 'YES':
        percentiles = ['5','17','50','83','95']

        # Store results in a pandas DataFrame and then transpose it
        table_data = pd.DataFrame({
            f'{xaxLAB}':np.round([Xp05_,Xp17_, Xp50_,Xp83_,Xp95_],1),
            f'{yaxLAB}': np.round([Yp05_,Yp17_, Yp50_,Yp83_,Yp95_],1)
        }, index=[f'p{p}' for p in percentiles])



        # Create table at bottom of each subplot
        table = ax.table(cellText=table_data.values, 
                         rowLabels=table_data.index, 
                         colLabels=table_data.columns,
                         cellLoc='center', 
                         loc='bottom',
                         bbox=[0, -0.5, 1, 0.3])  # Adjust bbox for table positioning within plot
        ax.set_adjustable('datalim')
        ax.autoscale()

    return PLOT_VAR
# ^^cx^
