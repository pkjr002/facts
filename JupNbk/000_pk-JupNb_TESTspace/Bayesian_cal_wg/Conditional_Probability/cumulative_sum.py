# Compute cumulative sums along each column
        cumulative_sums = np.cumsum(PLOT_VAR, axis=0)

        # Define the percentiles you want to trace
        percentiles = [17, 83]

        # Find the contour levels corresponding to these percentiles
        contour_levels = []
        for percentile in percentiles:
            # Convert percentile to a decimal fraction
            threshold = percentile / 100.0
            
            # For each column, find the first index where the cumulative sum exceeds the threshold
            # and get the corresponding value from PLOT_VAR
            levels = []
            for col in range(cumulative_sums.shape[1]):
                idx = np.argmax(cumulative_sums[:, col] >= threshold)
                levels.append(PLOT_VAR[idx, col])
            
            # Average the levels across all columns to get a single level for the contour
            contour_levels.append(np.mean(levels))
        #
        contour = ax.contourf(Xgrid, Ygrid, PLOT_VAR, levels=contour_levels, cmap='viridis')