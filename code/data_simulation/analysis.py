'''
This function is for running a separate analysis of 4FGL parameters without generating catalogs.
'''

def analysis(agn_rows, pulsar_rows, directory: str = "./plots/analysis"):
    """Conducts a full analysis of all spectral (and one spatial) parameters for chosen gamma-ray sources. Included in
    this analysis is the fitting of a PDF to the parameters of all sources of a given type within the 4FGL, as well as
    a correlation analysis of these parameters.

    Parameters
    ----------
    agn_rows : ndarray
        2D m x 7 array of spectral and spatial parameters of m AGN sources.
    pulsar_rows : ndarray
        2D n x 7 array of spectral and spatial parameters of n AGN sources.
    directory : str
        Directory in which to store plots produced during analysis.

    """
    # Create directory to store results
    Path(directory + "/parameter_distributions").mkdir(parents=True, exist_ok=True)
    Path(directory + "/parameter_correlations").mkdir(parents=True, exist_ok=True)

    # PREPARE DATA

    # Convert to pandas dataframes for covariance and correlation calculations, as well as plotting
    agns = agn_rows.to_pandas()
    pulsars = pulsar_rows.to_pandas()

    # Remove sources with NaN values
    pulsars.dropna(inplace=True)
    agns.dropna(inplace=True)

    # PARAMETER DISTRIBUTIONS

    print("PARAMETER DISTRIBUTION ANALYSIS")
    print('-' * 60)

    prob_dist = ['normal', 'lognorm', 'cauchy']

    # AGNs

    print("AGNs")

    # Fit probability distributions and determine goodness-of-fit for each parameter
    for parameter in agns:

        # Remove NaN values
        mask = ~np.isnan(agns[parameter])
        values = agns[parameter][mask]

        # Iterate over candidate distributions and fit each to parameters
        for dist in prob_dist:
            # Perform chi_squared goodness of fit test
            chi_squared_test(values, num_bins=100, distribution=dist)

            # Perform K-S goodness of fit test
            kolmogorov_smirnov_test(values=values, distribution=dist)

    # Plot AGN parameter distributions
    plot_parameter_distributions(agns, source_type='AGN', directory=directory + "/parameter_distributions")

    # Pulsars

    print("Pulsars")

    # Fit probability distributions and determine goodness-of-fit for each parameter
    for parameter in pulsars:

        mask = ~np.isnan(pulsars[parameter])
        values = pulsars[parameter][mask]

        # Iterate over candidate distributions and fit each to parameters
        for dist in prob_dist:
            # Perform chi_squared goodness of fit test
            chi_squared_test(values, num_bins=100, distribution=dist)

            # Perform K-S goodness of fit test
            kolmogorov_smirnov_test(values=values, distribution=dist)

    # Plot pulsar parameter distributions
    plot_parameter_distributions(pulsars, source_type='Pulsars', directory=directory + "/parameter_distributions")

    # CORRELATION ANALYSIS

    print("PARAMETER CORRELATION ANALYSIS")
    print('-' * 60)

    # AGNs

    print("AGNS")

    print('-' * 60)

    # Plot parameters against one another to visualise relationships
    plot_parameter_relationships(agns, source_type="AGN", directory=directory + "/parameter_correlations")

    # Plot AGN parameter correlation matrices
    plot_correlation_matrices(agns, source_type="AGN", directory=directory + "/parameter_correlations")

    # Fit relationships to identified correlated variables - pivot energy and spectral slope (alpha)
    plot_fitting_correlated_variable_dependency(agns["Pivot_Energy"], agns["LP_Index"], source_type="AGN",
                                                directory=directory + "/parameter_correlations")

    print('-' * 60)

    # Fit relationships to identified correlated variables - pivot energy and flux_density
    plot_fitting_correlated_variable_dependency(agns["Pivot_Energy"], agns["LP_Flux_Density"], logarithmic_fit=False,
                                                source_type="AGN", directory=directory + "/parameter_correlations")

    print('-' * 60)

    # Pulsars

    print("Pulsars")

    print('-' * 60)

    # Plot parameters against one another to visualise relationships
    plot_parameter_relationships(pulsars, source_type="Pulsar", directory=directory + "/parameter_correlations")

    # Plot pulsar parameter correlation matrices
    plot_correlation_matrices(pulsars, source_type="Pulsar", directory=directory + "/parameter_correlations")

    # Fit relationships to identified correlated variables - pivot energy and flux_density
    plot_fitting_correlated_variable_dependency(pulsars["Pivot_Energy"], pulsars["PLEC_Flux_Density"],
                                                source_type="Pulsar", directory=directory + "/parameter_correlations")


