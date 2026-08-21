'''
This function is for running a separate analysis of 4FGL parameters without generating catalogs.
'''
import pandas as pd

from analysis.goodness_of_fit import chi_squared_test, kolmogorov_smirnov_test
from analysis.visualisation import *
import copy
import numpy as np
from pathlib import Path
from read_write_functions import catalog_data_preparation


def analysis(agn_rows, pulsar_rows, signif=0.01, directory: str = "./plots/analysis"):
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

    prob_dist = ['normal', 'lognorm', 'cauchy', 'gumbel']

    # AGNs

    print("AGNs")

    agn_results = []

    agn_results.append(["Parameter", "Fitted PDF", "$\\nu$", "$\\Chi^2$", "$\\Chi^2_{\\text{min}}$",
                        "$P(\\Chi^2; \\nu)$", "$\\Chi^2$ Accept $H_0$", "$D_{\\text{crit}}$", "K",
                        "$P(K < D_{\\text{crit}})$", "KS Accept $H_0$"])

    # Fit probability distributions and determine goodness-of-fit for each parameter
    for parameter in agns:

        # Remove NaN values
        mask = ~np.isnan(agns[parameter])
        values = agns[parameter][mask]

        # Iterate over candidate distributions and fit each to parameters
        for dist in prob_dist:
            # Perform chi_squared goodness of fit test
            results_chi2 = chi_squared_test(copy.deepcopy(values), num_bins=60, distribution=dist, significance=signif)

            # Perform K-S goodness of fit test
            results_ks = kolmogorov_smirnov_test(values=copy.deepcopy(values), distribution=dist, alpha=signif)

            agn_results.append(tuple([values.name, dist] + results_chi2 + results_ks))

    # Plot AGN parameter distributions
    plot_parameter_distributions(agns, source_type='AGN', directory=directory + "/parameter_distributions")

    # Pulsars

    print("Pulsars")

    psr_results = []

    psr_results.append(["Parameter", "Fitted PDF", "$\\nu$", "$\\Chi^2$", "$\\Chi^2_{\\text{min}}$",
                        "$P(\\Chi^2; \\nu)$", "$\\Chi^2$ Accept $H_0$", "$D_{\\text{crit}}$", "K",
                        "$P(K < D_{\\text{crit}})$", "KS Accept $H_0$"])

    # Fit probability distributions and determine goodness-of-fit for each parameter
    for parameter in pulsars:

        mask = ~np.isnan(pulsars[parameter])
        values = pulsars[parameter][mask]

        # Iterate over candidate distributions and fit each to parameters
        for dist in prob_dist:
            # Perform chi_squared goodness of fit test
            # chi_squared_test(values, num_bins=100, distribution=dist)
            results_chi2 = chi_squared_test(copy.deepcopy(values), num_bins=30, distribution=dist, significance=signif)

            # Perform K-S goodness of fit test
            results_ks = kolmogorov_smirnov_test(values=copy.deepcopy(values), distribution=dist, alpha=signif)

            psr_results.append(tuple([values.name, dist] + results_chi2 + results_ks))

    # Plot pulsar parameter distributions
    plot_parameter_distributions(pulsars, source_type='Pulsars', directory=directory + "/parameter_distributions")

    # Save Results

    df_agn = pd.DataFrame(agn_results[1:], columns=agn_results[0])

    df_psr = pd.DataFrame(psr_results[1:], columns=psr_results[0])

    Path("./../results/analysis_results").mkdir(parents=True, exist_ok=True)

    df_agn.to_csv("./../results/analysis_results/agn_parameter_analysis.csv", index=False)
    df_psr.to_csv("./../results/analysis_results/psr_parameter_analysis.csv", index=False)

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


agn_4fgl, pulsar_4fgl, source_detection_threshold = catalog_data_preparation("/Volumes/T7/data/catalog/4FGL_DR4.fit")

print("Number of AGN: {}".format(len(agn_4fgl)))
print("Number of Pulsars: {}".format(len(pulsar_4fgl)))

analysis(agn_4fgl.copy(), pulsar_4fgl.copy())


# REFERENCES
# Creating Pandas Frame - https://www.geeksforgeeks.org/python/create-a-pandas-dataframe-from-lists/
# K-S Test Mean and Sigma - https://medium.com/@pabaldonedo/kolmogorov-smirnov-test-may-not-be-doing-what-you-think-when
# -parameters-are-estimated-from-the-data-2d5c3303a020
