"""
Main file to run for creating a realistic catalog (similar to the 4FGL) of simulated gamma-ray sources (AGN and pulsars)
based on intensive distribution and correlation analysis.
"""

# LIBRARIES
import argparse
import copy
import pandas as pd
from pathlib import Path

# Relative imports
from analysis.goodness_of_fit import chi_squared_test, kolmogorov_smirnov_test
from analysis.visualisation import *
from read_write_functions import catalog_data_preparation, save_catalog
from source_generation.agn_generation import generate_mock_agn_catalog
from source_generation.pulsar_generation import generate_mock_pulsar_catalog
import time
from verification.visualisation import plot_correlation, plot_luminosity_function, plot_spatial_distribution


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

    agn_results = [["Parameter", "Fitted PDF", "$\\nu$", "$\\Chi^2$", "$\\Chi^2_{\\text{min}}$",
                    "$P(\\Chi^2; \\nu)$", "$\\Chi^2$ Accept $H_0$", "$D_{\\text{crit}}$", "K",
                    "$P(K < D_{\\text{crit}})$", "KS Accept $H_0$"]]

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

    psr_results = [["Parameter", "Fitted PDF", "$\\nu$", "$\\Chi^2$", "$\\Chi^2_{\\text{min}}$",
                    "$P(\\Chi^2; \\nu)$", "$\\Chi^2$ Accept $H_0$", "$D_{\\text{crit}}$", "K",
                    "$P(K < D_{\\text{crit}})$", "KS Accept $H_0$"]]

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


def create_catalog(fermi_catalog: str, data_4fgl: tuple, threshold, verify: bool = False):
    """Creates a catalog of simulated AGN and pulsar sources given the 4FGL data from which to sample realistic
    parameter values and an energy flux from below which the luminosity function must be extrapolated.

    Parameters
    ----------
    fermi_catalog : str
        File in which 4FGL catalog is stored - used to plot luminosity functions of real and simulated data for valida
        tion and verification.

    data_4fgl : tuple
        Relevant AGN and pulsar data extracted from the 4FGL from which to create PDFs of parameters to be sampled from.

    threshold
        Lowest energy flux of any 4FGL source - below this threshold, the luminosity function must be extrapolated to
        generate faint sources.

    verify : bool
        Indicates whether the simulated data should undergo verification - involves plotting and comparison of simulated
        and real data.

    """
    start = time.time()

    # N.B. This makes a single simulated catalog of mock AGN and pulsars

    lat_agn, lat_pulsar = data_4fgl[0], data_4fgl[1]

    # Generate simulated AGN sources

    simulated_agn = generate_mock_agn_catalog(fermi_catalog, lat_agn, detection_threshold=threshold)

    # Generate simulated pulsar sources

    simulated_pulsar = generate_mock_pulsar_catalog(fermi_catalog, lat_pulsar, detection_threshold=threshold)

    end = time.time()

    simulation_time = end - start

    if verify:
        # Verify realism and correctness of generated gamma-ray sources
        verification(simulated_agn, simulated_pulsar, catalog_4fgl=fermi_catalog)

    return simulated_agn, simulated_pulsar, simulation_time


def verification(simulated_agns, simulated_pulsars, catalog_4fgl: str, directory: str = "./plots/verification"):
    """Verify the correlation of simulated parameters matches that of the real parameters and that the luminosity
    function of simulated sources matches that of the real sources.

    Parameters
    ----------

    simulated_agns : ndarray
        2D m x 7 array describing each of the 7 spectral and spatial parameters of m AGNs.
    simulated_pulsars : ndarray
        2D n x 9 array describing each of the 9 spectral and spatial parameters of n pulsars.
    catalog_4fgl : str
        Name of file containing FITS-formatted 4FGL catalog.
    directory:
        Name of file location in which store plots used for verifying simulated datas' realism.

    """
    # Verify simulated data realism and correctness

    # Create directory to store results
    Path(directory).mkdir(parents=True, exist_ok=True)

    # SPECTRAL PARAMETERS

    # Compare the luminosity function of the simulated AGNs with that of those in the 4FGL
    plot_luminosity_function(catalog_4fgl, simulated_agns[:, 4],
                             directory=directory, source_type="AGN")

    # Compare the luminosity function of the simulated pulsars with that of those in the 4FGL
    plot_luminosity_function(catalog_4fgl, simulated_pulsars[:, 5],
                             directory=directory, source_type="Pulsar")

    # SPATIAL DISTRIBUTION

    # Plot simulated AGN spatial distribution
    plot_spatial_distribution(simulated_agns[:, 5], simulated_agns[:, 6], source_type='AGN',
                              directory=directory)

    # Plot simulated pulsar spatial distributions
    plot_spatial_distribution(simulated_pulsars[:, 6], simulated_pulsars[:, 7], source_type='Pulsar',
                              directory=directory)

    # CORRELATION ANALYSIS

    agn_correlated_variables = [("Pivot_Energy", "LP_Flux_Density"), ("Pivot_Energy", "LP_Index")]
    agn_correlated_variables_indices = [(0, 1), (0, 2)]
    num_correlated_agn_variables = len(agn_correlated_variables)

    pulsar_correlated_variables = [("Pivot_Energy", "PLEC_Flux_Density")]
    pulsar_correlated_variables_indices = [(0, 1)]
    num_correlated_psr_variables = len(pulsar_correlated_variables)

    for a in range(num_correlated_agn_variables):
        variable_names = agn_correlated_variables[a]
        indices = agn_correlated_variables_indices[a]

        plot_correlation(catalog_4fgl, var1_name=variable_names[0], var2_name=variable_names[1],
                         simulated_var1=simulated_agns[:, indices[0]], simulated_var2=simulated_agns[:, indices[1]],
                         source_type="AGN", directory=directory)

    for p in range(num_correlated_psr_variables):
        variable_names = pulsar_correlated_variables[p]
        indices = pulsar_correlated_variables_indices[p]

        plot_correlation(catalog_4fgl, var1_name=variable_names[0], var2_name=variable_names[1],
                         simulated_var1=simulated_pulsars[:, indices[0]],
                         simulated_var2=simulated_pulsars[:, indices[1]], source_type="Pulsar", directory=directory)


# MAIN PROGRAM

if __name__ == "__main__":

    # INPUT PARAMETER PARSING

    parser = argparse.ArgumentParser(description="Generates a series of catalogs of simulated gamma-ray sources (AGNs"
                                                 "and pulsars) with spectral and spatial parameter distributions "
                                                 "identical to that of a specified catalog (e.g. 4FGL) and stores them "
                                                 "in a fermitools-compatible XML format.")

    parser.add_argument("--catalog", required=True, type=str, help="File path to catalog in fits format of"
                                                                   " gamma-ray sources with parameter distributions the"
                                                                   " simulated sources should follow.",
                        default="/Volumes/T7/data/catalog/4FGL_DR4.fit")

    parser.add_argument("--number", required=True, type=int, help="The number of catalogs of simulated "
                                                                  "gamma-ray sources to generate.")

    parser.add_argument("--analyse", required=True, choices=["yes", "no"], help="Indicate whether a "
                                                                                "statistical analysis of the input "
                                                                                "catalog's parameters should be "
                                                                                "conducted.", default="no")

    parser.add_argument("--verify", required=True, choices=["yes", "no", "first"], help="Indicate whether"
                                                                                        "the distributions and "
                                                                                        "correlations of the simulated"
                                                                                        "catalogs' parameters should be"
                                                                                        "verified. 'yes' will conduct a"
                                                                                        "verification for each catalog,"
                                                                                        " 'no' will conduct no "
                                                                                        "verification and 'first' will "
                                                                                        "verify the first catalog "
                                                                                        "simulated.", default="no")

    args = parser.parse_args()

    file = args.catalog
    num_catalogs = args.number
    conduct_analysis = args.analyse
    conduct_verification = args.verify

    print("starting catalog simulation...")
    print("READ DATA: ", end='')

    # Read in catalog data
    agn_4fgl, pulsar_4fgl, source_detection_threshold = catalog_data_preparation(file)

    print("DONE")

    print("GENERATE CATALOGS")

    # Run analysis if instructed
    if conduct_analysis == "yes":
        print("ANALYSIS: ", end='')

        # Analyse parameters, their distributions, and their correlations
        analysis(agn_4fgl.copy(), pulsar_4fgl.copy())

        print("DONE")

    total_time = 0

    # Generate the specified number of catalogs of AGN and pulsars
    for c in range(num_catalogs):

        print("Catalog {}: ".format(c + 1), end='')

        if conduct_verification == "yes":
            run_verify = True
        elif conduct_verification == "no":
            run_verify = False
        else:
            if c == 0:
                run_verify = True
            else:
                run_verify = False

        new_agns, new_pulsars, generation_time = create_catalog(fermi_catalog=file, data_4fgl=(agn_4fgl.copy(),
                                                                                               pulsar_4fgl.copy()),
                                                                threshold=source_detection_threshold, verify=run_verify)

        total_time += generation_time

        # Save simulated sources to XML files
        save_catalog(simulated_agns=new_agns, simulated_pulsars=new_pulsars, file_name="./simulated_data/catalogs"
                                                                                       "/catalog_{}/".format(c + 1))

        print("DONE")

    print("catalog simulation finished")

    print("Total Time: {} s".format(total_time))
    print("Average Catalog Simulation Time: {} s".format(total_time / num_catalogs))

# REFERENCES

# Creating Path to Directories - https://stackoverflow.com/questions/273192/how-do-i-create-a-directory-and-any-missing-
# parent-directories
# Energy Flux - https://git.io/JO5FP
# ID8 Paper - Identification of point sources in gamma rays using U-shaped convolutional neural networks and a data
# challenge
# Log-Normal Distribution - https://en.wikipedia.org/wiki/Log-normal_distribution
# Main Functions - https://stackoverflow.com/questions/73378518/how-to-import-and-call-main-functions-from-different-
# files
# Masked to Ordinary Numpy Array - https://www.w3resource.com/python-exercises/numpy/convert-masked-numpy-array-to-regul
# ar-array-with-nan.php
# Radians to Degrees - https://stackoverflow.com/questions/9875964/how-can-i-convert-radians-to-degrees-with-python
# String Formatting - https://stackoverflow.com/questions/12018992/print-combining-strings-and-numbers
# Type Hinting - https://www.reddit.com/r/learnpython/comments/wme6p5/type_hinting_functions_with_multipe_return_types/
# 4FGL Parameter Overview - https://heasarc.gsfc.nasa.gov/W3Browse/fermi/fermilpsc.html
