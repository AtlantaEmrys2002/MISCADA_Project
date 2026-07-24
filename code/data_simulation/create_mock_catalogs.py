# This method is adapted from ID8. All code is my own (except where indicated), but Python
# implementation provided by the authors to *access* (not generate) data can be found in ID8 footnotes. Reasons for
# implementing are as follows:
# 3) I wanted to understand the method so that I could reimplement the code in C/C++ to make use of parallel processing
# and GPUs.
# 4) Once I had reimplemented (possibly in two languages) and made optimisations, I could then improve simulation
# techniques and bring in new ideas, e.g. time data, light curves, etc.
# 5) NOT ALL THE METHODS FOR SIMULATING DATA WERE PROVIDED IN THE ABOVE CODE - MORE ABOUT ACCESSING PRE-GENERATED DATA!
# - CHECK - IT'S ALL ABOUT ACCESSING PREGENERATED DATA - https://git.io/JO5FP - COULD USE TO READ MY XML FILES AND
# GENERATE PATCHES.

# LIBRARIES
import argparse
from pathlib import Path

# Relative imports
from analysis.goodness_of_fit import chi_squared_test, kolmogorov_smirnov_test
from analysis.visualisation import *
from read_write_functions import catalog_data_preparation, save_catalog
from source_generation.agn_generation import generate_mock_agn_catalog
from source_generation.pulsar_generation import generate_mock_pulsar_catalog
import time
from verification.visualisation import plot_correlation, plot_luminosity_function, plot_spatial_distribution


def analysis(agn_rows, pulsar_rows, directory="./plots/analysis"):
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


def create_catalog(fermi_catalog: str, data_4fgl: tuple, threshold, verify: bool = False):
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


def verification(simulated_agns, simulated_pulsars, catalog_4fgl: str, directory="./plots/verification"):
    # Verify simulated data realism and correctness

    # Create directory to store results
    Path(directory).mkdir(parents=True, exist_ok=True)

    # SPECTRAL PARAMETERS

    # # Compare the luminosity function of the simulated AGNs with that of those in the 4FGL
    # plot_luminosity_function("/Volumes/T7/data/catalog/4FGL_DR4.fit", simulated_agns[:, 4],
    #                          directory=directory, source_type="AGN")
    #
    # # Compare the luminosity function of the simulated pulsars with that of those in the 4FGL
    # plot_luminosity_function("/Volumes/T7/data/catalog/4FGL_DR4.fit", simulated_pulsars[:, 5],
    #                          directory=directory, source_type="Pulsar")

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

    ## CORRELATION ANALYSIS

    agn_correlated_variables = [("Pivot_Energy", "LP_Flux_Density"), ("Pivot_Energy", "LP_Index")]
    agn_correlated_variables_indices = [(0, 1), (0, 2)]

    pulsar_correlated_variables = [("Pivot_Energy", "PLEC_Flux_Density")]
    pulsar_correlated_variables_indices = [(0, 1)]

    for a in range(len(agn_correlated_variables)):

        variable_names = agn_correlated_variables[a]
        indices = agn_correlated_variables_indices[a]

        plot_correlation(catalog_4fgl, var1_name=variable_names[0], var2_name=variable_names[1],
                         simulated_var1=simulated_agns[:, indices[0]], simulated_var2=simulated_agns[:, indices[1]],
                         source_type="AGN", directory=directory)

    for p in range(len(pulsar_correlated_variables)):

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
    agn_4fgl, pulsar_4fgl, source_detection_threshold, fluxes_4fgl = catalog_data_preparation(file)

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
# ID8 Paper - Identification of point sources in gamma rays using U-shaped convolutional neural networks and a data
# challenge
# Log-Normal Distribution - https://en.wikipedia.org/wiki/Log-normal_distribution
# Main Functions - https://stackoverflow.com/questions/73378518/how-to-import-and-call-main-functions-from-different-
# files
# Masked to Ordinary Numpy Array - https://www.w3resource.com/python-exercises/numpy/convert-masked-numpy-array-to-regul
# ar-array-with-nan.php
# Pandas Documentation - https://pandas.pydata.org/docs/index.html
# Radians to Degrees - https://stackoverflow.com/questions/9875964/how-can-i-convert-radians-to-degrees-with-python
# String Formatting - https://stackoverflow.com/questions/12018992/print-combining-strings-and-numbers
# Type Hinting - https://www.reddit.com/r/learnpython/comments/wme6p5/type_hinting_functions_with_multipe_return_types/
# 4FGL Parameter Overview - https://heasarc.gsfc.nasa.gov/W3Browse/fermi/fermilpsc.html
