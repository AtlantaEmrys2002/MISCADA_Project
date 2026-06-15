# This method is adapted from ID8. All code is my own (except where indicated), but Python
# implementation provided by the authors to *access* (not generate) data can be found in ID8 footnotes. Reasons for
# implementing are as follows:
# 1) I wanted to find out how to simulate Fermi data and the description in the paper provided a step-by-step method.
# 2) I hoped to improve upon their implementation performance-wise - by implementing from scratch, I am familiar with
# the code and can improve it more easily.
# 3) I wanted to understand the method so that I could reimplement the code in C/C++ to make use of parallel processing
# and GPUs.
# 4) Once I had reimplemented (possibly in two languages) and made optimisations, I could then improve simulation
# techniques and bring in new ideas, e.g. time data, light curves, etc.
# 5) NOT ALL THE METHODS FOR SIMULATING DATA WERE PROVIDED IN THE ABOVE CODE - MORE ABOUT ACCESSING PRE-GENERATED DATA!
# - CHECK - IT'S ALL ABOUT ACCESSING PREGENERATED DATA - https://git.io/JO5FP - COULD USE TO READ MY XML FILES AND
# GENERATE PATCHES.
# 6) IT WILL BE USEFUL - TRAIN MODELS, BENCHMARK DETECTION SCHEMES, HAVE EXACT POSITIONS OF SOURCES SO CAN ALSO
# EVALUATE SENSITIVITY AND LOCALISATION

#TODO
# 1. Luminosity Function - use 3FGL (and cite the paper in notes so can cite in final report) to generate sources
# according to luminosity function - see graph in paper.
# 2. Simulate other source types, e.g. SN, differentiate between AGN types
# 4. DON'T FORGET TO ADD IN DIFFUSE BACKGROUND (GALACTIC AND INTERGALACTIC TO XML FILES)
# 6. Think you include the background files during gtmodel - but check!!!
# 7. CHECK ALL UNITS - ID43

# LIBRARIES
from pathlib import Path
from scipy import stats
from math import floor

# Relative imports
from analysis.goodness_of_fit import chi_squared_test, kolmogorov_smirnov_test
from analysis.visualisation import *
from data_simulation.source_generation.utils import split_normal
from read_write_functions import catalog_data_preparation
from source_generation.agn_generation import generate_mock_agn_catalog
from source_generation.pulsar_generation import generate_mock_pulsar_catalog
from verification.visualisation import plot_agn_luminosity_function, plot_spatial_distribution

# VISUALISATIONS


def normal_func(x, mean, sigma):

    var = sigma**2

    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(((x - mean) ** 2) / (2 * var)))


def visualising_pulsar_latitude_distributions(sigma_1, sigma_2, x_values, counts):

    # sigma_1 and sigma_2 are the fitted standard distributions of the two overlapping Gaussians of the 4FGL data
    # x_values and counts are the binned 4FGL data (latitude values and number of sources within that latitude range

    plt.bar(x_values, counts, alpha=0.7, label='4FGL Distribution')

    # Rounded standard deviations - not used except in plots
    std_1, std_2 = str(round(sigma_1, 2)), str(round(sigma_2, 2))

    latitude_range = np.linspace(-30, 30, 1000)

    plt.plot(latitude_range, split_normal(latitude_range, sigma_1=sigma_1, sigma_2=sigma_2), color='red',
             label='Mixture Gaussian $\mu=0$, \n $\sigma_1=$' + std_1 + "$, \sigma_2 = $" + std_2)

    plt.plot(latitude_range, normal_func(latitude_range, mean=0, sigma=sigma_1), color='green',
             linestyle='--', label='Gaussian: $\mu = 0$, $\sigma_1 = ' + std_1 + '$')

    plt.plot(latitude_range, normal_func(latitude_range, mean=0, sigma=sigma_2), color='orange',
             linestyle='-.', label='Gaussian: $\mu = 0$, $\sigma_2 = ' + std_2 + '$')

    plt.plot(latitude_range, split_normal(latitude_range, sigma_1=1.39, sigma_2=19.2), color='purple',
             label='Recommended by ID8')

    # Randomly sample pulsar latitudes from distribution created above
    X1 = stats.Normal(mu=0, sigma=sigma_1)
    X2 = stats.Normal(mu=0, sigma=sigma_2)

    # CHANGE WEIGHTS HERE TO REFLECT MSP VS YNG

    mixture = stats.Mixture([X1, X2])

    samples = mixture.sample(shape=(10000, 1))

    plt.hist(samples.flatten(), density=True, bins=40, label='Randomly Generated', color='pink', alpha=0.6)

    # Formatting
    plt.title('Distribution of Pulsar Latitudes')
    plt.xlabel('Latitude ($\degree$)')
    plt.ylabel('Source Density')
    plt.legend()

    plt.show()


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

    prob_dist = ['normal', 'lognorm']

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


def verification(simulated_agns, simulated_pulsars, directory="./plots/verification"):

    # Verify simulated data realism

    # Create directory to store results
    Path(directory).mkdir(parents=True, exist_ok=True)

    # SPECTRAL PARAMETERS

    # Compare the luminosity function of the simulated AGNs with that of those in the 4FGL
    plot_agn_luminosity_function("/Volumes/T7/data/catalog/4FGL_DR4.fit", simulated_agns[:, 4],
                                 directory=directory)

    # SPATIAL DISTRIBUTION

    # Plot simulated AGN spatial distribution
    plot_spatial_distribution(simulated_agns[:, 5], simulated_agns[:, 6], source_type='AGN',
                              directory=directory)

    # Plot simulated pulsar spatial distributions
    plot_spatial_distribution(simulated_pulsars[:, 6], simulated_pulsars[:, 7], source_type='Pulsar',
                              directory=directory)


# MAIN PROGRAM

print("starting catalog simulation...")

# Read in catalog data

file = "/Volumes/T7/data/catalog/4FGL_DR4.fit"

agn_rows, pulsar_rows, source_detection_threshold, fluxes_4fgl = catalog_data_preparation(file)

# Analyse parameters, their distributions, and their correlations

# analysis(agn_rows, pulsar_rows)

# Generate simulated AGN sources

agns = generate_mock_agn_catalog(file, agn_rows.copy(), num_agns=1000, detection_threshold=source_detection_threshold)

# Generate simulated pulsar sources

pulsars = generate_mock_pulsar_catalog(pulsar_rows.copy(), num_pulsars=200)

# Verify realism and correctness of generated gamma-ray sources
verification(agns, pulsars)

print("catalog simulation finished")

# REFERENCES

# Astropy Documentation - https://docs.astropy.org/en/stable/
# ID8 Paper - Identification of point sources in gamma rays using U-shaped convolutional neural networks and a data
# challenge
# Log-Normal Distribution - https://en.wikipedia.org/wiki/Log-normal_distribution
# Masked to Ordinary Numpy Array - https://www.w3resource.com/python-exercises/numpy/convert-masked-numpy-array-to-regul
# ar-array-with-nan.php
# Numpy Documentation - https://numpy.org/doc/stable/user/index.html
# Pandas Documentation - https://pandas.pydata.org/docs/index.html
# Python Documentation - https://docs.python.org/3/
# Scipy Documentation - https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html#scipy.integrate.
# quad
# String Formatting - https://stackoverflow.com/questions/12018992/print-combining-strings-and-numbers

# N.B. Useful conversion:
# flux_densities = agns['LP_Flux_Density'].to(u.ph / (u.cm * u.cm * u.GeV * u.s)).value.filled(np.nan)
