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
import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.metrics import root_mean_squared_error

# Relative imports
from analysis.goodness_of_fit import chi_squared_test, kolmogorov_smirnov_test
from analysis.visualisation import (plot_correlation_matrices, plot_parameter_distributions,
                                    plot_parameter_relationships)
from verification.visualisation import plot_agn_luminosity_function, plot_spatial_distribution
from read_write_functions import catalog_data_preparation
from source_generation.pulsar_generation import generate_mock_pulsar_catalog, pulsar_statistics
from data_simulation.source_generation.utils import split_normal

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


print("starting...")


def fitting_agn_pivot_energy_flux_density_relation(agns):

    # IMPORTANT
    # N.B. Could come back to and take into account ERROR on each observation of pivot energy etc. (using uncertainty
    # for both E_0 and F_0)

    # Format data

    # Remove pulsar columns
    agns.remove_columns(['PLEC_Flux_Density', 'PLEC_IndexS', 'PLEC_Exp_Index', 'PLEC_ExpfactorS'])

    # Remove spatial column for AGNS - AGNs are known to be approximately isotropically distributed on the sky
    agns.remove_column('GLAT')

    # Convert units

    # Select pivot energy values and convert from MeV to GeV

    agns['Pivot_Energy'] = agns['Pivot_Energy'].to(u.GeV)
    agns['LP_Flux_Density'] = agns['LP_Flux_Density'].to(u.ph / (u.cm * u.cm * u.GeV * u.s))

    agns = agns.to_pandas()

    # Remove sources with NaN values
    agns.dropna(inplace=True)

    # Create plot

    plt.rcParams["figure.figsize"] = (8, 4)
    fig, ax = plt.subplots(1, 2)

    # Plot raw data

    log_pivot_energy = np.log(agns['Pivot_Energy'])
    log_flux_density = np.log(agns['LP_Flux_Density'])

    ax[0].scatter(agns['Pivot_Energy'], agns['LP_Flux_Density'], s=8, marker='+')
    ax[1].scatter(log_pivot_energy, log_flux_density, s=8, marker='+')

    # Fit relationships between AGN flux density and pivot energy

    x_values = np.linspace(np.min(agns['Pivot_Energy']), np.max(agns['Pivot_Energy']), 1000)
    log_x_values = np.linspace(np.min(log_pivot_energy), np.max(log_pivot_energy), 1000)

    # Straight line
    m, c = np.polyfit(log_pivot_energy, log_flux_density, deg=1)
    ax[1].plot(log_x_values, m * log_x_values + c, color='red', label='Linear')

    # non-logarithmic fit
    ax[0].plot(x_values, (x_values ** m) * (np.e ** c), label='Log Linear', color='red')

    # Normalised/Standardised/Studentised Residual https://stats.stackexchange.com/questions/22653/raw-residuals-versus-standardised-residuals-versus-studentised-residuals-what
    residuals_straight = log_flux_density - (m * log_pivot_energy + c)

    print("RMSE of Linear Fit: {}".format(root_mean_squared_error(log_flux_density, m * log_pivot_energy + c)))
    print('m: {} c: {}'.format(m, c))

    # Quadratic
    # x_values = np.linspace(np.min(log_pivot_energy), np.max(log_pivot_energy), 1000)
    a, b, c = np.polyfit(log_pivot_energy, log_flux_density, deg=2)
    ax[1].plot(np.log(x_values), (a * np.log(x_values) ** 2) + (b * np.log(x_values)) + c, color='green', label='Quadratic', linestyle='-.')

    # non-logarithmic fit
    ax[0].plot(np.e ** log_x_values, np.e ** ((a * log_x_values ** 2) + (b * log_x_values) + c), color='green', label='Log Quadratic', linestyle='-.')

    residuals_quad = log_flux_density - ((a * log_pivot_energy ** 2) + (b * log_pivot_energy) + c)

    print("RMSE of Quadratic Fit: {}".format(root_mean_squared_error(log_flux_density, (a * log_pivot_energy ** 2)
                                                                     + (b * log_pivot_energy) + c)))
    print('a: {} b: {} c: {}'.format(a, b, c))

    print("No. Real Solutions: ".format((b ** 2) - (4 * a * c)))

    # Cubic
    # x_values = np.linspace(np.min(log_pivot_energy), np.max(log_pivot_energy), 1000)
    a, b, c, d = np.polyfit(log_pivot_energy, log_flux_density, deg=3)
    ax[1].plot(log_x_values, (a * log_x_values ** 3) + (b * log_x_values ** 2) + (c * log_x_values) + d, color='orange',
               label='Cubic', linestyle='--')

    # non-logarithmic fit
    ax[0].plot(np.e ** np.log(x_values), np.e ** ((a * np.log(x_values) ** 3) + (b * np.log(x_values) ** 2) + (c * np.log(x_values)) + d), linestyle='--', label='Log Cubic', color='orange')

    residuals_cubic = log_flux_density - ((a * log_pivot_energy ** 3) + (b * log_pivot_energy ** 2) + (c * log_pivot_energy)
                                    + d)

    print("RMSE of Cubic Fit: {}".format(root_mean_squared_error(log_flux_density, (a * log_pivot_energy ** 3) + (b * log_pivot_energy ** 2) + (c * log_pivot_energy)
                                    + d)))
    print('a: {} b: {} c: {} d: {}'.format(a, b, c, d))

    # n^4

    # x_values = np.linspace(np.min(log_pivot_energy), np.max(log_pivot_energy), 1000)
    a, b, c, d, e = np.polyfit(log_pivot_energy, log_flux_density, deg=4)
    ax[1].plot(log_x_values, (a * log_x_values ** 4) + (b * log_x_values ** 3) + (c * log_x_values ** 2) +
               (d * log_x_values) + e, color='purple', label='$n^4$', linestyle=':')

    # non-logarithmic fit
    ax[0].plot(np.e ** np.log(x_values), np.e ** ((a * np.log(x_values) ** 4) + (b * np.log(x_values) ** 3) + (c * np.log(x_values) ** 2) +
               (d * np.log(x_values)) + e), color='purple', label='Log $n^4$', linestyle=':')

    residuals_4 = log_flux_density - ((a * log_pivot_energy ** 4) + (b * log_pivot_energy ** 3) + (c * log_pivot_energy ** 2) + (d * log_pivot_energy) + e)

    print("RMSE of $n^4$ Fit: {}".format(root_mean_squared_error(log_flux_density, (a * log_pivot_energy ** 4) + (b * log_pivot_energy ** 3) + (c * log_pivot_energy ** 2) + (d * log_pivot_energy) + e)))
    print('a: {} b: {} c: {} d: {} e: {}'.format(a, b, c, d, e))

    # Calculate least squares fit - in case variables need to be normally distributed, we know that log of E_0 and log
    # of F_0 are both normally distributed (log-normal distribution)

    # Formatting

    fig.suptitle('Fitting AGN Flux Density-Pivot Energy Relationship')

    ax[0].set_xlabel('$E_0$')
    ax[0].set_ylabel('$F_{0, AGN}$')
    ax[0].set_title('Flux Density against Pivot Energy', fontsize=10)
    ax[0].legend()

    ax[1].set_title('Log-Log Plot of Flux Density against Pivot Energy', fontsize=10)
    ax[1].set_xlabel('log $E_0$')
    ax[1].set_ylabel('log $F_{0, AGN}$')
    ax[1].legend()

    # Scientific notation
    t = ax[0].yaxis.get_offset_text()
    t.set_x(-0.2)

    fig.tight_layout()
    fig.align_titles()

    fig.show()

    # Residuals

    plt.rcParams["figure.figsize"] = (27, 6)

    fig2, ax2 = plt.subplots(1, 4)

    fig2.suptitle('Residuals')

    ax2[0].scatter(log_pivot_energy, residuals_straight / (np.std(residuals_straight, ddof=1)), s=4)
    ax2[1].scatter(log_pivot_energy, residuals_quad / np.std(residuals_quad, ddof=1), s=4)

    print("Standard Deviation of Residuals for Quadratic Fit: {}".format(np.std(residuals_quad, ddof=1)))

    # CHECKING SIMULATED NOISE
    # ax2[1].scatter(log_pivot_energy, np.random.normal(loc=0, scale=np.std(residuals_quad, ddof=1), size=len(log_pivot_energy)), s=2)

    # LOOKING AT MEASUREMENTS TEXTBOOK - WANT ABOUT 96% BETWEEN -2 and +2
    print("Percentage of normalised residuals outside of [-2, 2]: {}".format(1 - np.sum(np.abs(residuals_quad / np.std(residuals_quad)) > 2)/len(residuals_quad)))

    ax2[2].scatter(log_pivot_energy, residuals_cubic / np.std(residuals_cubic, ddof=1), s=4)
    ax2[3].scatter(log_pivot_energy, residuals_4 / np.std(residuals_4, ddof=1), s=4)

    ax2[0].set_title('Linear Fit to Log-Log Plot')
    ax2[1].set_title('Quadratic Fit to Log-Log Plot')
    ax2[2].set_title('Cubic Fit to Log-Log Plot')
    ax2[3].set_title('$n^4$ Fit to Log-Log Plot')

    ax2[0].set_xlabel('log $E_0$')
    ax2[1].set_xlabel('log $E_0$')
    ax2[2].set_xlabel('log $E_0$')
    ax2[3].set_xlabel('log $E_0$')

    ax2[0].set_ylabel('$y_i - \hat{y}_i$')
    ax2[1].set_ylabel('$y_i - \hat{y}_i$')
    ax2[2].set_ylabel('$y_i - \hat{y}_i$')
    ax2[3].set_ylabel('$y_i - \hat{y}_i$')

    fig2.show()


def fitting_agn_pivot_energy_spectral_slope_relation(agns):

    plt.rcParams["figure.figsize"] = (10, 5)

    # Unit conversion
    agns['Pivot_Energy'] = agns['Pivot_Energy'].to(u.GeV)

    # Convert data type
    agns = agns.to_pandas()

    # Remove sources with NaN values
    agns.dropna(inplace=True)

    pivot_energies, alphas = agns['Pivot_Energy'], agns['LP_Index']
    log_pivot_energies, log_alphas = np.log(pivot_energies), np.log(alphas)

    # Create plot

    fig, ax = plt.subplots(1, 2)

    # Plot data

    ax[0].scatter(pivot_energies, alphas, s=4, marker='+')
    ax[1].scatter(log_pivot_energies, log_alphas, s=4, marker="+")

    # Fitting

    x_values = np.linspace(np.min(pivot_energies), np.max(pivot_energies), 1000)
    log_x_values = np.log(x_values)
    exp_x_values = np.exp(log_x_values)

    # Linear Fit
    m, c = np.polyfit(log_pivot_energies, log_alphas, deg=1)
    ax[0].plot(x_values, (x_values ** m) * (np.e ** c), label='Log Linear', color='red')
    ax[1].plot(log_x_values, m * log_x_values + c, color='red', label='Linear')
    linear_residuals = log_alphas - (m * log_pivot_energies + c)
    print("RMSE of Linear Fit: {}".format(root_mean_squared_error(log_alphas, m * log_pivot_energies + c)))
    print('m: {} c: {}'.format(m, c))

    # Quadratic fit
    a, b, c = np.polyfit(log_pivot_energies, log_alphas, deg=2)
    ax[0].plot(exp_x_values, np.e ** ((a * log_x_values ** 2) + (b * log_x_values) + c), color='green',
               label='Log Quadratic', linestyle='-.')
    ax[1].plot(log_x_values, (a * log_x_values ** 2) + (b * log_x_values) + c, color='green',
               label='Quadratic', linestyle='-.')
    quadratic_residuals = log_alphas - ((a * log_pivot_energies ** 2) + (b * log_pivot_energies) + c)
    print("RMSE of Quadratic Fit: {}".format(root_mean_squared_error(log_alphas, (a * log_pivot_energies ** 2)
                                                                     + (b * log_pivot_energies) + c)))
    print('a: {} b: {} c: {}'.format(a, b, c))

    # Cubic fit
    a, b, c, d = np.polyfit(log_pivot_energies, log_alphas, deg=3)
    ax[0].plot(exp_x_values, np.e ** ((a * log_x_values ** 3) + (b * log_x_values ** 2) + (c * log_x_values) + d),
               linestyle='--', label='Log Cubic', color='orange')
    ax[1].plot(log_x_values, (a * log_x_values ** 3) + (b * log_x_values ** 2) + (c * log_x_values) + d, color='orange',
               label='Cubic', linestyle='--')
    cubic_residuals = log_alphas - ((a * log_pivot_energies ** 3) + (b * log_pivot_energies ** 2)
                                    + (c * log_pivot_energies) + d)
    print("RMSE of Cubic Fit: {}".format(root_mean_squared_error(log_alphas, ((a * log_pivot_energies ** 3)
                                                                              + (b * log_pivot_energies ** 2)
                                                                              + (c * log_pivot_energies) + d))))
    print('a: {} b: {} c: {} d: {}'.format(a, b, c, d))

    # Quartic fit
    a, b, c, d, e = np.polyfit(log_pivot_energies, log_alphas, deg=4)
    ax[0].plot(exp_x_values, np.e ** ((a * log_x_values ** 4) + (b * log_x_values ** 3) + (c * log_x_values ** 2) +
                                      (d * log_x_values) + e), color='purple', label='Log Quartic', linestyle=':')
    ax[1].plot(log_x_values, (a * log_x_values ** 4) + (b * log_x_values ** 3) + (c * log_x_values ** 2) +
               (d * log_x_values) + e, color='purple', label='Quartic', linestyle=':')
    quartic_residuals = log_alphas - ((a * log_pivot_energies ** 4) + (b * log_pivot_energies ** 3) +
                                      (c * log_pivot_energies ** 2) + (d * log_pivot_energies) + e)
    print("RMSE of Quartic Fit: {}".format(root_mean_squared_error(log_alphas, ((a * log_pivot_energies ** 4) +
                                                                                (b * log_pivot_energies ** 3) +
                                                                                (c * log_pivot_energies ** 2) +
                                                                                (d * log_pivot_energies) + e))))
    print('a: {} b: {} c: {} d: {} e: {}'.format(a, b, c, d, e))

    # Suggested fit
    # This paper states that the spectral index depends linearly on ln E - https://journals-aps-org.ezphost.dur.ac.uk/
    # prd/abstract/10.1103/k5dp-5str
    m, c = np.polyfit(log_pivot_energies, alphas, deg=1)
    ax[0].plot(x_values, (m * log_x_values) + c, color='black', label='Suggested')
    ax[1].plot(log_x_values, np.log((log_x_values * m) + c), color='black', label='Suggested')
    suggested_residuals = log_alphas - (np.log((log_pivot_energies * m) + c))
    print("RMSE of Suggested Fit: {}".format(root_mean_squared_error(log_alphas, np.log((log_pivot_energies * m) + c))))
    print('m: {} c: {}'.format(m, c))

    # Formatting

    ax[0].set_xlabel("$E_0$")
    ax[0].set_ylabel("$\\alpha$")
    ax[0].set_title("Pivot Energy vs Spectral Slope")
    ax[0].legend()

    ax[1].set_xlabel("log $E_0$")
    ax[1].set_ylabel("log $\\alpha$")
    ax[1].set_title("Log-Log Plot of Pivot Energy vs Spectral Slope")
    ax[1].legend()

    fig.suptitle("Fitting AGN $E_0$-$\\alpha$ Dependency")

    fig.tight_layout()

    fig.show()

    # Residuals

    plt.rcParams["figure.figsize"] = (30, 6)

    fig2, ax2 = plt.subplots(1, 5)

    fig2.suptitle('Residuals')

    ax2[0].scatter(log_pivot_energies, linear_residuals / (np.std(linear_residuals, ddof=1)), s=4)
    ax2[1].scatter(log_pivot_energies, quadratic_residuals / np.std(quadratic_residuals, ddof=1), s=4)
    ax2[2].scatter(log_pivot_energies, cubic_residuals / np.std(cubic_residuals, ddof=1), s=4)
    ax2[3].scatter(log_pivot_energies, quartic_residuals / np.std(quartic_residuals, ddof=1), s=4)
    ax2[4].scatter(log_pivot_energies, suggested_residuals / np.std(suggested_residuals, ddof=1), s=4)

    print("Standard Deviation of Residuals for Suggested Fit: {}".format(np.std(suggested_residuals, ddof=1)))

    # Formatting

    ax2[0].set_title('Linear Fit to Log-Log Plot')
    ax2[1].set_title('Quadratic Fit to Log-Log Plot')
    ax2[2].set_title('Cubic Fit to Log-Log Plot')
    ax2[3].set_title('Quartic Fit to Log-Log Plot')
    ax2[4].set_title('Suggested Fit to Log-Log Plot')

    ax2[0].set_xlabel('log $E_0$')
    ax2[1].set_xlabel('log $E_0$')
    ax2[2].set_xlabel('log $E_0$')
    ax2[3].set_xlabel('log $E_0$')
    ax2[4].set_xlabel('log $E_0$')

    ax2[0].set_ylabel('$y_i - \hat{y}_i$')
    ax2[1].set_ylabel('$y_i - \hat{y}_i$')
    ax2[2].set_ylabel('$y_i - \hat{y}_i$')
    ax2[3].set_ylabel('$y_i - \hat{y}_i$')
    ax2[4].set_ylabel('$y_i - \hat{y}_i$')

    fig2.show()


def analysis(agn_rows, pulsar_rows, directory="./plots/analysis"):

    # Create directory to store results
    Path(directory).mkdir(parents=True, exist_ok=True)

    # PREPARE DATA

    # Convert to pandas dataframes for covariance and correlation calculations, as well as plotting
    agns = agn_rows.to_pandas()
    pulsars = pulsar_rows.to_pandas()

    # PARAMETER DISTRIBUTIONS

    print("PARAMETER DISTRIBUTION ANALYSIS")
    print("\n")

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
    # plot_parameter_distributions(agn_rows.copy(), source_type='AGN', directory=directory)
    plot_parameter_distributions(agns, source_type='AGN', directory=directory)

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
    # plot_parameter_distributions(pulsar_rows.copy(), source_type='Pulsars', directory=directory)
    plot_parameter_distributions(pulsars, source_type='Pulsars', directory=directory)

    # CORRELATION ANALYSIS

    print("PARAMETER CORRELATION ANALYSIS")
    print("\n")

    # AGNs

    # Plot parameters against one another to visualise relationships
    plot_parameter_relationships(agns, source_type="AGN", directory=directory)

    # Plot AGN parameter correlation matrices
    plot_correlation_matrices(agns, source_type="AGN", directory=directory)

    # Pulsars

    # Plot parameters against one another to visualise relationships
    plot_parameter_relationships(pulsars, source_type="Pulsar", directory=directory)

    # Plot pulsar parameter correlation matrices
    plot_correlation_matrices(pulsars, source_type="Pulsar", directory=directory)


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






def num_sources_to_generate(energy_fluxes_4fgl, detection_threshold):

    # THIS DETERMINES THE LUMINOSITY FUNCTION OF MOCK CATALOG - NOT FINISHED YET


    # The minimum energy flux of our generated sources is an order of magnitude less than the 4FGL
    our_threshold = detection_threshold / 10

    # Following method detailed in ID8

    # Bin 4FGL data
    min_bin_val = np.log10(np.max(energy_fluxes_4fgl))
    max_bin_val = np.log10(np.max(energy_fluxes_4fgl))

    bin_edges = 10 ** np.linspace(min_bin_val, max_bin_val)
    counts, bin_intervals = np.histogram(energy_fluxes_4fgl, bins=bin_edges)

    # Calculate width of bins
    bin_width = bin_intervals[1] - bin_intervals

    # Calculate number of random

    # Find bin with the most number of AGNs
    peak = np.argmax(counts)

    # Generate random numbers for number of energy flux bins to the right of the peak
    n_noise = np.random.uniform(low=0.8, high=1.3, size=len(bin_intervals) - peak)


# Read in catalog data

agn_rows, pulsar_rows, source_detection_threshold, fluxes_4fgl = catalog_data_preparation("/Volumes/T7/data/catalog/4FGL_DR4.fit")

# print(pulsar_rows)

# Analyse parameters, their distributions, and their correlations

# analysis(agn_rows, pulsar_rows)

# fitting_agn_pivot_energy_flux_density_relation(agn_rows.copy())

# fitting_agn_pivot_energy_spectral_slope_relation(agn_rows.copy())

# Generate simulated AGN sources

# agns = generate_mock_agn_catalog(agn_rows.copy(), num_agns=300, detection_threshold=source_detection_threshold)

# Generate simulated pulsar sources

pulsars = generate_mock_pulsar_catalog(pulsar_statistics(pulsar_rows.copy()), 10)

print(pulsars)

# Verify realism and correctness of generated gamma-ray sources
# verification(agns, pulsars)


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
