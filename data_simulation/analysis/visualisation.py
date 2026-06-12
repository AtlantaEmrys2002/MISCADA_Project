from astropy.table import QTable
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.stats import lognorm, norm
from .goodness_of_fit import chi_squared_test, kolmogorov_smirnov_test


def agn_luminosity_function(catalog: str, energy_fluxes: npt.NDArray[np.float64]) -> None:

    # Data Processing

    # Read 4FGL Catalog
    catalog = QTable.read(catalog, format='fits', hdu=1)['CLASS1', 'Energy_Flux100']

    # Reformat columns
    catalog['CLASS1'] = np.asarray([k.decode('utf-8').strip().lower() for k in catalog['CLASS1'].value.filled('-')])

    # Select all rows that describe AGN
    agn_mask = np.isin(catalog['CLASS1'].data, np.array(['bcu', 'sey', 'ssrq', 'bll', 'fsrq', 'rdg', 'nlsy1', 'agn']))
    agns = catalog[agn_mask]

    energy_fluxes_4fgl = agns['Energy_Flux100'].value

    # Plotting

    # Set plot size
    plt.rcParams["figure.figsize"] = (6.4, 4.8)

    # Plot 4FGL data
    bin_edges = 10**np.linspace(-14, -9, 50)
    counts, bins = np.histogram(energy_fluxes_4fgl, bins=bin_edges)
    plt.stairs(counts, bins, label='4FGL')

    # Plot simulated data
    bin_edges = 10**np.linspace(-15, -8, 50)
    counts, bins = np.histogram(energy_fluxes, bins=bin_edges)
    plt.stairs(counts, bins, label='Simulated')

    # Formatting

    plt.title('AGN Luminosity Function')

    plt.xlabel('Energy Flux')
    plt.ylabel('No. Sources')

    plt.xscale('log')
    plt.yscale('log')

    plt.xlim(10**-14, 10**-8)
    plt.ylim(top=10**4)

    plt.legend()

    plt.show()


def log_normal_parameter(values):

    # This calculates the parameters for creating a log normal distribution based on parameter data

    mean, sigma = np.mean(values), np.std(values, ddof=1)

    mean_square = mean ** 2
    std_square = sigma ** 2

    mean_log = np.log(mean_square / (np.sqrt(mean_square + std_square)))
    std_log = np.sqrt(np.log(1 + (std_square / mean_square)))

    return np.exp(mean_log), std_log


def analysing_agn_parameters(agns):

    # ID8 assert that F_0 follows log normal distribution and other params in differential energy flux follow Gaussian
    # we check this

    # PIVOT ENERGY ANALYSIS

    plt.rcParams["figure.figsize"] = (10, 10)

    fig, ax = plt.subplots(2, 2)

    # READ IN DATA

    agns = agns.to_pandas()

    # PLOT DISTRIBUTIONS

    for pair in zip(ax.flatten(), [agns[col] for col in agns]):

        subplot = pair[0]

        # Remove NaN values
        values = pair[1][~np.isnan(pair[1])]

        # Plot actual 4FGL source distribution
        counts, bins = np.histogram(values, bins=100, density=True)
        subplot.stairs(counts, bins, label='4FGL Distribution')

        # Calculate distribution parameters of data
        mean, sigma = np.mean(values), np.std(values, ddof=1)
        scale, s = log_normal_parameter(values)

        x_values = np.linspace(np.min(values), np.max(values), 1000)

        # Plot Gaussian using mean and standard deviation of data
        subplot.plot(x_values, norm.pdf(x_values, loc=mean, scale=sigma), label='Gaussian', linestyle='-.')

        # Plot Log-Normal distribution using mean and standard deviation of data (recommended by ID8 for F_0, but not
        # any of the other AGN parameters)
        subplot.plot(x_values, lognorm.pdf(x_values, s=s, scale=scale), color='red',
                     label='Log-Normal', linestyle='--')

        chi_squared_test(values, num_bins=100, distribution='normal')
        chi_squared_test(values, num_bins=100, distribution="lognorm")
        kolmogorov_smirnov_test(values=values, distribution='normal')
        kolmogorov_smirnov_test(values=values, distribution='lognorm')

    # FORMATTING

    fig.suptitle('Distributions of 4FGL AGN Parameters')

    ax[0, 0].set_title('Differential Flux Densities, $F_0$')
    ax[0, 1].set_title('Pivot Energies, $E_0$')
    ax[1, 0].set_title('Spectral Slopes, $\\alpha$')
    ax[1, 1].set_title('Spectral Curvature, $\\beta$')

    ax[0, 0].set_xlabel('$F_0$ [ph cm$^{-2}$ MeV$^{-1}$ s$^{-1}$]')
    ax[0, 1].set_xlabel('$E_0$ [MeV]')
    ax[1, 0].set_xlabel('$\\alpha$')
    ax[1, 1].set_xlabel('$\\beta$')

    # Label axes and enable legends
    for a in ax.flatten():
        a.set_ylabel('Source Density')
        a.legend()

    fig.tight_layout()

    plt.show()


def analysing_pulsar_parameters(pulsars):

    # Create plot
    plt.rcParams["figure.figsize"] = (8, 12)
    fig, ax = plt.subplots(3, 2)
    fig.delaxes(ax[2, 1])

    pulsars = pulsars.to_pandas()

    for pair in zip(ax.flatten(), [pulsars[col] for col in pulsars]):

        subplot = pair[0]

        # Remove NaN values
        values = pair[1][~np.isnan(pair[1])]

        # Plot actual 4FGL source distribution
        counts, bins = np.histogram(values, bins=30, density=True)
        subplot.stairs(counts, bins, label='4FGL Distribution')

        # Calculate distribution parameters of data
        mean, sigma = np.mean(values), np.std(values, ddof=1)
        scale, s = log_normal_parameter(values)

        # Plot Gaussian using mean and standard deviation of data
        x_values = np.linspace(np.min(values), np.max(values), 1000)
        subplot.plot(x_values, norm.pdf(x_values, loc=mean, scale=sigma), label='Gaussian', linestyle='-.')

        # Plot Log-Normal distribution using mean and standard deviation of data (recommended by ID8 for F_0, but not
        # any of the other AGN parameters)
        subplot.plot(x_values, lognorm.pdf(x_values, s=s, scale=scale), color='red', label='Log-Normal',
                     linestyle='--')

        chi_squared_test(values, num_bins=30, distribution='normal')
        chi_squared_test(values, num_bins=30, distribution='lognorm')
        kolmogorov_smirnov_test(values=values, distribution='normal')
        kolmogorov_smirnov_test(values=values, distribution='lognorm')

    # FORMATTING

    fig.suptitle('Distributions of 4FGL Pulsar Parameters')

    ax[0, 0].set_title('Differential Flux Densities, $F_0$')
    ax[0, 1].set_title('Pivot Energies, $E_0$')
    ax[1, 0].set_title('Spectral Slopes, $\\Gamma$')
    ax[1, 1].set_title('Exponential Indices, $b$')
    ax[2, 0].set_title('Exponential Factors, $a$')

    ax[0, 0].set_xlabel('$F_0$ [ph cm$^{-2}$ MeV$^{-1}$ s$^{-1}$]')
    ax[0, 1].set_xlabel('$E_0$ [MeV]')
    ax[1, 0].set_xlabel('$\\Gamma$')
    ax[1, 1].set_xlabel('$b$')
    ax[2, 0].set_xlabel('$a\ [MeV$^{-b}$]$')

    ax[0, 0].set_ylim(0, 1 * 10 ** 11)

    # Label axes and enable legends
    for a in ax.flatten():
        a.set_ylabel('Source Density')
        a.legend()

    fig.tight_layout()

    plt.show()


def plot_parameter_distributions(sources, source_type: str):

    # Used to label axes
    axis_labels = {"LP_Flux_Density": "Differential Flux Density, $F_0$ \n [ph cm$^{-2}$ MeV$^{-1}$ s$^{-1}$",
                   "Pivot_Energy": "Pivot Energy, $E_0$ [MeV]", "LP_Index": "Spectral Slope, $\\alpha$",
                   "LP_beta": "Spectral Curvature, $\\beta$", "PLEC_IndexS": "Spectral Slope, $\\Gamma$",
                   "PLEC_Flux_Density": "Differential Flux Density, $F_0$ \n [ph cm$^{-2}$ MeV$^{-1}$ s$^{-1}$",
                   "PLEC_Exp_Index": "Exponential Index, $b$", "PLEC_ExpfactorS": "Exponential Factor, $a$"}

    num_parameters = len(sources.colnames)

    # CREATE PLOTS

    # Calculate appropriate number of parameters
    num_cols = (num_parameters // 2) + (num_parameters % 2)
    num_rows = 2

    plt.rcParams["figure.figsize"] = (num_rows * 4, num_cols * 4)
    fig, ax = plt.subplots(num_cols, num_rows)

    if (num_parameters % 2) == 1:
        fig.delaxes(ax[num_cols - 1, 1])

    # GENERATE DATA

    # Format sources
    sources = sources.to_pandas()

    for pair in zip(ax.flatten(), [sources[col] for col in sources]):

        subplot = pair[0]

        # Remove NaN values
        values = pair[1][~np.isnan(pair[1])]

        # Plot actual 4FGL source distribution
        counts, bins = np.histogram(values, bins=30, density=True)
        subplot.stairs(counts, bins, label='4FGL Distribution')

        # Calculate distribution parameters of data
        mean, sigma = np.mean(values), np.std(values, ddof=1)
        scale, s = log_normal_parameter(values)

        # Plot Gaussian using mean and standard deviation of data
        x_values = np.linspace(np.min(values), np.max(values), 1000)
        subplot.plot(x_values, norm.pdf(x_values, loc=mean, scale=sigma), label='Gaussian', linestyle='-.')

        # Plot Log-Normal distribution using mean and standard deviation of data (recommended by ID8 for F_0, but not
        # any of the other AGN parameters)
        subplot.plot(x_values, lognorm.pdf(x_values, s=s, scale=scale), color='red', label='Log-Normal',
                     linestyle='--')

        subplot.set_xlabel(axis_labels[values.name])

    # FORMATTING

    fig.suptitle('Distributions of 4FGL {} Parameters'.format(source_type))

    # Label axes and enable legends
    for a in ax.flatten():
        a.set_ylabel('Source Density')
        a.legend()

    fig.tight_layout()

    plt.show()


# REFERENCES

# Astropy Documentation - https://docs.astropy.org/en/stable/index_user_docs.html
# Numpy Documentation - https://numpy.org/doc/stable/index.html
# Numpy Typing - https://stackoverflow.com/questions/35673895/type-hinting-annotation-pep-484-for-numpy-ndarray
# Scipy Documentation - https://docs.scipy.org/doc/scipy/index.html
