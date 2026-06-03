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
# 5. CHECK MIN-MAX VALUES IN XML
# 6. Think you include the background files during gtmodel - but check!!!
# 7. CHECK ALL UNITS - ID43

# LIBRARIES
import numpy as np
from scipy.integrate import quad
from astropy.table import QTable
from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import warnings
from xml.dom import minidom
import time
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture
from scipy.stats import rv_continuous
from scipy import stats
from scipy.stats.sampling import NumericalInversePolynomial, NumericalInverseHermite, RatioUniforms, SimpleRatioUniforms, TransformedDensityRejection
from sympy.stats import ContinuousRV, MultivariateNormal, sample
from sympy import Interval, oo, Symbol
from scipy.integrate import trapezoid

# VISUALISATIONS


def spatial_visualisations(galactic_longitudes, galactic_latitudes, num_agn, source_type='AGNs'):

    xs, ys = [], []

    for k in range(num_agn):

        ra_dec = SkyCoord(l=galactic_longitudes[k] * u.rad, b=galactic_latitudes[k] * u.rad,
                          frame='galactic').transform_to('icrs')

        xs.append(ra_dec.ra.to_value(u.degree))
        ys.append(ra_dec.dec.to_value(u.degree))

    fig, ax = plt.subplots(figsize=(8, 4.2), subplot_kw=dict(projection="aitoff"))
    ax.set_title("Distribution of Simulated " + source_type + " on the sky", pad=20)
    ax.grid(True)
    ax.scatter(xs, ys, marker='o', s=2, alpha=0.3)
    fig.subplots_adjust(top=0.95, bottom=0.0)
    plt.show()


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


def agn_luminosity_function(energy_fluxes):

    # I implemented this to recreate a plot style (luminosity function) shown in ID8

    # 4FGL

    catalog = QTable.read("/Volumes/T7/data/catalog/4FGL_DR4.fit", format='fits', hdu=1)

    catalog['CLASS1'].name = 'Prev_CLASS1'

    # catalog['CLASS1'] = np.asarray([k.strip().lower() for k in catalog['Prev_CLASS1']])

    catalog['CLASS1'] = np.asarray(
        [k.decode('utf-8').strip().lower() for k in catalog['Prev_CLASS1'].value.filled('-')])

    catalog.remove_column('Prev_CLASS1')

    # Select all rows that describe AGN
    agn_mask = ((catalog['CLASS1'] == 'bcu') | (catalog['CLASS1'] == 'sey') | (catalog['CLASS1'] == 'ssrq') |
                (catalog['CLASS1'] == 'bll') | (catalog['CLASS1'] == 'fsrq') | (catalog['CLASS1'] == 'rdg') |
                (catalog['CLASS1'] == 'nlsy1') | (catalog['CLASS1'] == 'agn'))

    agns = catalog[agn_mask]

    energy_fluxes_4fgl = agns['Energy_Flux100'].value

    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('Energy Flux')
    plt.ylabel('No. Sources')

    plt.xlim(10**-14, 10**-8)

    plt.ylim(top=10**4)

    bin_edges = 10**np.linspace(-14, -9, 50)

    counts, bins = np.histogram(energy_fluxes_4fgl, bins=bin_edges)

    plt.stairs(counts, bins, label='4fgl')

    # SIMULATED

    plt.xscale('log')

    plt.xlabel('Energy Flux')
    plt.ylabel('No. Sources')

    bin_edges = 10**np.linspace(-15, -8, 50)

    # print(bin_edges)

    counts, bins = np.histogram(energy_fluxes, bins=bin_edges)

    plt.stairs(counts, bins, label='simulated')

    plt.legend()

    plt.show()


def fgl_agn_luminosity_function(file):

    catalog = QTable.read(file, format='fits', hdu=1)

    energy_fluxes = catalog['Energy_Flux100'].value

    plt.xscale('log')

    plt.xlabel('Energy Flux')
    plt.ylabel('No. Sources')

    bin_edges = 10**np.linspace(-15, -8, 40)

    counts, bins = np.histogram(energy_fluxes, bins=bin_edges)

    plt.stairs(counts, bins)

    plt.legend(True)

    plt.show()

# fgl_agn_luminosity_function("/Volumes/T7/data/catalog/4FGL_DR4.fit")


# SPECTRAL MODELS

def agn_spectral_model(E, E_0, F_0, alpha, beta):

    division = E/E_0

    # with suppress(RuntimeWarning):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exponent = - alpha - (beta * np.log(division))

    dF_dE = F_0 * np.power(division, exponent)

    return E * dF_dE


def pulsar_spectral_model(E, F_0, E_0, Gamma, a, b):

    exponent = a * (np.power(E_0, b) - np.power(E, b))

    dF_dE = F_0 * np.power((E/E_0), -Gamma) * np.exp(exponent)

    return E * dF_dE


def energy_flux_agn(pivot_energy, flux_density, spectral_slope, curvature):

    # Integrate over 0.1 - 100 GeV
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     energy = quad(agn_spectral_model, 0.1, 100, args=(pivot_energy, flux_density, spectral_slope, curvature))[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        energy = quad(agn_spectral_model, 100, 100000, args=(pivot_energy, flux_density, spectral_slope, curvature))[0]

    # Convert from MeV to Erg (see ID43)

    return energy


def s1_agn(pivot_energy, flux_density, spectral_slope, curvature):

    # Integrate above 1 GeV
    # s1 = quad(agn_spectral_model, 1, np.inf, args=(pivot_energy, flux_density, spectral_slope, curvature))[0]

    s1 = quad(agn_spectral_model, 1000, np.inf, args=(pivot_energy, flux_density, spectral_slope, curvature))[0]

    return s1


def s10_agn(pivot_energy, flux_density, spectral_slope, curvature):

    # Integrate above 1 GeV
    # s1 = quad(agn_spectral_model, 10, np.inf, args=(pivot_energy, flux_density, spectral_slope, curvature))[0]

    s1 = quad(agn_spectral_model, 10000, np.inf, args=(pivot_energy, flux_density, spectral_slope, curvature))[0]

    return s1


def energy_flux_pulsar(pivot_energy, flux_density, spectral_slope, exponential_index, exponential_factor):

    # Integrate over 0.1 - 100 GeV - change to 100 - 100000 MeV
    # energy = quad(pulsar_spectral_model, 0.1, 100, args=(pivot_energy, flux_density, spectral_slope,
    #                                                      exponential_index, exponential_factor))[0]

    energy = quad(pulsar_spectral_model, 100, 100000, args=(pivot_energy, flux_density, spectral_slope,
                                                            exponential_index, exponential_factor))[0]

    return energy


# DATA PREP

def catalog_data_preparation(file_name):

    # Access 4FGL catalog - note, file originally called gll_psc_v35.fit
    # Read catalog data and format as astropy QTable (allowing for units to be associated with each column, if necessary
    # Assume catalog data conforms to standard NASA format
    catalog = QTable.read(file_name, format='fits', hdu=1)

    # Select relevant columns
    columns = ('Pivot_Energy', 'LP_Flux_Density', 'PLEC_Flux_Density', 'LP_Index', 'LP_beta', 'PLEC_IndexS',
               'PLEC_Exp_Index', 'PLEC_ExpfactorS', 'CLASS1', 'GLAT')
    catalog = catalog[columns]

    # print(np.asarray([k.decode('utf-8').strip().lower() for k in catalog['CLASS1'].value.filled('-')]))

    # Reformat CLASS1 column - remove empty spaces and make all lower case
    catalog['CLASS1'].name = 'Prev_CLASS1'

    # catalog['CLASS1'] = np.asarray([k.strip().lower() for k in catalog['Prev_CLASS1']])

    catalog['CLASS1'] = np.asarray([k.decode('utf-8').strip().lower() for k in catalog['Prev_CLASS1'].value.filled('-')])

    catalog.remove_column('Prev_CLASS1')

    # Select all rows that describe pulsars
    pulsar_mask = (catalog['CLASS1'] == 'psr')

    # Select all rows that describe AGN
    agn_mask = ((catalog['CLASS1'] == 'bcu') | (catalog['CLASS1'] == 'sey') | (catalog['CLASS1'] == 'ssrq') |
                (catalog['CLASS1'] == 'bll') | (catalog['CLASS1'] == 'fsrq') | (catalog['CLASS1'] == 'rdg') |
                (catalog['CLASS1'] == 'nlsy1') | (catalog['CLASS1'] == 'agn'))

    catalog.remove_column('CLASS1')

    # Separate into AGN and pulsars
    pulsars = catalog[pulsar_mask]
    agns = catalog[agn_mask]

    return agns, pulsars


def normal_func(x, mean, sigma):

    var = sigma**2

    # return (1/np.sqrt(2 * np.pi * var)) * np.e**-(((x - mean)**2) / (2 * var))

    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(((x - mean) ** 2) / (2 * var)))


# This function from here - https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-
# from-a-list
# Using median instead of mean to check for outliers
# def reject_outliers(data, m=100.):
#     d = np.abs(data - np.median(data))
#     mdev = np.median(d)
#     s = d/mdev if mdev else np.zeros(len(d))
#     return data[s < m]

def log_norm_pdf(x, mu, sigma):

    upper = (np.log(x) - mu)**2
    lower = 2 * (sigma ** 2)
    fraction = -1 * upper / lower

    f_x = (1/(x * sigma * np.sqrt(2 * np.pi))) * np.exp(fraction)

    return f_x


def analysing_agn_parameters(agns):

    # ID8 assert that F_0 follows log normal distribution and other params in differential energy flux follow Gaussian
    # we check this

    # PIVOT ENERGY ANALYSIS

    plt.rcParams["figure.figsize"] = (10, 10)

    fig, ax = plt.subplots(2, 2)

    # READ IN DATA

    fds = agns['LP_Flux_Density'].to(u.ph / (u.cm * u.cm * u.MeV * u.s)).value.filled(np.nan)
    pivot_energies = agns['Pivot_Energy'].value
    alphas = agns['LP_Index'].data.filled(np.nan)
    betas = agns['LP_beta'].data.filled(np.nan)

    # PLOT DISTRIBUTIONS

    for pair in zip(ax.flatten(), [fds, pivot_energies, alphas, betas]):

        subplot = pair[0]

        # Remove NaN values
        values = pair[1][~np.isnan(pair[1])]

        # Plot actual 4FGL source distribution
        counts, bins = np.histogram(values, bins=100, density=True)
        subplot.stairs(counts, bins, label='4FGL Distribution')

        # Calculate mean and standard deviation of data
        mean, sigma = np.mean(values), np.std(values, ddof=1)

        # Plot Gaussian using mean and standard deviation of data
        x_values = np.linspace(np.min(values), np.max(values), 1000)
        y_values = normal_func(x_values, mean, sigma)
        subplot.plot(x_values, y_values, label='Gaussian', linestyle='-.')

        # Plot Log-Normal distribution using mean and standard deviation of data (recommended by ID8 for F_0, but not
        # any of the other AGN parameters)

        mean_square = mean ** 2
        std_square = sigma ** 2

        mean_log = np.log(mean_square / (np.sqrt(mean_square + std_square)))
        std_log = np.sqrt(np.log(1 + (std_square / mean_square)))

        subplot.plot(x_values, log_norm_pdf(x_values, mean_log, std_log), label='Log-Normal', color='red',
                     linestyle='--')

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

    # pivot_energies, flux_densities, spectral_slopes, exponential_indices, exponential_factors

    fds = pulsars['PLEC_Flux_Density'].to(u.ph / (u.cm * u.cm * u.MeV * u.s)).value.filled(np.nan)
    pivot_energies = pulsars['Pivot_Energy'].value
    Gammas = pulsars['PLEC_IndexS'].data.filled(np.nan)
    b_values = pulsars['PLEC_Exp_Index'].data.filled(np.nan)

    # Select exponential factors - CHECK (SAYS IN UNITS OF MeV^-b BUT NEED with GeV)
    a_values = pulsars['PLEC_ExpfactorS'].data

    for pair in zip(ax.flatten(), [fds, pivot_energies, Gammas, b_values, a_values]):

        subplot = pair[0]

        # Remove NaN values
        values = pair[1][~np.isnan(pair[1])]

        # Plot actual 4FGL source distribution
        counts, bins = np.histogram(values, bins=20, density=True)
        subplot.stairs(counts, bins, label='4FGL Distribution')

        # Calculate mean and standard deviation of data
        mean, sigma = np.mean(values), np.std(values, ddof=1)

        # Plot Gaussian using mean and standard deviation of data
        x_values = np.linspace(np.min(values), np.max(values), 1000)
        y_values = normal_func(x_values, mean, sigma)
        subplot.plot(x_values, y_values, label='Gaussian', linestyle='-.')

        # Plot Log-Normal distribution using mean and standard deviation of data (recommended by ID8 for F_0, but not
        # any of the other AGN parameters)

        mean_square = mean ** 2
        std_square = sigma ** 2

        mean_log = np.log(mean_square / (np.sqrt(mean_square + std_square)))
        std_log = np.sqrt(np.log(1 + (std_square / mean_square)))

        subplot.plot(x_values, log_norm_pdf(x_values, mean_log, std_log), label='Log-Normal', color='red',
                     linestyle='--')

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

    # Label axes and enable legends
    for a in ax.flatten():
        a.set_ylabel('Source Density')
        a.legend()

    fig.tight_layout()

    plt.show()


def agn_statistics(agns):

    # Select alpha values and convert from masked to ordinary numpy array
    alphas = agns['LP_Index'].data.filled(np.nan)

    # Select beta values and convert from masked to ordinary numpy array
    betas = agns['LP_beta'].data.filled(np.nan)

    # Select pivot energy values and convert from MeV to GeV
    pivot_energies = agns['Pivot_Energy'].to(u.GeV).value

    # Select flux density values and convert from ph / (cm2 MeV s) to ph / (cm2 GeV s)
    flux_densities = agns['LP_Flux_Density'].to(u.ph / (u.cm * u.cm * u.GeV * u.s)).value.filled(np.nan)

    log_flux_densities = np.log(flux_densities)

    mean_alpha, std_alpha = np.nanmean(alphas), np.nanstd(alphas, ddof=1)

    log_pivot_energies = np.log(pivot_energies)

    mean_log_pivot_energy, std_log_pivot_energy = np.nanmean(log_pivot_energies), np.nanstd(log_pivot_energies, ddof=1)

    mean_log_flux_density, std_log_flux_density = np.nanmean(log_flux_densities), np.nanstd(log_flux_densities, ddof=1)

    return (mean_alpha, std_alpha, mean_log_pivot_energy, std_log_pivot_energy, mean_log_flux_density, std_log_flux_density,
            betas)


def pulsar_statistics(pulsars):

    # Select pivot energy values and convert from MeV to GeV
    pivot_energies = pulsars['Pivot_Energy'].to(u.GeV).value

    # Select Gamma values and convert from masked to ordinary numpy array
    Gammas = pulsars['PLEC_IndexS'].data.filled(np.nan)

    # Select exponential indices
    b_values = pulsars['PLEC_Exp_Index'].data.filled(np.nan)

    # Select exponential factors - CHECK (SAYS IN UNITS OF MeV^-b BUT NEED with GeV)
    a_values = pulsars['PLEC_ExpfactorS'].data

    # Select flux density values and convert from ph / (cm2 MeV s) to ph / (cm2 GeV s)
    flux_densities = pulsars['PLEC_Flux_Density'].to(u.ph / (u.cm * u.cm * u.GeV * u.s)).value
    log_flux_densities = np.log(flux_densities)

    mean_Gamma, std_Gamma = np.nanmean(Gammas), np.nanstd(Gammas, ddof=1)

    mean_b, std_b = np.nanmean(b_values), np.nanstd(b_values, ddof=1)

    log_a_values = np.log(a_values)

    # mean_a, std_a = np.nanmean(a_values), np.nanstd(a_values, ddof=1)

    mean_log_a, std_log_a = np.nanmean(log_a_values), np.nanstd(log_a_values, ddof=1)

    mean_log_flux_density, std_log_flux_density = np.nanmean(log_flux_densities), np.nanstd(log_flux_densities)

    # mean_pivot_energy, std_pivot_energy = np.nanmean(pivot_energies), np.nanstd(pivot_energies, ddof=1)

    log_pivot_energies = np.log(pivot_energies)

    mean_log_pivot_energy, std_log_pivot_energy = np.nanmean(log_pivot_energies), np.nanstd(log_pivot_energies, ddof=1)

    pulsar_latitudes = pulsars['GLAT']

    return (mean_Gamma, std_Gamma, mean_b, std_b, mean_log_a, std_log_a, mean_log_flux_density, std_log_flux_density,
            mean_log_pivot_energy, std_log_pivot_energy, pulsar_latitudes)


# GENERATE FIXED NUMBER OF AGNS WITHIN GIVEN ENERGY FLUX RANGE FOR FLAT EXTRAPOLATION AT LOWER ENERGY FLUXES
def agn_generation(agn_stats, energy_flux_low, energy_flux_high):

    (mean_alpha_agn, std_alpha_agn, mean_log_pivot_energy_agn, std_log_pivot_energy_agn, mean_log_flux_density_agn,
     std_log_flux_density_agn, betas_agn) = agn_stats

    while True:

        # (mean_alpha_agn, std_alpha_agn, mean_log_pivot_energy_agn, std_log_pivot_energy_agn, mean_log_flux_density_agn,
        #  std_log_flux_density_agn, betas_agn) = agn_stats

        # Generate new pivot energy
        # pivot_energy = np.random.normal(loc=mean_pivot_energy_agn, scale=std_pivot_energy_agn, size=1)[0]

        pivot_energy = np.random.lognormal(mean=mean_log_pivot_energy_agn, sigma=std_log_pivot_energy_agn,
                                             size=1)[0]

        # Generate new flux density - log-normal for flux densities
        flux_density = np.random.lognormal(mean=mean_log_flux_density_agn, sigma=std_log_flux_density_agn, size=1)[0]

        # Generate new spectral slope (alpha)
        spectral_slope = np.random.normal(loc=mean_alpha_agn, scale=std_alpha_agn, size=1)[0]

        # Generate new curvature by directly sampling 4FGL
        beta = np.random.choice(betas_agn, size=1, replace=True)[0]

        energy_flux = energy_flux_agn(pivot_energy, flux_density, spectral_slope, beta) * 1.602 * 10**(-6)

        # longitude
        longitude = np.random.uniform(low=0, high=2 * np.pi, size=1)[0]

        # latitude
        sin_galactic_latitudes = np.random.uniform(low=-1, high=1, size=1)[0]
        latitude = np.arcsin(sin_galactic_latitudes)

        if (energy_flux >= energy_flux_low) and (energy_flux < energy_flux_high):

            return np.asarray([pivot_energy, flux_density, spectral_slope, beta, energy_flux, longitude, latitude])


def generate_mock_agn_catalog(agn_stats, num_agns=100, extra=3400):

    # SPECTRAL PARAMETERS

    # (mean_alpha_agn, std_alpha_agn, mean_pivot_energy_agn, std_pivot_energy_agn, mean_log_flux_density_agn,
    #  std_log_flux_density_agn, betas_agn) = agn_stats

    (mean_alpha_agn, std_alpha_agn, mean_log_pivot_energy_agn, std_log_pivot_energy_agn, mean_log_flux_density_agn,
     std_log_flux_density_agn, betas_agn) = agn_stats

    # Generate new pivot energies - in ID8, they randomly select pivot energies from a Gaussian distribution. However,
    # the distribution of pivot energies in the 4FGL follows log-normal more precisely (see my plots in agn analysis
    # above). Therefore, I have changed the way pivot energies are randomly generated (now use log-normal distribution)
    # pivot_energies = np.random.normal(loc=mean_pivot_energy_agn, scale=std_pivot_energy_agn, size=num_agns)

    pivot_energies = np.random.lognormal(mean=mean_log_pivot_energy_agn, sigma=std_log_pivot_energy_agn, size=num_agns)

    # Generate new flux densities - log-normal for flux densities
    flux_densities = np.random.lognormal(mean=mean_log_flux_density_agn, sigma=std_log_flux_density_agn, size=num_agns)

    # Generate new spectral slopes (alphas)
    spectral_slopes = np.random.normal(loc=mean_alpha_agn, scale=std_alpha_agn, size=num_agns)

    # Generate new curvatures by directly sampling 4FGL
    betas = np.random.choice(betas_agn, size=num_agns, replace=True)

    # Combine into one array
    parameters = np.stack((pivot_energies, flux_densities, spectral_slopes, betas), axis=-1)

    # Energy fluxes
    energy_fluxes = np.fromiter((energy_flux_agn(x[0], x[1], x[2], x[3]) for x in parameters), np.float64)

    # Convert energy fluxes so ergs included in units instead of photons
    energy_fluxes *= 1.602 * 10**(-6)

    # Select rows with valid energy fluxes
    mask = ~np.isnan(energy_fluxes)
    energy_fluxes = energy_fluxes[mask]
    parameters = parameters[mask]

    # Combine two arrays to create mock catalog's spectral parameters
    parameters = np.concatenate((parameters, np.array([energy_fluxes]).T), axis=1)

    # SPATIAL PARAMETERS

    # l
    galactic_longitudes = np.random.uniform(low=0, high=2*np.pi, size=len(parameters))

    # b
    sin_galactic_latitudes = np.random.uniform(low=-1, high=1, size=len(parameters))
    galactic_latitudes = np.arcsin(sin_galactic_latitudes)

    # Combine two arrays to create mock catalog's spatial parameters
    parameters = np.concatenate((parameters, np.array([galactic_longitudes]).T), axis=1)

    # Combine two arrays to create mock catalog's spatial parameters
    parameters = np.concatenate((parameters, np.array([galactic_latitudes]).T), axis=1)

    # Remove all with energy fluxes below our chosen energy flux threshold
    # N.B. Used expression found here - https://stackoverflow.com/questions/72404872/remove-rows-in-a-2d-numpy-array-if-
    # they-contain-a-specific-element - to select desired rows

    # Cut at detection threshold
    # parameters = parameters[~(parameters[:, 4] <= np.float64(2.0 * 10**(-12))), :]
    parameters = parameters[~(parameters[:, 4] <= np.float64(1.0 * 10 ** (-12))), :]

    # ADD MORE SOURCES - FOLLOW LUMINOSITY FUNCTION OF 4FGL - ENSURE ENOUGH

    extra_sources = []

    for x in range(extra):

        # CHANGED TO 1.0 * 10 ** -12 as recommended by 4FGL DR4 (ID22) paper for outside galactic plane

        # new_source = agn_generation(agn_stats, 2.0 * 10**(-12), 1000)
        new_source = agn_generation(agn_stats, 1.0 * 10 ** (-12), 1000)

        # print(new_source)

        extra_sources.append(new_source)

        print(x + 1)

    parameters = np.vstack((parameters, np.asarray(extra_sources)))

    # FAINT SOURCE FLAT EXTRAPOLATION

    # Flat extrapolation of AGN - assume constant below given threshold (not Gaussian)

    # Bin data and take average of first three
    bin_edges = 10**np.linspace(-14, -9, 50)
    counts, _ = np.histogram(parameters[:, 4], bins=bin_edges)

    # Take average number of sources of first five bins for flat extrapolation
    first_non_empty_bin = np.nonzero(counts)[0][0]
    mean_counts_per_bin, std_counts_per_bin = (np.mean(counts[first_non_empty_bin: first_non_empty_bin + 10]),
                                               np.std(counts[first_non_empty_bin: first_non_empty_bin + 10], ddof=1))

    faint_sources = []

    our_threshold = np.argwhere(bin_edges >= 3.4 * 10**(-13))[0][0]

    for x in range(our_threshold, first_non_empty_bin):

        # Generate number of sources in bin - approximately flat/same as bins at peak
        counts_per_bin_flat_extrapolation = round(np.random.normal(loc=mean_counts_per_bin, scale=std_counts_per_bin,
                                                                   size=1)[0])

        for k in range(counts_per_bin_flat_extrapolation):
            new_source = agn_generation(agn_stats, bin_edges[x], bin_edges[x + 1])
            faint_sources.append(new_source)

    faint_sources = np.asarray(faint_sources)

    parameters = np.vstack((parameters, faint_sources))

    agn_luminosity_function(parameters[:, 4])

    return parameters


# CHOSE THIS MYSELF - THINK THIS IS WHAT ID8 WAS SUGGESTING
def split_normal(x, sigma_1, sigma_2):

    mu = 0

    upper = -1 * ((x - mu) ** 2)

    # I defined cutoff between distributions as within 10 degree of galactic plane (lat = 0 degrees)

    # ID8 found different values A_1 and A_2 - normalizing factor is the same for both distributions here (as indicated
    # in split normal distribution wiki - even though this is not a traditional split normal, but a mixture normal)
    A = np.sqrt(2/np.pi) * 1/(sigma_1 + sigma_2)

    mask = np.abs(x - mu) < 10

    return np.where(mask, A * np.exp(upper / (2 * (sigma_1 ** 2))), A * np.exp(upper / 2 * (sigma_2 ** 2)))


def split_normal_fixed(x):

    mu = 0

    # CHECK A IS FROM FORMULA - some use the same A (CHECK IT IS THE SAME)

    sigma_1 = 0.88

    sigma_2 = 1.71

    upper = -1 * ((x - mu) ** 2)

    A = np.sqrt(2/np.pi) * 1/(sigma_1 + sigma_2)

    return np.where(np.abs(x - mu) < 10, A * np.exp(upper / (2 * (sigma_1 ** 2))), A * np.exp(upper / 2 * (sigma_2 ** 2)))


# GENERATE FIXED NUMBER OF AGNS WITHIN GIVEN ENERGY FLUX RANGE FOR FLAT EXTRAPOLATION AT LOWER ENERGY FLUXES
def pulsar_generation(pulsar_stats, energy_flux_low, energy_flux_high, sigma_1, sigma_2):

    # sigma_1 and sigma_2 are the fitted standard deviations of Gaussian distribution

    (mean_Gamma_pulsars, std_Gamma_pulsars, mean_b_pulsars, std_b_pulsars, mean_log_a_pulsars, std_log_a_pulsars,
     mean_log_flux_density_pulsars, std_log_flux_density_pulsars, mean_log_pivot_energy_pulsars,
     std_log_pivot_energy_pulsars, latitudes_pulsars) = pulsar_stats

    while True:

        # SPECTRAL PARAMETERS

        # Generate new pivot energies - in ID8, they randomly select pivot energies from a Gaussian distribution.
        # However, the distribution of pivot energies in the 4FGL follows log-normal more precise (CHECK THIS IS TRUE
        # FOR PULSARS_
        pivot_energy = np.random.lognormal(mean=mean_log_pivot_energy_pulsars, sigma=std_log_pivot_energy_pulsars,
                                             size=1)[0]

        # Generate new flux densities - log-normal for flux densities
        flux_density = np.random.lognormal(mean=mean_log_flux_density_pulsars, sigma=std_log_flux_density_pulsars,
                                             size=1)[0]

        # Gaussian recommended in ID8 - AT THE MOMENT - may change to log-normal

        # Generate new spectral slopes (Gammas)
        spectral_slope = np.random.normal(loc=mean_Gamma_pulsars, scale=std_Gamma_pulsars, size=1)[0]

        # Generate new exponential factors (as)
        exponential_factor = np.random.lognormal(loc=mean_log_a_pulsars, scale=std_log_a_pulsars, size=1)[0]

        # Generate new exponential indices (bs)
        exponential_index = np.random.normal(loc=mean_b_pulsars, scale=std_b_pulsars, size=1)[0]

        # Energy fluxes and convert energy fluxes so ergs included in units instead of photons

        energy_flux = energy_flux_pulsar(pivot_energy, flux_density, spectral_slope, exponential_index,
                                         exponential_factor) * 1.602 * 10 ** (-6)

        # SPATIAL PARAMETERS

        # l - uniform distribution assumed
        longitude = np.random.uniform(low=0, high=2 * np.pi, size=1)[0]

        # b - double Gaussian - two overlapping sampled as one

        # Randomly sample pulsar latitudes from distribution created above
        X1 = stats.Normal(mu=0, sigma=sigma_1)
        X2 = stats.Normal(mu=0, sigma=sigma_2)

        # CHANGE WEIGHTS HERE TO REFLECT MSP VS YNG

        mixture = stats.Mixture([X1, X2])

        latitude = mixture.sample(shape=(1, 1)).flatten()[0]

        if (energy_flux >= energy_flux_low) and (energy_flux < energy_flux_high):

            return np.asarray([pivot_energy, flux_density, spectral_slope, exponential_index, exponential_factor,
                               energy_flux, longitude, latitude])


def generate_mock_pulsar_catalog(pulsar_stats, num_pulsars=350, extra=10):

    # CHECK ALL PARAMETER DISTRIBUTIONS AND FIT GAUSSIAN OR LOG-NORMAL - MAYBE RESEARCH OTHER DISTRIBUTIONS IT COULD BE

    (mean_Gamma_pulsars, std_Gamma_pulsars, mean_b_pulsars, std_b_pulsars, mean_log_a_pulsars, std_log_a_pulsars,
     mean_log_flux_density_pulsars, std_log_flux_density_pulsars, mean_log_pivot_energy_pulsars,
     std_log_pivot_energy_pulsars, latitudes_pulsars) = pulsar_stats

    # SPECTRAL PARAMETERS

    # Generate new pivot energies - in ID8, they randomly select pivot energies from a Gaussian distribution. However,
    # the distribution of pivot energies in the 4FGL follows log-normal more precise (CHECK THIS IS TRUE FOR PULSARS_
    pivot_energies = np.random.lognormal(mean=mean_log_pivot_energy_pulsars, sigma=std_log_pivot_energy_pulsars,
                                         size=num_pulsars)

    # Generate new flux densities - log-normal for flux densities
    flux_densities = np.random.lognormal(mean=mean_log_flux_density_pulsars, sigma=std_log_flux_density_pulsars,
                                         size=num_pulsars)

    # Gaussian recommended in ID8 - AT THE MOMENT - may change to log-normal

    # Generate new spectral slopes (Gammas)
    spectral_slopes = np.random.normal(loc=mean_Gamma_pulsars, scale=std_Gamma_pulsars, size=num_pulsars)

    # Generate new exponential factors (as) - log-normal (CHANGED FROM ID8, which used normal/Gaussian BASED ON RESULTS
    # FOUND IN analysing_pulsar_parameters() function)
    exponential_factors = np.random.lognormal(loc=mean_log_a_pulsars, scale=std_log_a_pulsars, size=num_pulsars)

    # Generate new exponential indices (bs)
    exponential_indices = np.random.normal(loc=mean_b_pulsars, scale=std_b_pulsars, size=num_pulsars)

    # Combine into one array
    parameters = np.stack((pivot_energies, flux_densities, spectral_slopes, exponential_indices, exponential_factors), axis=-1)

    # Energy fluxes
    energy_fluxes = np.fromiter((energy_flux_pulsar(x[0], x[1], x[2], x[3], x[4]) for x in parameters), np.float64)

    # Convert energy fluxes so ergs included in units instead of photons
    energy_fluxes *= 1.602 * 10**(-6)

    # Select rows with valid energy fluxes
    mask = ~np.isnan(energy_fluxes)
    energy_fluxes = energy_fluxes[mask]
    parameters = parameters[mask]

    # Combine two arrays to create mock catalog's spectral parameters
    parameters = np.concatenate((parameters, np.array([energy_fluxes]).T), axis=1)

    # SPATIAL PARAMETERS

    # l - uniform distribution assumed
    galactic_longitudes = np.random.uniform(low=0, high=2 * np.pi, size=len(parameters))

    # b - double Gaussian - two overlapping sampled as one
    counts, bins = np.histogram(latitudes_pulsars.value, bins=100, density=True)

    bin_width = np.abs(bins[1] - bins[0])

    start_value = bins[0] + (bin_width / 2)

    x_values = [start_value + (k * bin_width) for k in range(len(bins) - 1)]

    popt, _ = curve_fit(f=split_normal, xdata=np.asarray(x_values), ydata=np.asarray(counts), bounds=([0, 0], [90, 90]))

    # Visualise distributions

    # visualising_pulsar_latitude_distributions(sigma_1=popt[0], sigma_2=popt[1], x_values=x_values, counts=counts)

    # Randomly sample pulsar latitudes from distribution created above
    X1 = stats.Normal(mu=0, sigma=popt[0])
    X2 = stats.Normal(mu=0, sigma=popt[1])

    # CHANGE WEIGHTS HERE TO REFLECT MSP VS YNG

    mixture = stats.Mixture([X1, X2])

    galactic_latitudes = mixture.sample(shape=(num_pulsars, 1)).flatten()

    # Combine two arrays to create mock catalog's spatial parameters
    parameters = np.concatenate((parameters, np.array([galactic_longitudes]).T), axis=1)

    # Combine two arrays to create mock catalog's spatial parameters
    parameters = np.concatenate((parameters, np.array([galactic_latitudes]).T), axis=1)

    # CUTOFF THRESHOLD AND LUMINOSITY FUNCTION CHECK

    # Cut at detection threshold

    parameters = parameters[~(parameters[:, 5] <= np.float64(1.0 * 10 ** (-12))), :]

    # ADD MORE SOURCES - FOLLOW LUMINOSITY FUNCTION OF 4FGL - ENSURE ENOUGH

    extra_sources = []

    for x in range(extra):

        # CHANGED TO 1.0 * 10 ** -12 as recommended by 4FGL DR4 (ID22) paper for outside galactic plane

        new_source = pulsar_generation(pulsar_stats, 1.0 * 10 ** (-12), 1000, sigma_1=popt[0], sigma_2=popt[1])

        extra_sources.append(new_source)

    parameters = np.vstack((parameters, np.asarray(extra_sources)))

    # FAINT SOURCE FLAT EXTRAPOLATION

    # Flat extrapolation of AGN - assume constant below given threshold (not Gaussian)

    # Bin data and take average of first three
    bin_edges = 10 ** np.linspace(-14, -9, 50)
    counts, _ = np.histogram(parameters[:, 5], bins=bin_edges)

    # Take average number of sources of first five bins for flat extrapolation
    first_non_empty_bin = np.nonzero(counts)[0][0]
    mean_counts_per_bin, std_counts_per_bin = (np.mean(counts[first_non_empty_bin: first_non_empty_bin + 10]),
                                               np.std(counts[first_non_empty_bin: first_non_empty_bin + 10], ddof=1))

    faint_sources = []

    our_threshold = np.argwhere(bin_edges >= 3.4 * 10 ** (-13))[0][0]

    for x in range(our_threshold, first_non_empty_bin):

        # Generate number of sources in bin - approximately flat/same as bins at peak
        counts_per_bin_flat_extrapolation = round(np.random.normal(loc=mean_counts_per_bin, scale=std_counts_per_bin,
                                                                   size=1)[0])

        for k in range(counts_per_bin_flat_extrapolation):
            new_source = pulsar_generation(pulsar_stats, bin_edges[x], bin_edges[x + 1], sigma_1=popt[0], sigma_2=popt[1])
            faint_sources.append(new_source)

    faint_sources = np.asarray(faint_sources)

    parameters = np.vstack((parameters, faint_sources))

    # CHECK LUMIN FUNCTION - NOT FINISHED

    return parameters


def pulsar_xml_writer(sources):

    # CREATE DOCUMENT

    root = minidom.Document()

    xml = root.createElement('source_library')

    xml.setAttribute('title', 'source library')

    root.appendChild(xml)

    # PARAMETERS

    pulsar_parameters = ["Prefactor", "Index1", "Scale", "Expfactor", "Index2"]

    # Ranges are taken from https://git.io/JO5FP - i.e. recommended by ID8
    ranges_of_pulsar_parameters = [("0.00000001", "1000000000.0"), ("-50000.0", "5000.0"),
                                   ("-3000000.0", "3000000000000.0"), ("-100000000", "1000000"), ("0", "20")]

    for k in range(len(sources)):

        source = root.createElement("source")

        source.setAttribute("name", "PSR_" + str(k))
        source.setAttribute("type", "PointSource")

        xml.appendChild(source)

        # SPECTRAL

        spectrum = root.createElement("spectrum")

        spectrum.setAttribute("type", "PLSuperExpCutoff2")

        for x in range(len(pulsar_parameters)):

            param = root.createElement("parameter")

            param.setAttribute("max", str(ranges_of_pulsar_parameters[x][1]))
            param.setAttribute("min", str(ranges_of_pulsar_parameters[x][0]))
            param.setAttribute("name", pulsar_parameters[x])

            if x == 0:

                # FLUX DENSITY OR PREFACTOR

                param.setAttribute("free", "1")
                param.setAttribute("scale", str(sources[k][1]))
                param.setAttribute("value", "1")

            elif x == 1:

                # SPECTRAL SLOPE OR GAMMA OR INDEX1

                param.setAttribute("free", "1")
                param.setAttribute("scale", "1.0")
                param.setAttribute("value", str(sources[k][2]))

            elif x == 2:

                # SCALE Eb OR PIVOT ENERGY

                param.setAttribute("free", "0")
                param.setAttribute("scale", "1.0")
                param.setAttribute("value", str(sources[k][0]))

            elif x == 3:

                # EXPONENTIAL FACTOR A

                param.setAttribute("free", "1")
                param.setAttribute("scale", "1.0")
                param.setAttribute("value", str(sources[k][4]))

            else:

                # INDEX2 OR B OR EXPONENTIAL INDEX

                param.setAttribute("free", "0")
                param.setAttribute("scale", "1")
                param.setAttribute("value", str(sources[k][3]))

            spectrum.appendChild(param)

        source.appendChild(spectrum)

        # SPATIAL

        spatial = root.createElement("spatialModel")

        spatial.setAttribute("type", "SkyDirFunction")

        ra = root.createElement("parameter")
        dec = root.createElement("parameter")

        ra.setAttribute("free", "0")
        ra.setAttribute("max", "360.")
        ra.setAttribute("min", "-360.")
        ra.setAttribute("name", "RA")
        ra.setAttribute("scale", "1.0")

        dec.setAttribute("free", "0")
        dec.setAttribute("max", "90.")
        dec.setAttribute("min", "-90.")
        dec.setAttribute("name", "DEC")
        dec.setAttribute("scale", "1.0")

        # Convert galactic coordinates to equatorial

        ra_dec = SkyCoord(l=sources[k][6] * u.rad, b=sources[k][7] * u.deg, frame='galactic').transform_to('icrs')
        ra.setAttribute("value", str(ra_dec.ra.to_value(u.degree)))
        dec.setAttribute("value", str(ra_dec.dec.to_value(u.degree)))

        spatial.appendChild(ra)
        spatial.appendChild(dec)

        source.appendChild(spatial)

    # SAVE

    xml_str = root.toprettyxml(indent="\t")

    save_path_file = "pulsars.xml"

    with open(save_path_file, "w") as f:
        f.write(xml_str)


    # NOT FINISHED


def agn_xml_writer(sources):

    # CREATE DOCUMENT

    root = minidom.Document()

    xml = root.createElement('source_library')

    xml.setAttribute('title', 'source library')

    root.appendChild(xml)

    # ADD SOURCES

    # Background

    # extragalactic_background = root.createElement("source")
    #
    # extragalactic_background.setAttribute("name", "EG")
    # extragalactic_background.setAttribute("type", "DiffuseSource")
    #
    # xml.appendChild(extragalactic_background)

    # AGN

    # Ranges are taken from https://git.io/JO5FP - i.e. recommended by ID8
    agn_parameters = ["norm", "alpha", "Eb", "beta"]
    ranges_of_agn_parameters = [("0.001", "1000.0"), ("-5000.0", "1000.0"), ("0.0000001", "10000000000000.0"),
                                ("-100.0", "100")]

    for k in range(len(sources)):

        source = root.createElement("source")

        source.setAttribute("name", "AGN_" + str(k))
        source.setAttribute("type", "PointSource")

        xml.appendChild(source)

        # SPECTRAL

        spectrum = root.createElement("spectrum")

        spectrum.setAttribute("type", "LogParabola")

        for x in range(len(agn_parameters)):

            param = root.createElement("parameter")
            param.setAttribute("free", "1")
            param.setAttribute("max", str(ranges_of_agn_parameters[x][1]))
            param.setAttribute("min", str(ranges_of_agn_parameters[x][0]))
            param.setAttribute("name", agn_parameters[x])

            if x % 2 != 0:
                param.setAttribute("scale", "-1.0")
                if x == 1:
                    param.setAttribute("value", str(sources[k][2]))
                else:
                    param.setAttribute("value", str(sources[k][3]))
            elif x == 2:
                param.setAttribute("scale", "1.0")
                param.setAttribute("value", str(sources[k][0]))
            else:
                # Set to value of
                param.setAttribute("scale", str(sources[k][1]))
                param.setAttribute("value", str(1))

            spectrum.appendChild(param)

        source.appendChild(spectrum)

        # SPATIAL

        spatial = root.createElement("spatialModel")

        spatial.setAttribute("type", "SkyDirFunction")

        ra = root.createElement("parameter")
        dec = root.createElement("parameter")

        ra.setAttribute("free", "0")
        ra.setAttribute("max", "360.")
        ra.setAttribute("min", "-360.")
        ra.setAttribute("name", "RA")
        ra.setAttribute("scale", "1.0")

        dec.setAttribute("free", "0")
        dec.setAttribute("max", "90.")
        dec.setAttribute("min", "-90.")
        dec.setAttribute("name", "DEC")
        dec.setAttribute("scale", "1.0")

        # Convert galactic coordinates to equatorial

        ra_dec = SkyCoord(l=sources[k][4] * u.rad, b=sources[k][5] * u.deg, frame='galactic').transform_to('icrs')
        ra.setAttribute("value", str(ra_dec.ra.to_value(u.degree)))
        dec.setAttribute("value", str(ra_dec.dec.to_value(u.degree)))

        spatial.appendChild(ra)
        spatial.appendChild(dec)

        source.appendChild(spatial)

    # FORMAT

    xml_str = root.toprettyxml(indent="\t")

    # SAVE

    save_path_file = "agns.xml"

    with open(save_path_file, "w") as f:
        f.write(xml_str)
#
print("starting...")

agn_rows, pulsar_rows = catalog_data_preparation("/Volumes/T7/data/catalog/4FGL_DR4.fit")

print("data preparation completed.")
#
# start = time.time()
#
# agns = generate_mock_agn_catalog(agn_statistics(agn_rows), 100, extra=3000)
#
# print("agns created.")
#
# agn_xml_writer(agns)
#
# print("agns saved.")
#
# pulsars = generate_mock_pulsar_catalog(pulsar_statistics(pulsar_rows), 10, extra=200)
#
# print("pulsars created")
#
# pulsar_xml_writer(pulsars)
#
# print("pulsars saved")
#
# end = time.time()
#
# print("completed.")
#
# print("TIME: " + str(end - start) + "s")

analysing_pulsar_parameters(pulsar_rows)


# REFERENCES

# Astropy Documentation - https://docs.astropy.org/en/stable/
# ID8 Paper - Identification of point sources in gamma rays using U-shaped convolutional neural networks and a data
# challenge
# Masked to Ordinary Numpy Array - https://www.w3resource.com/python-exercises/numpy/convert-masked-numpy-array-to-regul
# ar-array-with-nan.php
# Numpy Documentation - https://numpy.org/doc/stable/user/index.html
# Scipy Documentation - https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html#scipy.integrate.
# quad

