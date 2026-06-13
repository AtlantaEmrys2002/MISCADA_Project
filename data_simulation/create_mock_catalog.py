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
from scipy.integrate import quad
from astropy.table import QTable
from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import root_mean_squared_error

# Relative imports
from analysis.goodness_of_fit import chi_squared_test, kolmogorov_smirnov_test
from analysis.visualisation import (agn_luminosity_function, correlation_matrices, plot_parameter_distributions,
                                    plot_parameter_relationships)
from physical_properties.spectral_models import agn_spectral_model
from physical_properties.agn_parameters import energy_flux_agn, agn_flux_density, agn_spectral_slope
from physical_properties.pulsar_parameters import energy_flux_pulsar

# VISUALISATIONS

# IN PROCESS OF MOVING INTO SEPARATE FILE
def spatial_visualisations(galactic_longitudes, galactic_latitudes, num_sources, source_type):

    xs, ys = [], []

    for k in range(num_sources):

        ra_dec = SkyCoord(l=galactic_longitudes[k] * u.rad, b=galactic_latitudes[k] * u.rad,
                          frame='galactic').transform_to('icrs')

        xs.append(ra_dec.ra.to_value(u.degree))
        ys.append(ra_dec.dec.to_value(u.degree))

    fig, ax = plt.subplots(figsize=(8, 4.2), subplot_kw=dict(projection="aitoff"))
    ax.set_title("Distribution of Simulated " + source_type + " on the Sky", pad=20)
    ax.grid(True)
    ax.scatter(xs, ys, marker='o', s=2, alpha=0.3)
    fig.subplots_adjust(top=0.95, bottom=0.0)
    plt.show()


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

    # mean_Gamma, std_Gamma = np.nanmean(Gammas), np.nanstd(Gammas, ddof=1)

    log_Gammas = np.log(Gammas)

    mean_log_Gamma, std_log_Gamma = np.nanmean(log_Gammas), np.nanstd(log_Gammas, ddof=1)

    mean_b, std_b = np.nanmean(b_values), np.nanstd(b_values, ddof=1)

    log_a_values = np.log(a_values)

    # mean_a, std_a = np.nanmean(a_values), np.nanstd(a_values, ddof=1)

    mean_log_a, std_log_a = np.nanmean(log_a_values), np.nanstd(log_a_values, ddof=1)

    mean_log_flux_density, std_log_flux_density = np.nanmean(log_flux_densities), np.nanstd(log_flux_densities)

    # mean_pivot_energy, std_pivot_energy = np.nanmean(pivot_energies), np.nanstd(pivot_energies, ddof=1)

    log_pivot_energies = np.log(pivot_energies)

    mean_log_pivot_energy, std_log_pivot_energy = np.nanmean(log_pivot_energies), np.nanstd(log_pivot_energies, ddof=1)

    pulsar_latitudes = pulsars['GLAT']

    # return (mean_Gamma, std_Gamma, mean_b, std_b, mean_log_a, std_log_a, mean_log_flux_density, std_log_flux_density,
    #         mean_log_pivot_energy, std_log_pivot_energy, pulsar_latitudes)

    # return (mean_Gamma, std_Gamma, b_values, len(pulsars), mean_log_a, std_log_a, mean_log_flux_density, std_log_flux_density,
    #         mean_log_pivot_energy, std_log_pivot_energy, pulsar_latitudes)

    return (mean_log_Gamma, std_log_Gamma, b_values, len(pulsars), mean_log_a, std_log_a, mean_log_flux_density, std_log_flux_density,
    mean_log_pivot_energy, std_log_pivot_energy, pulsar_latitudes)


# GENERATE FIXED NUMBER OF AGNS WITHIN GIVEN ENERGY FLUX RANGE FOR FLAT EXTRAPOLATION AT LOWER ENERGY FLUXES
def agn_generation(agn_stats, energy_flux_low, energy_flux_high):

    (mean_alpha_agn, std_alpha_agn, mean_log_pivot_energy_agn, std_log_pivot_energy_agn, mean_log_flux_density_agn,
     std_log_flux_density_agn, betas_agn) = agn_stats

    while True:

        # Generate new pivot energy
        # pivot_energy = np.random.normal(loc=mean_pivot_energy_agn, scale=std_pivot_energy_agn, size=1)[0]

        pivot_energy = np.random.lognormal(mean=mean_log_pivot_energy_agn, sigma=std_log_pivot_energy_agn,
                                             size=1)[0]

        # Generate new flux density - log-normal for flux densities
        flux_density_ln = np.random.lognormal(mean=mean_log_flux_density_agn, sigma=std_log_flux_density_agn, size=1)[0]

        # FLUX DENSITIES AND PIVOT ENERGY ARE DEPENDENT - THEREFORE FITTED RELATIONSHIP AND CALCULATE
        # FLUX DENSITY AS SUCH (THEREFORE, CHANGED SO GENERATION IS RELATED to PIVOt ENERGY)
        flux_density = agn_flux_density(pivot_energy)[0]

        # Generate new spectral slope (alpha)
        # spectral_slope = np.random.normal(loc=mean_alpha_agn, scale=std_alpha_agn, size=1)[0]

        # FLUX DENSITY, PIVOT ENERGY, AND SPECTRAL SLOPE ARE DEPENDENT
        spectral_slope = agn_spectral_slope(pivot_energy)[0]

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
    # flux_densities = np.random.lognormal(mean=mean_log_flux_density_agn, sigma=std_log_flux_density_agn, size=num_agns)

    # FLUX DENSITIES AND PIVOT ENERGY ARE DEPENDENT - THEREFORE FITTED RELATIONSHIP AND CALCULATE
    # FLUX DENSITY AS SUCH (THEREFORE, CHANGED SO GENERATION IS RELATED to PIVOt ENERGY)
    flux_densities = agn_flux_density(pivot_energies)

    # Generate new spectral slopes (alphas)
    # spectral_slopes = np.random.normal(loc=mean_alpha_agn, scale=std_alpha_agn, size=num_agns)

    spectral_slopes = agn_spectral_slope(pivot_energies)

    # Generate new curvatures by directly sampling 4FGL
    betas = np.random.choice(betas_agn, size=num_agns, replace=True)

    # Combine into one array
    parameters = np.stack((pivot_energies, flux_densities, spectral_slopes, betas), axis=-1)

    # Energy fluxes
    energy_fluxes = np.fromiter((energy_flux_agn(x[0], x[1], x[2], x[3]) for x in parameters), np.float64)

    # Convert energy fluxes so ergs included in units instead of photons - see ID43
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

        # print(x + 1)

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

    agn_luminosity_function("/Volumes/T7/data/catalog/4FGL_DR4.fit", parameters[:, 4], directory="./plots/analysis")

    # CHECK SIMULATED FLUX DENSITIES AND PIVOT ENERGIES HAVE SAME CORRELATION AS IN 4FGL
    # plt.title('Simulated $F_0$ against $E_0$')
    # plt.xlabel('log $E_0$')
    # plt.ylabel('log $F_{0, AGN}$')
    # plt.scatter(np.log(parameters[:, 0]), np.log(parameters[:, 1]), s=2)
    # plt.show()
    #
    # # CHECK SIMULATED PIVOT ENERGIES AND SPECTRAL INDICES (ALPHAS) HAVE SAME CORRELATION AS IN 4FGL
    # plt.title('Simulated $\\alpha$ against $E_0$')
    # plt.xlabel('log $E_0$')
    # plt.ylabel('log $\\alpha$')
    # plt.scatter(np.log(parameters[:, 0]), np.log(parameters[:, 2]), s=2)
    # plt.show()

    print(np.max(galactic_latitudes))
    print(np.max(galactic_longitudes))

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


# GENERATE FIXED NUMBER OF AGNS WITHIN GIVEN ENERGY FLUX RANGE FOR FLAT EXTRAPOLATION AT LOWER ENERGY FLUXES
def pulsar_generation(pulsar_stats, energy_flux_low, energy_flux_high, sigma_1, sigma_2):

    # sigma_1 and sigma_2 are the fitted standard deviations of Gaussian distribution

    (mean_log_Gamma_pulsars, std_log_Gamma_pulsars, b_values, len_pulsars, mean_log_a_pulsars, std_log_a_pulsars,
     mean_log_flux_density_pulsars, std_log_flux_density_pulsars, mean_log_pivot_energy_pulsars,
     std_log_pivot_energy_pulsars, latitudes_pulsars) = pulsar_stats

    unique_values = np.unique_all(b_values)
    choice_values = unique_values.values
    new_weights = unique_values.counts/len_pulsars

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

        # Generate new spectral slopes (Gammas) - CHANGED FROM GAUSSIAN TO LOG-NORMAL BASED ON CHI SQUARED GOODNESS OF FIT
        # ANALYSIS (THIS IS A CHANGE FROM ID8, WHICH RECOMMENDED GAUSSIAN)
        # spectral_slope = np.random.normal(loc=mean_Gamma_pulsars, scale=std_Gamma_pulsars, size=1)[0]
        spectral_slope = np.random.lognormal(mean=mean_log_Gamma_pulsars, sigma=std_log_Gamma_pulsars, size=1)[0]

        # Generate new exponential factors (as)
        exponential_factor = np.random.lognormal(mean=mean_log_a_pulsars, sigma=std_log_a_pulsars, size=1)[0]

        # Generate new exponential indices (bs)
        # exponential_index = np.random.normal(loc=mean_b_pulsars, scale=std_b_pulsars, size=1)[0]
        exponential_index = np.random.choice(choice_values, p=new_weights)

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

    # (mean_Gamma_pulsars, std_Gamma_pulsars, b_values, len_pulsars, mean_log_a_pulsars, std_log_a_pulsars,
    #  mean_log_flux_density_pulsars, std_log_flux_density_pulsars, mean_log_pivot_energy_pulsars,
    #  std_log_pivot_energy_pulsars, latitudes_pulsars) = pulsar_stats

    (mean_log_Gamma_pulsars, std_log_Gamma_pulsars, b_values, len_pulsars, mean_log_a_pulsars, std_log_a_pulsars,
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
    # spectral_slopes = np.random.normal(loc=mean_Gamma_pulsars, scale=std_Gamma_pulsars, size=num_pulsars)

    spectral_slopes = np.random.lognormal(mean=mean_log_Gamma_pulsars, sigma=std_log_Gamma_pulsars, size=num_pulsars)

    # Generate new exponential factors (as) - log-normal (CHANGED FROM ID8, which used normal/Gaussian BASED ON RESULTS
    # FOUND IN analysing_pulsar_parameters() function)
    exponential_factors = np.random.lognormal(mean=mean_log_a_pulsars, sigma=std_log_a_pulsars, size=num_pulsars)

    # Generate new exponential indices (bs) by selecting from values available
    # exponential_indices = np.random.normal(loc=mean_b_pulsars, scale=std_b_pulsars, size=num_pulsars)
    unique_values = np.unique_all(b_values)
    choice_values = unique_values.values
    # Divide by number of pulsars in 4FGL to get probabilities
    new_weights = unique_values.counts/len_pulsars

    exponential_indices = np.random.choice(choice_values, p=new_weights, size=num_pulsars)

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

    # # Convert to degrees
    # galactic_longitudes = np.rad2deg(galactic_longitudes)

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

    # CONVERT TO DEGREES
    #
    # print(galactic_latitudes)
    #
    # galactic_latitudes = np.deg2rad(galactic_latitudes)

    # print('HELLO')
    #
    # print(galactic_latitudes)
    #
    # print("HELLO_2")


    # Combine two arrays to create mock catalog's spatial parameters
    parameters = np.concatenate((parameters, np.array([galactic_longitudes]).T), axis=1)

    # Combine two arrays to create mock catalog's spatial parameters
    parameters = np.concatenate((parameters, np.array([galactic_latitudes]).T), axis=1)

    # CUTOFF THRESHOLD AND LUMINOSITY FUNCTION CHECK

    # Cut at detection threshold

    # CHECK THIS AND ADD BACK IN - NEED CUT OFF THRESHOLD!!!!!!!!!!!!!!!!!!!
    #
    # parameters = parameters[~(parameters[:, 5] <= np.float64(1.0 * 10 ** (-12))), :]
    #
    # # ADD MORE SOURCES - FOLLOW LUMINOSITY FUNCTION OF 4FGL - ENSURE ENOUGH
    #
    # extra_sources = []
    #
    # for x in range(extra):
    #
    #     print(x + 1)
    #
    #     # CHANGED TO 1.0 * 10 ** -12 as recommended by 4FGL DR4 (ID22) paper for outside galactic plane
    #
    #     new_source = pulsar_generation(pulsar_stats, 1.0 * 10 ** (-12), 1000, sigma_1=popt[0], sigma_2=popt[1])
    #
    #     extra_sources.append(new_source)
    #
    # parameters = np.vstack((parameters, np.asarray(extra_sources)))
    #
    # # FAINT SOURCE FLAT EXTRAPOLATION
    #
    # # Flat extrapolation of AGN - assume constant below given threshold (not Gaussian)
    #
    # # Bin data and take average of first three
    # bin_edges = 10 ** np.linspace(-14, -9, 50)
    # counts, _ = np.histogram(parameters[:, 5], bins=bin_edges)
    #
    # # Take average number of sources of first five bins for flat extrapolation
    # first_non_empty_bin = np.nonzero(counts)[0][0]
    # mean_counts_per_bin, std_counts_per_bin = (np.mean(counts[first_non_empty_bin: first_non_empty_bin + 10]),
    #                                            np.std(counts[first_non_empty_bin: first_non_empty_bin + 10], ddof=1))
    #
    # faint_sources = []
    #
    # our_threshold = np.argwhere(bin_edges >= 3.4 * 10 ** (-13))[0][0]
    #
    # for x in range(our_threshold, first_non_empty_bin):
    #
    #     # Generate number of sources in bin - approximately flat/same as bins at peak
    #     counts_per_bin_flat_extrapolation = round(np.random.normal(loc=mean_counts_per_bin, scale=std_counts_per_bin,
    #                                                                size=1)[0])
    #
    #     for k in range(counts_per_bin_flat_extrapolation):
    #         new_source = pulsar_generation(pulsar_stats, bin_edges[x], bin_edges[x + 1], sigma_1=popt[0], sigma_2=popt[1])
    #         faint_sources.append(new_source)
    #
    # faint_sources = np.asarray(faint_sources)
    #
    # parameters = np.vstack((parameters, faint_sources))

    # CHECK LUMIN FUNCTION - NOT FINISHED!!!!!!

    return parameters


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
    correlation_matrices(agns, source_type="AGN", directory=directory)

    # Pulsars

    # Plot parameters against one another to visualise relationships
    plot_parameter_relationships(pulsars, source_type="Pulsar", directory=directory)

    # Plot pulsar parameter correlation matrices
    correlation_matrices(pulsars, source_type="Pulsar", directory=directory)


agn_rows, pulsar_rows = catalog_data_preparation("/Volumes/T7/data/catalog/4FGL_DR4.fit")

# SELECT APPROPRIATE COLUMNS (MOVE THIS INTO DATA PREP IF DON'T ENCOUNTER ANY PROBLEMS)

agn_rows = agn_rows['LP_Flux_Density', 'Pivot_Energy', 'LP_Index', 'LP_beta']
pulsar_rows = pulsar_rows['PLEC_Flux_Density', 'Pivot_Energy', 'PLEC_IndexS', 'PLEC_Exp_Index', 'PLEC_ExpfactorS', 'GLAT']

# Analyse parameters, their distributions, and their correlations

# analysis(agn_rows, pulsar_rows)

# fitting_agn_pivot_energy_flux_density_relation(agn_rows.copy())

# fitting_agn_pivot_energy_spectral_slope_relation(agn_rows.copy())

agns = generate_mock_agn_catalog(agn_statistics(agn_rows.copy()), num_agns=100, extra=30)

print(agns[0])

pulsars = generate_mock_pulsar_catalog(pulsar_statistics(pulsar_rows.copy()), 10, extra=5)

print(pulsars[0])

# spatial_visualisations(pulsars[:, 6], pulsars[:, 7], num_sources=len(pulsars), source_type='Pulsars')


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

