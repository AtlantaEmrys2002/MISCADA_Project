# This method is adapted from ID4. All code is my own (except where indicated), but an official and alternative Python
# implementation provided by the authors can be found here - https://git.io/JO5FP. I did not consult it when creating my
# version. The reasons I chose to reimplement
# are as follows:
# 1) I wanted to find out how to simulate Fermi data and the description in the paper provided a step-by-step method.
# 2) I hoped to improve upon their implementation performance-wise - by implementing from scratch, I am familiar with
# the code and can improve it more easily.
# 3) I wanted to understand the method so that I could reimplement the code in C/C++ to make use of parallel processing
# and GPUs.
# 4) Once I had reimplemented (possibly in two languages) and made optimisations, I could then improve simulation
# techniques and bring in new ideas, e.g. time data, light curves, etc.

# LIBRARIES
import numpy as np
from scipy.integrate import quad
from astropy.table import QTable
from astropy import units as u


# SPECTRAL MODELS

def agn_spectral_model(E, E_0, F_0, alpha, beta):

    division = E/E_0

    exponent = - alpha - (beta * np.log(division))

    dF_dE = F_0 * np.power(division, exponent)

    return E * dF_dE


def pulsar_spectral_model(E, F_0, E_0, Gamma, a, b):

    exponent = a * (np.power(E_0, b) - np.power(E, b))

    dF_dE = F_0 * np.power((E/E_0), -Gamma) * np.exp(exponent)

    return E * dF_dE


def energy_flux_agn(pivot_energy, flux_density, spectral_slope, curvature):

    # Integrate over 0.1 - 100 GeV
    energy = quad(agn_spectral_model, 0.1, 100, args=(pivot_energy, flux_density, spectral_slope, curvature))[0]

    return energy


def s1_agn(pivot_energy, flux_density, spectral_slope, curvature):


    # Integrate above 1 GeV
    s1 = quad(agn_spectral_model, 1, np.inf, args=(pivot_energy, flux_density, spectral_slope, curvature))[0]

    return s1


def s10_agn(pivot_energy, flux_density, spectral_slope, curvature):

    # Integrate above 1 GeV
    s1 = quad(agn_spectral_model, 10, np.inf, args=(pivot_energy, flux_density, spectral_slope, curvature))[0]

    return s1


def energy_flux_pulsar(pivot_energy, flux_density, spectral_slope, exponential_index, exponential_factor):

    # Integrate over 0.1 - 100 GeV
    energy = quad(pulsar_spectral_model, 0.1, 100, args=(pivot_energy, flux_density, spectral_slope,
                                                         exponential_index, exponential_factor))[0]

    return energy


def catalog_data_preparation(file_name):

    # Access 4FGL catalog - note, file originally called gll_psc_v35.fit
    # Read catalog data and format as astropy QTable (allowing for units to be associated with each column, if necessary
    # Assume catalog data conforms to standard NASA format
    catalog = QTable.read(file_name, format='fits', hdu=1)

    # Select relevant columns
    columns = ('Pivot_Energy', 'LP_Flux_Density', 'PLEC_Flux_Density', 'LP_Index', 'LP_beta', 'PLEC_IndexS', 'PLEC_Exp_Index',
               'PLEC_ExpfactorS', 'CLASS1')
    catalog = catalog[columns]

    # Reformat CLASS1 column - remove empty spaces and make all lower case
    catalog['CLASS1'].name = 'Prev_CLASS1'
    catalog['CLASS1'] = np.asarray([k.strip().lower() for k in catalog['Prev_CLASS1']])
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
    flux_densities = agns['LP_Flux_Density'].to(u.ph / (u.cm * u.cm * u.GeV * u.s)).value

    log_flux_densities = np.log(flux_densities)

    mean_alpha, std_alpha = np.nanmean(alphas), np.nanstd(alphas, ddof=1)

    mean_pivot_energy, std_pivot_energy = np.nanmean(pivot_energies), np.nanstd(pivot_energies, ddof=1)

    mean_log_flux_density, std_log_flux_density = np.nanmean(log_flux_densities), np.nanstd(log_flux_densities, ddof=1)

    return (mean_alpha, std_alpha, mean_pivot_energy, std_pivot_energy, mean_log_flux_density, std_log_flux_density,
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

    mean_a, std_a = np.nanmean(a_values), np.nanstd(a_values, ddof=1)

    mean_log_flux_density, std_log_flux_density = np.nanmean(log_flux_densities), np.nanstd(log_flux_densities)

    mean_pivot_energy, std_pivot_energy = np.nanmean(pivot_energies), np.nanstd(pivot_energies, ddof=1)

    return (mean_Gamma, std_Gamma, mean_b, std_b, mean_a, std_a, mean_log_flux_density, std_log_flux_density,
            mean_pivot_energy, std_pivot_energy)


def generate_mock_agn_catalog(agn_stats, num_agns=3000):

    (mean_alpha_agn, std_alpha_agn, mean_pivot_energy_agn, std_pivot_energy_agn, mean_log_flux_density_agn,
     std_log_flux_density_agn, betas_agn) = agn_stats

    # Generate new pivot energies
    pivot_energies = np.random.normal(loc=mean_pivot_energy_agn, scale=std_pivot_energy_agn, size=num_agns)

    # Generate new flux densities - log-normal for flux densities
    log_flux_densities = np.random.normal(loc=mean_log_flux_density_agn, scale=std_log_flux_density_agn, size=num_agns)
    flux_densities = np.exp(log_flux_densities)

    # Generate new spectral slopes (alphas)
    spectral_slopes = np.random.normal(loc=mean_alpha_agn, scale=std_alpha_agn, size=num_agns)

    # Generate new curvatures by directly sampling 4FGL
    betas = np.random.choice(betas_agn, size=num_agns, replace=True)

    # Combine into one array
    parameters = np.stack((pivot_energies, flux_densities, spectral_slopes, betas), axis=-1)

    # pivot_energy, flux_density, spectral_slope, curvature

    # Energy fluxes
    energy_fluxes = np.fromiter((energy_flux_agn(x[0], x[1], x[2], x[3]) for x in parameters), np.float64)

    print(energy_fluxes)

    print(energy_fluxes.shape)

    x = energy_fluxes[~np.isnan(energy_fluxes)]

    print(x.shape)

def generate_mock_pulsar_catalog(pulsar_stats, num_pulsars=250):

    (mean_Gamma_pulsars, std_Gamma_pulsars, mean_b_pulsars, std_b_pulsars, mean_a_pulsars, std_a_pulsars,
     mean_log_flux_density_pulsars, std_log_flux_density_pulsars, mean_pivot_energy_pulsars, std_pivot_energy_pulsars) \
        = pulsar_stats


agn_rows, pulsar_rows = catalog_data_preparation("/Volumes/T7/data/catalog/4FGL_DR4.fit")

pulsar_statistics(pulsar_rows)


generate_mock_agn_catalog(agn_statistics(agn_rows), 1000)

# generate_mock_pulsar_catalog(pulsar_statistics(pulsar_rows), 10)




# REFERENCES

# Astropy Documentation - https://docs.astropy.org/en/stable/
# ID8 Paper - Identification of point sources in gamma rays using U-shaped convolutional neural networks and a data
# challenge
# Masked to Ordinary Numpy Array - https://www.w3resource.com/python-exercises/numpy/convert-masked-numpy-array-to-regul
# ar-array-with-nan.php
# Scipy Documentation - https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html#scipy.integrate.
# quad

