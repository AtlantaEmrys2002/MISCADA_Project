"""
Methods for generating a series of realistic simulated pulsars whose luminosity function resembles that of the 4FGL
catalog.
"""

from astropy.table import QTable
import numpy as np
from .pulsar_spectral_parameters import energy_flux_pulsar, pulsar_flux_density
from scipy.stats import cauchy, gumbel_r, lognorm, norm
from .utils import luminosity_function_calculator


def pulsar_generator(pulsar_stats, energy_flux_low: np.float64 = 0., energy_flux_high: np.float64 = 1000.):
    """Generates the realistic spectral and spatial parameters of a pulsar, sampling from the derived distributions of
    each parameter. The pulsar must have a resultant integral energy flux of between energy_flux_low and
    energy_flux_high to be returned.

    Parameters
    ----------
    pulsar_stats : ndarray
        Array of stats specifying the mean and standard deviation of the pivot energies, spectral slopes, and flux
        densities of the 4FGL pulsars, as well as all the possible values of exponential index (b).

    energy_flux_low : np.float64
        Lowest possible energy flux the simulated pulsar may have.

    energy_flux_high : np.float64
        Highest possible energy flux the simulated pulsar may have.

    """

    (mean_Gamma_pulsars, std_Gamma_pulsars, a_distribution_parameters, mean_log_pivot_energy_pulsars,
     std_log_pivot_energy_pulsars, choice_values, new_weights, gamma_noise_std, a_noise_std) = pulsar_stats

    while True:

        # SPECTRAL PARAMETERS

        # Generate new pivot energies - in ID8, they randomly select pivot energies from a Gaussian distribution.
        # However, the distribution of pivot energies in the 4FGL follows log-normal more precise
        pivot_energy = (np.random.lognormal(mean=mean_log_pivot_energy_pulsars, sigma=std_log_pivot_energy_pulsars))

        # Generate new flux densities - log-normal for flux densities was initially used then used cubic correlation
        # of log pivot energy and log flux density
        flux_density = pulsar_flux_density(pivot_energy)[0]

        # Generate new spectral slopes (Gammas) - Gaussian distribution most likely
        # (see chi-squared test)
        # spectral_slope = np.random.normal(loc=mean_Gamma_pulsars, scale=std_Gamma_pulsars)

        # NEW - with noise

        spectral_slope = (np.random.normal(loc=mean_Gamma_pulsars, scale=std_Gamma_pulsars) +
                          np.random.normal(loc=0, scale=gamma_noise_std))

        # Generate new exponential factors (as)
        # exponential_factor = gumbel_r(*a_distribution_parameters).rvs()

        # NEW - with noise

        exponential_factor = gumbel_r(*a_distribution_parameters).rvs() + np.random.normal(loc=0, scale=a_noise_std)

        # Generate new exponential indices (bs) - changed to random choice instead of ID8's Gaussian
        exponential_index = np.random.choice(choice_values, p=new_weights)

        # Generate energy fluxes
        energy_flux = energy_flux_pulsar(pivot_energy, flux_density, spectral_slope, exponential_index,
                                         exponential_factor)

        # NEW

        with np.errstate(over="ignore"):



            # Check energy flux in given range AND that the energy flux is valid
            # if (energy_flux >= energy_flux_low) and (energy_flux < energy_flux_high) and (~np.isnan(energy_flux)):
            if (energy_flux >= energy_flux_low) and (energy_flux < energy_flux_high) and (np.isfinite(energy_flux)):
                # SPATIAL PARAMETERS

                # l - uniform distribution assumed
                longitude = np.random.uniform(low=-(2 * np.pi), high=2 * np.pi)

                # b - double Gaussian - two overlapping sampled as one

                # Randomly sample pulsar latitudes from distribution created above
                latitude = cauchy(loc=0, scale=np.float64(0.018077045649988577)).rvs()

                # Clip to correct range
                latitude = np.clip(latitude, a_min=-np.pi / 2, a_max=np.pi / 2)

                return np.asarray([pivot_energy, flux_density, spectral_slope, exponential_index, exponential_factor,
                                   energy_flux, longitude, latitude])


def luminosity_function_pulsar(catalog: str, detection_threshold):
    """ Determines the number of sources to generate within a given energy flux - i.e. ensures that the luminosity
    function of simulated pulsar resembles that of the 4FGL.

    Parameters
    ----------
    catalog: str
        Name of file in which FITS-formatted catalog resides.
    detection_threshold: np.float64
         Integral energy flux below which the luminosity function of faint sources must be extrapolated.

    """
    # Used to build luminosity function of the simulated AGNs

    # Read 4FGL Catalog
    catalog = QTable.read(catalog, format='fits', hdu=1)['CLASS1', 'Energy_Flux100']

    # Reformat columns
    catalog['CLASS1'] = np.asarray([k.decode('utf-8').strip().lower() for k in catalog['CLASS1'].value.filled('-')])

    # Select all rows that describe pulsars
    pulsar_mask = (catalog["CLASS1"] == "psr")

    pulsars = catalog[pulsar_mask]

    energy_fluxes_4fgl = pulsars['Energy_Flux100'].value

    # Number of sources in the lowest energy flux bin - different values for pulsars
    n_min = np.random.uniform(low=15, high=25)

    return luminosity_function_calculator(energy_fluxes_4fgl=energy_fluxes_4fgl, n_min=n_min,
                                          detection_threshold=detection_threshold, num_bins=10)


def generate_mock_pulsar_catalog(catalog: str, pulsars, noise_params, detection_threshold):
    """Generates an array of simulated pulsar sources with realistic energy spectra and spatial locations within the
    sky.

    Parameters
    ----------
    catalog: str
        Name of file in which FITS-formatted 4FGL catalog is stored.
    pulsars
        Dataframe with parameter values for all pulsars in 4FGL catalog.
    detection_threshold
        Integral energy flux below which the luminosity function of faint sources must be extrapolated.

    Returns
    -------

    ndarray
        n x 9 array detailing the 9 spectral and spatial parameters of each of the n pulsars.

    """
    # Select pivot energy values
    pivot_energies = pulsars['Pivot_Energy'].value

    # Select Gamma values and convert from masked to ordinary numpy array
    Gammas = pulsars['PLEC_IndexS'].data.filled(np.nan)

    # NEW NOISE

    gamma_noise = noise_params["Unc_PLEC_IndexS"].data.filled(np.nan)[
        ~np.isnan(noise_params["Unc_PLEC_IndexS"].data.filled(np.nan))]

    gamma_noise_std = np.sqrt(np.sum(gamma_noise ** 2) / gamma_noise.shape[0])



    # Select exponential indices
    b_values = pulsars['PLEC_Exp_Index'].data.filled(np.nan)

    # Select index a values
    a_values = pulsars['PLEC_ExpfactorS'].data

    # NEW NOISE

    a_noise = noise_params["Unc_PLEC_ExpfactorS"].data.filled(np.nan)[
        ~np.isnan(noise_params["Unc_PLEC_ExpfactorS"].data.filled(np.nan))]

    a_noise_std = np.sqrt(np.sum(a_noise ** 2) / a_noise.shape[0])

    # Log-normal for Gamma even though says Gaussian in ID8
    mean_Gamma_pulsars, std_Gamma_pulsars = np.nanmean(Gammas), np.nanstd(Gammas, ddof=1)

    # Log-normal for values (ID8 recommends Gaussian)
    a_distribution_parameters = gumbel_r.fit(a_values)

    # Log-normal for pivot energy even though ID8 says Gaussian
    log_pivot_energies = np.log(pivot_energies)

    mean_log_pivot_energy_pulsars, std_log_pivot_energy_pulsars = (np.nanmean(log_pivot_energies),
                                                                   np.nanstd(log_pivot_energies, ddof=1))

    # Generate new exponential indices (bs) by selecting from values available (instead of Gaussian recommended by ID8)
    unique_values = np.unique_all(b_values)
    choice_values = unique_values.values

    # Divide by number of pulsars in 4FGL to get probabilities
    new_weights = unique_values.counts / b_values.shape[0]

    pulsar_stats = (mean_Gamma_pulsars, std_Gamma_pulsars, a_distribution_parameters, mean_log_pivot_energy_pulsars,
                    std_log_pivot_energy_pulsars, choice_values, new_weights, gamma_noise_std, a_noise_std)

    # CREATE NEW SOURCES

    # Determine how many AGNs to generate based on 4FGL luminosity function
    target_counts, target_bin_intervals, target_peak = (
        luminosity_function_pulsar(catalog=catalog, detection_threshold=detection_threshold))

    num_target_intervals = target_bin_intervals.shape[0]

    actual_counts = np.zeros_like(target_counts)

    parameters = []

    while (actual_counts[0] < target_counts[0]) and np.any(np.less(actual_counts[target_peak:],
                                                                   target_counts[target_peak:])):

        # Due to uncomplimentary functionality - need max of intervals[:-1] - see reference to Digitize Error
        new_source = pulsar_generator(pulsar_stats, energy_flux_low=np.min(target_bin_intervals),
                                      energy_flux_high=np.max(target_bin_intervals[:-1]))

        idx = np.digitize(new_source[5], target_bin_intervals)

        if 0 < idx < num_target_intervals:

            if actual_counts[idx] < target_counts[idx]:
                parameters.append(np.array(new_source))
                actual_counts[idx] += 1

                print("PULSAR: {}".format(len(parameters)))

        else:

            parameters.append(np.array(new_source))
            actual_counts[idx] += 1

            print("PULSAR: {}".format(len(parameters)))

    parameters = np.array(parameters)

    return parameters

# REFERENCES

# Exclude NaNs - https://stackoverflow.com/questions/17126543/numpy-array-get-the-subset-slice-of-an-array-which-is-not
# -nan
