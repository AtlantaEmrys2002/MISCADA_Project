"""
Methods for generating a series of realistic simulated AGN whose luminosity function resembles that of the 4FGL catalog.
"""

from .agn_spectral_parameters import agn_flux_density, agn_spectral_slope, energy_flux_agn
from astropy.table import QTable
import numpy as np
from scipy.stats import gumbel_r, lognorm
from .utils import luminosity_function_calculator


def agn_generator(agn_stats, energy_flux_low: np.float64 = 0., energy_flux_high: np.float64 = 1000.):
    """
    Generates the realistic spectral and spatial parameters of an AGN, sampling from the derived distributions of each
    parameter. The AGN must have a resultant integral energy flux of between energy_flux_low and energy_flux_high to be
    returned.

    Parameters
    ----------
    agn_stats : ndarray
        Array of stats specifying the mean and standard deviation of the pivot energies of the 4FGL AGNs, as well as all
        the possible values of spectral curvature (beta).

    energy_flux_low:
        Lowest possible energy flux the simulated AGN may have.

    energy_flux_high
        Highest possible energy flux the simulated pulsar may have.

    """
    # Default is effectively source with any energy flux

    (mean_log_pivot_energy_agn, std_log_pivot_energy_agn, beta_dist_params, beta_noise_std) = agn_stats

    while True:

        # SPECTRAL PARAMETERS

        # Generate new pivot energy

        # Although ID8 stated that they only randomly sampled flux densities according to a log-normal distribution, a
        # log-normal distribution fits pivot energies much better than their recommended Gaussian (and this makes sense
        # as differential flux density and pivot energy are correlated). Therefore, I have changed the way pivot
        # energies are randomly generated (and now use a log-normal distribution). There is no uncertainty in PE in 4FGL
        pivot_energy = (np.random.lognormal(mean=mean_log_pivot_energy_agn, sigma=std_log_pivot_energy_agn))

        # Flux densities and pivot energies are correlated and depend on one another - therefore, I fitted a polynomial
        # relationship to the log of both values and add noise to improve data realism instead of randomly sampling
        # flux densities.
        flux_density = agn_flux_density(pivot_energy)[0]

        # Generate new spectral slope (alpha)

        # Like flux densities, spectral slopes are dependent on pivot energies, therefore, fitted relationship to
        # spectral slope - pivot energy relationship (there is a paper that indicates that alpha is linearly dependent
        # on the log of pivot energy, so chose to use that one)
        spectral_slope = agn_spectral_slope(pivot_energy)[0]

        # Generate new curvature by directly sampling 4FGL
        # beta = gumbel_r(*beta_dist_params).rvs()


        # ADD NOISE

        beta = gumbel_r(*beta_dist_params).rvs() + np.random.normal(loc=0, scale=beta_noise_std)


        # Calculate energy flux of source
        energy_flux = energy_flux_agn(pivot_energy, flux_density, spectral_slope, beta)

        # IS NEW


        with np.errstate(over="ignore"):

            if (energy_flux >= energy_flux_low) and (energy_flux < energy_flux_high) and (np.isfinite(energy_flux)):
                # SPATIAL PARAMETERS

                # longitude
                longitude = np.random.uniform(low=-(2 * np.pi), high=2 * np.pi)

                # latitude
                sin_galactic_latitudes = np.random.uniform(low=-1, high=1)
                latitude = np.arcsin(sin_galactic_latitudes)

                return np.array([pivot_energy, flux_density, spectral_slope, beta, energy_flux, longitude, latitude])


def luminosity_function_agn(catalog: str, detection_threshold: np.float64):
    """ Determines the number of sources to generate within a given energy flux - i.e. ensures that the luminosity
    function of simulated AGN resembles that of the 4FGL.

    Parameters
    ----------
    catalog: str
        Name of file in which FITS-formatted catalog resides.
    detection_threshold: np.float64
         Integral energy flux below which the luminosity function of faint sources must be extrapolated.

    """
    # Read 4FGL Catalog
    catalog = QTable.read(catalog, format='fits', hdu=1)['CLASS1', 'Energy_Flux100']

    # Reformat columns
    catalog['CLASS1'] = np.asarray([k.decode('utf-8').strip().lower() for k in catalog['CLASS1'].value.filled('-')])

    # Select all rows that describe AGN
    agn_mask = np.isin(catalog['CLASS1'].data, np.array(['bcu', 'sey', 'ssrq', 'bll', 'fsrq', 'rdg', 'nlsy1', 'agn']))
    agns = catalog[agn_mask]

    energy_fluxes_4fgl = agns['Energy_Flux100'].value

    # Number of sources in the lowest energy flux bin
    n_min = np.random.uniform(low=50, high=250)

    return luminosity_function_calculator(energy_fluxes_4fgl=energy_fluxes_4fgl, n_min=n_min,
                                          detection_threshold=detection_threshold)


def generate_mock_agn_catalog(catalog: str, agn_data, noise_params,
                              detection_threshold: np.float64 = np.float64(1.0 * 10 ** (-12))):
    """Generates an array of simulated AGN sources with realistic energy spectra and spatial locations within the sky.

    Parameters
    ----------
    catalog: str
        Name of file in which FITS-formatted 4FGL catalog is stored.
    agn_data
        Dataframe with parameter values for all AGN in 4FGL catalog.
    detection_threshold
        Integral energy flux below which the luminosity function of faint sources must be extrapolated.

    Returns
    -------

    ndarray
        m x 7 array detailing the 7 spectral and spatial parameters of each of the m AGNs.

    """
    # Changed threshold from 2.0 * 10 ** -12 TO 1.0 * 10 ** -12 as threshold recommended by 4FGL DR4 (ID22) paper for
    # outside galactic plane and detection threshold has decreased since ID8 was published

    # SPECTRAL PARAMETERS

    # Select beta values and convert from masked to ordinary numpy array
    betas_agn = agn_data['LP_beta'].data.filled(np.nan)

    # NEW - Gumbel distribution

    beta_distribution_params = gumbel_r.fit(betas_agn[~np.isnan(betas_agn)])


    # NEW - NOISE

    noise_params = noise_params.data.filled(np.nan)[~np.isnan(noise_params.data.filled(np.nan))]

    beta_noise_std = np.sqrt(np.sum(noise_params ** 2) / noise_params.shape[0])

    # Select pivot energy values
    pivot_energies = agn_data['Pivot_Energy'].value

    # Calculate mean and standard deviation of log of pivot energies for random sampling
    log_pivot_energies = np.log(pivot_energies)
    mean_log_pivot_energy_agn, std_log_pivot_energy_agn = (np.nanmean(log_pivot_energies),
                                                           np.nanstd(log_pivot_energies, ddof=1))

    # CREATE NEW SOURCES

    # Determine how many AGNs to generate based on 4FGL luminosity function
    target_counts, target_bin_intervals, target_peak = luminosity_function_agn(catalog=catalog,
                                                                               detection_threshold=detection_threshold)

    num_intervals = target_bin_intervals.shape[0]

    actual_counts = np.zeros_like(target_counts)

    parameters = []

    print("TARGET COUNTS: {}".format(np.sum(target_counts)))

    while (actual_counts[0] < target_counts[0]) and np.any(np.less(actual_counts[target_peak:],
                                                                   target_counts[target_peak:])):

        # Due to uncomplimentary functionality - need max of intervals[:-1] - see reference to Digitize Error
        new_source = agn_generator((mean_log_pivot_energy_agn, std_log_pivot_energy_agn,
                                    beta_distribution_params, beta_noise_std),
                                   energy_flux_low=np.min(target_bin_intervals),
                                   energy_flux_high=np.max(target_bin_intervals[:-1]))

        idx = np.digitize(new_source[4], target_bin_intervals)

        if 0 < idx < num_intervals:

            if actual_counts[idx] < target_counts[idx]:
                parameters.append(np.array(new_source))
                actual_counts[idx] += 1

                print("AGN: {}".format(len(parameters)))

        else:

            parameters.append(np.array(new_source))
            actual_counts[idx] += 1

            print("AGN: {}".format(len(parameters)))

    return np.array(parameters)

# REFERENCES

# Digitise Error - https://stackoverflow.com/questions/4355132/numpy-digitize-returns-values-out-of-range
# Exclude NaNs - https://stackoverflow.com/questions/17126543/numpy-array-get-the-subset-slice-of-an-array-which-is-not
# -nan
# Exclude NaNs - https://stackoverflow.com/questions/11620914/how-do-i-remove-nan-values-from-a-numpy-array
# Numpy Float Handling - https://stackoverflow.com/questions/58083198/how-to-handle-both-float-and-array-input-in-python
# Numpy Function Mapping - https://stackoverflow.com/questions/35215161/most-efficient-way-to-map-function-over-numpy-
# array
