from . agn_spectral_parameters import agn_flux_density, agn_spectral_slope, energy_flux_agn
from astropy.table import QTable
from math import floor
import numpy as np


# GENERATE FIXED NUMBER OF AGNS WITHIN GIVEN ENERGY FLUX RANGE FOR FLAT EXTRAPOLATION AT LOWER ENERGY FLUXES
def agn_generator(agn_stats, energy_flux_low=0., energy_flux_high=1000.):

    # Default is effectively source with any energy flux

    (mean_log_pivot_energy_agn, std_log_pivot_energy_agn, betas_agn) = agn_stats

    while True:

        # SPECTRAL PARAMETERS

        # Generate new pivot energy

        # Although ID8 stated that they only randomly sampled flux densities according to a log-normal distribution, a
        # log-normal distribution fits pivot energies much better than their recommended Gaussian (and this makes sense
        # as differential flux density and pivot energy are correlated). Therefore, I have changed the way pivot
        # energies are randomly generated (and now use a log-normal distribution)
        pivot_energy = np.random.lognormal(mean=mean_log_pivot_energy_agn, sigma=std_log_pivot_energy_agn, size=1)[0]

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
        beta = np.random.choice(betas_agn)

        # Calculate energy flux of source
        energy_flux = energy_flux_agn(pivot_energy, flux_density, spectral_slope, beta)

        if (energy_flux >= energy_flux_low) and (energy_flux < energy_flux_high) and (~np.isnan(energy_flux)):

            # SPATIAL PARAMETERS

            # longitude
            longitude = np.random.uniform(low=0, high=2 * np.pi)

            # latitude
            sin_galactic_latitudes = np.random.uniform(low=-1, high=1)
            latitude = np.arcsin(sin_galactic_latitudes)

            return np.array([pivot_energy, flux_density, spectral_slope, beta, energy_flux, longitude, latitude])


def luminosity_function_agn(catalog: str, detection_threshold):

    # Used to build luminosity function of the simulated AGNs

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

    # The minimum energy flux of our generated sources is an order of magnitude less than the 4FGL
    our_threshold = detection_threshold / 10

    # Following method detailed in ID8

    # Bin 4FGL data
    min_bin_val = np.log10(np.min(energy_fluxes_4fgl))
    max_bin_val = np.log10(np.max(energy_fluxes_4fgl))

    log_linspace = np.linspace(min_bin_val, max_bin_val)

    bin_edges = 10 ** log_linspace
    counts, bin_intervals = np.histogram(energy_fluxes_4fgl, bins=bin_edges)

    # Calculate width of bins
    bin_width = log_linspace[1] - log_linspace[0]

    # FLAT EXTRAPOLATION TO FAINTER DETECTION THRESHOLD

    # Extend to one order of magnitude less than the detection threshold of the 4FGL (similar premise to ID8) -
    # assume constant below given threshold (not Gaussian)

    # Number of bins between current lowest energy bin and our faint source threshold
    num_extra_bins = floor((np.log10(bin_intervals[0]) - np.log10(our_threshold)) / bin_width)

    extra_intervals = [10 ** (np.log10(bin_intervals[0]) - (bin_width * x)) for x in range(num_extra_bins, 0, -1)]

    # Create extra bin intervals

    # Calculate number of random

    # Find bin with the most AGNs
    peak = np.argmax(counts)

    # For any bin to the right of the peak that has 0 or 1 expected counts, set to 2
    for k in range(peak, len(counts)):
        if counts[k] == 1 or counts[k] == 0:
            counts[k] = 2

    # Generate random numbers for number of energy flux bins to the right of the peak
    n_noise = list(np.random.uniform(low=0.8, high=1.3, size=len(bin_intervals) - 1 - peak))

    # Create some noise in energy bins greater than peak
    for k in range(peak, len(n_noise)):
        counts[k + peak] = counts[k + peak] * n_noise[k]

    bin_intervals = np.array(extra_intervals + list(bin_intervals))

    # Set number of counts equal to peak for original 4FGL bins to the left of the peak
    for k in range(0, peak):
        counts[k] = n_min

    counts = [n_min for _ in range(num_extra_bins)] + list(counts)

    # Convert counts to int
    counts = [int(k) for k in counts]

    peak = peak + num_extra_bins

    return np.array([counts])[0], np.array(bin_intervals), peak


def generate_mock_agn_catalog(catalog, agn_data, detection_threshold=np.float64(1.0 * 10 ** (-12))):

    # Changed threshold from 2.0 * 10 ** -12 TO 1.0 * 10 ** -12 as threshold recommended by 4FGL DR4 (ID22) paper for
    # outside galactic plane and detection threshold has decreased since ID8 was published

    # SPECTRAL PARAMETERS

    # Select beta values and convert from masked to ordinary numpy array
    betas_agn = agn_data['LP_beta'].data.filled(np.nan)

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

    actual_counts = np.zeros_like(target_counts)

    parameters = []

    while (actual_counts[0] < target_counts[0]) and np.any(np.less(actual_counts[target_peak:],
                                                                   target_counts[target_peak:])):

        # Due to uncomplimentary functionality - need max of intervals[:-1] - see reference to Digitize Error
        new_source = agn_generator((mean_log_pivot_energy_agn, std_log_pivot_energy_agn, betas_agn),
                                   energy_flux_low=np.min(target_bin_intervals),
                                   energy_flux_high=np.max(target_bin_intervals[:-1]))

        idx = np.digitize(new_source[4], target_bin_intervals)

        if 0 < idx < len(target_bin_intervals):

            if actual_counts[idx] < target_counts[idx]:

                parameters.append(np.array(new_source))
                actual_counts[idx] += 1

        else:

            parameters.append(np.array(new_source))
            actual_counts[idx] += 1

    # CHECK SIMULATED FLUX DENSITIES AND PIVOT ENERGIES HAVE SAME CORRELATION AS IN 4FGL
    # plt.title('Simulated $F_0$ against $E_0$')
    # plt.xlabel('log $E_0$')
    # plt.ylabel('log $F_{0, AGN}$')
    # plt.scatter(np.log(parameters[:, 0]), np.log(parameters[:, 1]), s=2)
    # plt.show()
    #
    # # CHECK SIMULATED PIVOT ENERGIES AND SPECTRAL INDICES (ALPHAS) HAVE SAME CORRELATION AS IN 4FGL
    # plt.title('Simulated $\\alpha$ against $E_0$')
    # plt.xlabel('$E_0$')
    # plt.ylabel('$\\alpha$')
    # plt.scatter(parameters[:, 0], parameters[:, 2], s=2)
    # plt.show()

    return np.array(parameters)


# REFERENCES

# Digitise Error - https://stackoverflow.com/questions/4355132/numpy-digitize-returns-values-out-of-range
# Numpy Float Handling - https://stackoverflow.com/questions/58083198/how-to-handle-both-float-and-array-input-in-python
