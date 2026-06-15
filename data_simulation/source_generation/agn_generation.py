from . agn_spectral_parameters import agn_flux_density, agn_spectral_slope, energy_flux_agn
from astropy import units as u
import numpy as np

import matplotlib.pyplot as plt


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


def generate_mock_agn_catalog(agn_data, num_agns=200, detection_threshold=np.float64(1.0 * 10 ** (-12))):

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

    parameters = []

    for x in range(num_agns):
        new_source = agn_generator((mean_log_pivot_energy_agn, std_log_pivot_energy_agn, betas_agn),
                                    energy_flux_low=detection_threshold, energy_flux_high=1000)
        parameters.append(np.array(new_source))

    parameters = np.array(parameters)

    # FAINT SOURCE FLAT EXTRAPOLATION

    # Flat extrapolation of AGN - assume constant below given threshold (not Gaussian)

    # Bin data and take average of first three
    bin_edges = 10 ** np.linspace(-14, -9)
    counts, _ = np.histogram(parameters[:, 4], bins=bin_edges)

    # Take average number of sources of first five bins that contain some sources for flat extrapolation
    first_non_empty_bin = np.nonzero(counts)[0][0]
    mean_counts_per_bin, std_counts_per_bin = (np.mean(counts[first_non_empty_bin: first_non_empty_bin + 10]),
                                               np.std(counts[first_non_empty_bin: first_non_empty_bin + 10], ddof=1))

    faint_sources = []

    # Extend to one order of magnitude less than the detection threshold of the 4FGL (similar premise to ID8)
    our_threshold = np.argwhere(bin_edges >= detection_threshold / 10)[0][0]

    for x in range(our_threshold, first_non_empty_bin):

        # Generate number of sources in bin - approximately flat/same as bins at peak
        counts_per_bin_flat_extrapolation = round(np.random.normal(loc=mean_counts_per_bin, scale=std_counts_per_bin))

        for k in range(counts_per_bin_flat_extrapolation):
            new_source = agn_generator((mean_log_pivot_energy_agn, std_log_pivot_energy_agn, betas_agn),
                                        bin_edges[x], bin_edges[x + 1])

            faint_sources.append(new_source)

    faint_sources = np.asarray(faint_sources)

    parameters = np.vstack((parameters, faint_sources))

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

    return parameters


# REFERENCES

# Numpy Float Handling - https://stackoverflow.com/questions/58083198/how-to-handle-both-float-and-array-input-in-python
