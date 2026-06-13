from . agn_spectral_parameters import agn_flux_density, agn_spectral_slope, energy_flux_agn
from astropy import units as u
import numpy as np


# GENERATE FIXED NUMBER OF AGNS WITHIN GIVEN ENERGY FLUX RANGE FOR FLAT EXTRAPOLATION AT LOWER ENERGY FLUXES
def agn_generation(agn_stats, energy_flux_low=0, energy_flux_high=1000):

    # Default is effectively source with any energy flux

    (mean_log_pivot_energy_agn, std_log_pivot_energy_agn, betas_agn) = agn_stats

    while True:

        # Generate new pivot energy

        # Although ID8 stated that they only randomly sampled flux densities according to a log-normal distribution, a
        # log-normal distribution fits pivot energies much better (and this makes sense as differential flux density and
        # pivot energy are correlated).
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

        # longitude
        longitude = np.random.uniform(low=0, high=2 * np.pi)

        # latitude
        sin_galactic_latitudes = np.random.uniform(low=-1, high=1)
        latitude = np.arcsin(sin_galactic_latitudes)

        if (energy_flux >= energy_flux_low) and (energy_flux < energy_flux_high):

            return np.array([pivot_energy, flux_density, spectral_slope, beta, energy_flux, longitude, latitude])


# def agn_statistics(agns):
#
#     # Select beta values and convert from masked to ordinary numpy array
#     betas = agns['LP_beta'].data.filled(np.nan)
#
#     # Select pivot energy values and convert from MeV to GeV
#     pivot_energies = agns['Pivot_Energy'].to(u.GeV).value
#
#     log_pivot_energies = np.log(pivot_energies)
#
#     mean_log_pivot_energy, std_log_pivot_energy = np.nanmean(log_pivot_energies), np.nanstd(log_pivot_energies, ddof=1)
#
#     return mean_log_pivot_energy, std_log_pivot_energy, betas
