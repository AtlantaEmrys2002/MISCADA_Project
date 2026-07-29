"""
Methods for generating pulsar spectral parameters based either on fitted distributions or correlations with other
spectral parameters.
"""

import numpy as np
from .spectral_models import pulsar_spectral_model, pulsar_photon_flux
from scipy.integrate import quad
import warnings

poly = np.polynomial.Polynomial([179.66839590413852, -89.19289721267037, 12.958983638603955, -0.6310345163342851])


def energy_flux_pulsar(pivot_energy, flux_density, spectral_slope, exponential_index, exponential_factor,
                       min_energy=100.0, max_energy=100000.0):
    """Calculates the integral energy flux of a source, given the spectral parameters, between 100 and 100000 MeV - this
    is just to guide the luminosity function calculation.

    Parameters
    ----------
    pivot_energy
        Pivot energy [MeV] of pulsar
    flux_density
        Flux density [photons/cm2/MeV/s] of pulsar
    spectral_slope
        Spectral slope of pulsar spectrum (Gamma)
    exponential_index
        Exponential index b of pulsar spectrum
    exponential_factor
        Exponential factor a of pulsar spectrum [MeV^-b]
    min_energy



    """
    # Integrate over 0.1 - 100 GeV
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Integrate over 0.1 - 100 GeV (100 - 100000 MeV)
        energy = quad(pulsar_spectral_model, min_energy, max_energy, args=(flux_density, pivot_energy, spectral_slope,
                                                                           exponential_factor, exponential_index))[0]

    # Convert energy fluxes so ergs included in units instead of photons - see ID43
    return energy * 1.602 * 10 ** (-6)


def integral_photon_flux_pulsar(pivot_energy, flux_density, spectral_slope, exponential_index, exponential_factor,
                                min_energy=100.0, max_energy=100000.0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Integrate over 0.1 - 100 GeV (100 - 100000 MeV)
        energy = quad(pulsar_photon_flux, min_energy, max_energy, args=(flux_density, pivot_energy, spectral_slope,
                                                                        exponential_factor, exponential_index))[0]

    return energy


def pulsar_flux_density(pivot_energies, noise_std=0.8723404255319149):
    log_pivot_energies = np.log(pivot_energies)

    log_flux_densities = poly(log_pivot_energies)

    # Calculate number of noise elements to generate and create noise
    size = np.atleast_1d(log_flux_densities).shape
    noise = np.random.normal(loc=0, scale=noise_std, size=size)

    # Add noise
    log_flux_densities += noise

    flux_densities = np.exp(log_flux_densities)

    return flux_densities

# REFERENCES

# ID8 Paper - Identification of point sources in gamma rays using U-shaped convolutional neural networks and a data
# challenge
# Numpy Documentation - https://numpy.org/doc/stable/user/index.html
# Scipy Documentation - https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html#scipy.integrate.
# quad
