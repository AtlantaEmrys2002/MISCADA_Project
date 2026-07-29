"""
Methods for generating AGN spectral parameters based either on fitted distributions or correlations with other spectral
parameters.
"""

import numpy as np
from scipy.integrate import quad
from .spectral_models import agn_spectral_model, agn_photon_flux
import warnings

# polynomial with degree 2 relating pivot energies and flux densities (coefficients below are c, b, a) such that
# ax^2 + bx + c = 0
poly = np.polynomial.Polynomial([-3.106131156761208, -4.371650516868555, 0.1097545334282421])


def energy_flux_agn(pivot_energy, flux_density, spectral_slope, curvature, min_energy=100.0, max_energy=100000.0):
    """Calculates the integral energy flux of a source, given the spectral parameters, between 100 and 100000 MeV - this
    is just to guide the luminosity function calculation.

    Parameters
    ----------
    pivot_energy
        Pivot energy [MeV] of AGN.
    flux_density
        Flux density [photons/cm2/MeV/s] of AGN
    spectral_slope
        Spectral slope of AGN spectrum
    curvature
        AGN spectrum curvature
    min_energy : np.float64
        Minimum photon energy from which to integrate from
    max_energy:
        Maximum photon energy to integrate up to

    """
    # Integrate over 0.1 - 100 GeV
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        energy = quad(agn_spectral_model, min_energy, max_energy, args=(pivot_energy, flux_density, spectral_slope,
                                                                        curvature))[0]

    # Convert energy fluxes so ergs included in units instead of photons - see ID43
    return energy * 1.602 * 10 ** (-6)


def integral_photon_flux_agn(pivot_energy, flux_density, spectral_slope, curvature, min_energy=100.0,
                             max_energy=100000.0):
    """Calculates the integral photon flux of a source, given the spectral parameters, between 100 and 100000 MeV - this
    is just to guide the luminosity function calculation.

    Parameters
    ----------
    pivot_energy
        Pivot energy [MeV] of AGN.
    flux_density
        Flux density [photons/cm2/MeV/s] of AGN
    spectral_slope
        Spectral slope of AGN spectrum
    curvature
        AGN spectrum curvature
    min_energy : np.float64
        Minimum photon energy from which to integrate from
    max_energy:
        Maximum photon energy to integrate up to

    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        energy = quad(agn_photon_flux, min_energy, max_energy, args=(pivot_energy, flux_density, spectral_slope,
                                                                     curvature))[0]

    return energy


def agn_flux_density(pivot_energies, noise_std=0.9181233203181715):
    """Calculate the pivot energy of a source given its flux density - these two spectral parameters are correlated.

    Parameters
    ----------
    pivot_energies
        Pivot energy [MeV] of source(s)
    noise_std
        Standard deviation of Gaussian noise present in observed flux densities.

    """
    # Default values for a, b, and c based on correlation analysis between pivot energies and flux densities. Found that
    # relation between E_0 and F_0 could be simulated as logF = a * logE^2 + b * logE + c

    log_pivot_energies = np.log(pivot_energies)

    log_flux_densities = poly(log_pivot_energies)

    # Found standard deviation of the residuals of fit of 4FGL data and quadratic fitted to logE_0 and logF_0
    # (noise_std), assuming mean = 0. Add this simulated noise to log of flux densities to increase realism.

    # Calculate number of noise elements to generate and create noise
    size = np.atleast_1d(log_flux_densities).shape
    noise = np.random.normal(loc=0, scale=noise_std, size=size)

    # Add noise
    log_flux_densities += noise

    flux_densities = np.exp(log_flux_densities)

    return flux_densities


def agn_spectral_slope(pivot_energies, m=-0.3454412867556543, c=4.755250286569133, noise_std=0.13347913702284359):
    """Calculate the spectral slope of a source given its pivot energy - these two spectral parameters are correlated.

    Parameters
    ----------
    pivot_energies
        Pivot energy [Mev] of source(s)
    m
        Gradient of log-log plot relating pivot energies and spectral slopes
    c
        y-intercept of log-log plot relating pivot energies and spectral slopes
    noise_std
        Standard deviation of Gaussian noise present in observed spectral slopes
    """
    # Based on correlation analysis of pivot energies and flux densities and spectral slopes,
    # created this method for generating spectral slopes based on pivot energies after fitting relation
    # found in cited paper (see fitting_agn_pivot_energy_spectral_slope_relation() for more info)

    log_pivot_energies = np.log(pivot_energies)

    alphas = (log_pivot_energies * m) + c

    # Found standard deviation of the residuals of fit of 4FGL data (noise_std), assuming mean = 0. Add this simulated
    # noise to log of alphas to increase realism.

    # Calculate number of noise elements to generate and create noise
    size = np.atleast_1d(log_pivot_energies).shape

    noise = np.random.normal(loc=0, scale=noise_std, size=size)

    # Add noise

    alphas += noise

    return alphas

# REFERENCES

# ID8 Paper - Identification of point sources in gamma rays using U-shaped convolutional neural networks and a data
# challenge
# Numpy Documentation - https://numpy.org/doc/stable/user/index.html
# Numpy Polynomials - https://stackoverflow.com/questions/76603915/get-polynomial-x-at-y-python-3-10-numpy
# Scipy Documentation - https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html#scipy.integrate.
# quad
