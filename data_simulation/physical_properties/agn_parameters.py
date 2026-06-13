import numpy as np
from scipy.integrate import quad
from . spectral_models import agn_spectral_model
import warnings

# polynomial with degree 2 relating pivot energies and flux densities (coefficients below are c, b, a) such that
# ax^2 + bx + c = 0
poly = np.polynomial.Polynomial([-21.159501476671274, -2.8553356136745753, 0.10975452653160912])


def energy_flux_agn(pivot_energy, flux_density, spectral_slope, curvature):

    # Integrate over 0.1 - 100 GeV
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        energy = quad(agn_spectral_model, 100, 100000, args=(pivot_energy, flux_density, spectral_slope, curvature))[0]

    return energy


def agn_flux_density(pivot_energies, noise_std=0.9181233644485474):

    # Default values for a, b, and c based on correlation analysis between pivot energies and flux densities. Found that
    # relation between E_0 and F_0 could be simulated as logF = a * logE^2 + b * logE + c

    # a: 0.10975452653160912
    # b: -2.8553356136745753
    # c: -21.159501476671274

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


def agn_spectral_slope(pivot_energies, m=-0.3454412867553224, c=2.369026429991104, noise_std=0.1005791425704956):

    # Based on correlation analysis of pivot energies and flux densities and spectral slopes,
    # created this method for generating spectral slopes based on pivot energies after fitting relation
    # found in cited paper (see fitting_agn_pivot_energy_spectral_slope_relation() for more info)

    log_pivot_energies = np.log(pivot_energies)

    log_alphas = np.log((log_pivot_energies * m) + c)

    # Found standard deviation of the residuals of fit of 4FGL data (noise_std), assuming mean = 0. Add this simulated
    # noise to log of alphas to increase realism.

    # Calculate number of noise elements to generate and create noise
    size = np.atleast_1d(log_alphas).shape
    noise = np.random.normal(loc=0, scale=noise_std, size=size)

    # Add noise
    log_alphas += noise

    alphas = np.exp(log_alphas)

    return alphas


def s1_agn(pivot_energy, flux_density, spectral_slope, curvature):

    # Integrate above 1 GeV
    s1 = quad(agn_spectral_model, 1000, np.inf, args=(pivot_energy, flux_density, spectral_slope, curvature))[0]

    return s1


def s10_agn(pivot_energy, flux_density, spectral_slope, curvature):

    # Integrate above 10 GeV
    s1 = quad(agn_spectral_model, 10000, np.inf, args=(pivot_energy, flux_density, spectral_slope, curvature))[0]

    return s1

# REFERENCES

# ID8 Paper - Identification of point sources in gamma rays using U-shaped convolutional neural networks and a data
# challenge
# Numpy Documentation - https://numpy.org/doc/stable/user/index.html
# Scipy Documentation - https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html#scipy.integrate.
# quad
