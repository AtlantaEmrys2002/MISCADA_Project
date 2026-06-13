import numpy as np
from . spectral_models import pulsar_spectral_model
from scipy.integrate import quad


def energy_flux_pulsar(pivot_energy, flux_density, spectral_slope, exponential_index, exponential_factor):

    # Integrate over 0.1 - 100 GeV (100 - 100000 MeV)
    energy = quad(pulsar_spectral_model, 100, 100000, args=(pivot_energy, flux_density, spectral_slope,
                                                            exponential_index, exponential_factor))[0]

    return energy


def s1_agn(pivot_energy, flux_density, spectral_slope, curvature):

    # Integrate above 1 GeV
    s1 = quad(pulsar_spectral_model, 1000, np.inf, args=(pivot_energy, flux_density, spectral_slope, curvature))[0]

    return s1


def s10_agn(pivot_energy, flux_density, spectral_slope, curvature):

    # Integrate above 10 GeV
    s1 = quad(pulsar_spectral_model, 10000, np.inf, args=(pivot_energy, flux_density, spectral_slope, curvature))[0]

    return s1


# REFERENCES

# ID8 Paper - Identification of point sources in gamma rays using U-shaped convolutional neural networks and a data
# challenge
# Numpy Documentation - https://numpy.org/doc/stable/user/index.html
# Scipy Documentation - https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html#scipy.integrate.
# quad
