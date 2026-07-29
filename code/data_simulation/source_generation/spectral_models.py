"""
Methods for calculating the various fluxes of gamma-ray sources.
"""

import numpy as np
import warnings


def agn_photon_flux(E, E_0, F_0, alpha, beta):
    """Returns the photon flux of an AGN for a given energy E.

    Parameters
    ----------
    E
        Energy at which to calculate the photon flux.
    E_0
        Pivot energy [MeV] of AGN.
    F_0
        Flux density [photons/cm2/MeV/s] of AGN.
    alpha
        Spectral slope of AGN spectrum.
    beta
        Curvature of AGN spectrum.

    """
    division = E / E_0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exponent = - alpha - (beta * np.log(division))

    dF_dE = F_0 * np.power(division, exponent)

    return dF_dE


def agn_spectral_model(E, E_0, F_0, alpha, beta):
    """Returns the energy flux of an AGN for a given energy E.

    Parameters
    ----------
    E
        Energy at which to calculate energy flux.
    E_0
        Pivot energy [MeV] of AGN.
    F_0
        Flux density [photons/cm2/MeV/s] of AGN.
    alpha
        Spectral slope of AGN spectrum.
    beta
        Curvature of spectral slope of AGN

    """
    division = E / E_0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exponent = - alpha - (beta * np.log(division))

    dF_dE = F_0 * np.power(division, exponent)

    return E * dF_dE


def pulsar_photon_flux(E, F_0, E_0, Gamma, a, b):
    """ Returns the photon flux of pulsar at given energy.

    E
        Energy at which to calculate photon flux
    F_0
        Flux density [photons/cm2/MeV/s] of pulsar.
    E_0
        Pivot energy [MeV] of pulsar.
    Gamma
        Spectral slope of pulsar spectrum
    a
        Pulsar spectrum's exponential factor [Mev^-b]
    b
        Exponential index of pulsar spectrum

    """
    division = E / E_0

    exponent = a * (np.power(E_0, b) - np.power(E, b))

    dF_dE = F_0 * np.power(division, -Gamma) * np.exp(exponent)

    return dF_dE


def pulsar_spectral_model(E, F_0, E_0, Gamma, a, b):
    """Returns the energy flux of a pulsar at a given energy

    Parameters
    ----------
    E
        Energy at which to calculate energy flux
    F_0
        Flux density [photons/cm2/MeV/s] of pulsar
    E_0
        Pivot energy [MeV] of pulsar
    Gamma
        Spectral slope of pulsar spectrum
    a
        Pulsar spectrum's Exponential factor [Mev^-b]
    b
        Exponential index of pulsar spectrum

    """
    division = E / E_0

    exponent = a * (np.power(E_0, b) - np.power(E, b))

    dF_dE = F_0 * np.power(division, -Gamma) * np.exp(exponent)

    return E * dF_dE
