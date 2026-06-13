import numpy as np
import warnings


def agn_spectral_model(E, E_0, F_0, alpha, beta):

    division = E/E_0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exponent = - alpha - (beta * np.log(division))

    dF_dE = F_0 * np.power(division, exponent)

    return E * dF_dE


def pulsar_spectral_model(E, F_0, E_0, Gamma, a, b):

    exponent = a * (np.power(E_0, b) - np.power(E, b))

    dF_dE = F_0 * np.power((E/E_0), -Gamma) * np.exp(exponent)

    return E * dF_dE
