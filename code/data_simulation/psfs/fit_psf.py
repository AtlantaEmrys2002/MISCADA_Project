from astropy.io import fits
import numpy as np
from scipy.optimize import curve_fit
from . utils import dual_function


def scale_psf(psf_values, energy_bin, c_0=3.5, c_1=0.15, beta=0.8):

    # The constants included as arguments above were derived in
    # https://iopscience.iop.org/article/10.1088/0004-637X/765/1/54/pdf

    # Calculate energy scale factor
    scale_factor = np.sqrt(((c_0 * (energy_bin / 100) ** (-beta)) ** 2) + c_1)

    # Scale PSF values
    psf_values /= scale_factor

    return psf_values


def normalise_psf(thetas, psf_values):

    # How to normalise a function -
    # https://math.stackexchange.com/questions/4806473/forcing-a-function-to-integrate-to-1
    # How to integrate over solid angle for this specfic PSF -
    # https://gamma-astro-data-formats.readthedocs.io/en/v0.1/irfs/psf/index.html#psf-pdf

    # Apply this function AFTER energy scaling

    probs = ((2 * np.pi * thetas) ** 2) * psf_values

    # Integrate over probs
    approx_integral = np.sum(np.array(
        [((probs[k + 1] + probs[k]) / 2) * (thetas[k + 1] - thetas[k]) for k in range(len(psf_values) - 1)]))

    # Normalise such that the integral is 1
    probs /= approx_integral

    return probs


def fit_point_source_psf(file_name):

    function_params = []

    # The constants included as arguments above were derived in
    # https://iopscience.iop.org/article/10.1088/0004-637X/765/1/54/pdf

    with fits.open(file_name) as hdul:

        thetas = np.array([k[0] for k in hdul["THETA"].data])

        psf_data = hdul["PSF"].data

        num_bins = len(psf_data)

        for b in range(num_bins):

            # Lowest energy (MeV) of this bin
            energy_value = psf_data[b][0]

            # PSF values dP/dOmega - probability to find event in solid angle dOmega at offset r from point source
            psf_values = np.array(psf_data[b][2])

            psf_values = scale_psf(psf_values, energy_value)

            probs = normalise_psf(thetas, psf_values)

            # Fit King function (Moffat distribution to values to create a probability density function)
            popt, _ = curve_fit(dual_function, thetas, probs, maxfev=10000)

            function_params.append(popt)

    return np.array(function_params)
