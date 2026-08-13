"""
Functions for calculating the point spread function (PSF) for application to both diffuse and point sources.
"""

from astropy.io import fits
import healpy as hp
import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit
from .utils import dual_function, normalise_psf, scale_psf


def fit_diffuse_source_psf(roi_count_map: str, nside: int = 512) -> list[npt.NDArray[np.float64]]:
    """Derives point spread function from gtmodel output that can be applied to diffuse background expected counts map.

    Parameters
    ----------
    roi_count_map : str
        Location of FITS-formatted infinite statistics expected count map of ROI in real data.
    nside : int
        Order of HEALPix maps to create beam function (used as PSF) for.

    Returns
    -------
    list
        Energy-binned PSFs for diffuse sources.

    """
    # N.B. do not energy integrate this function, as we are looking at energy-binned gtmodel output - already covers
    # the entire energy bin

    with fits.open(roi_count_map) as hdul:
        # Centre of data
        midpoint = hdul[0].data[0].shape[0] // 2

        num_bins = hdul[0].data.shape[0]

        psfs = []

        for b in range(num_bins):
            counts = hdul[0].data[b]

            # Average x and y-axis through point
            x_axis = counts[midpoint]
            y_axis = counts[:, midpoint]

            minus_included = np.arange(-x_axis.shape[0], x_axis.shape[0])

            # Arithmetic mean along the x and y-axis (remember, this is radial)
            values = (x_axis + y_axis) / 2

            # Convert from pixels to radians - approximately
            side_length_pixel = np.sqrt((4 * np.pi) / (12 * nside ** 2))

            # The minus included is in degrees, therefore multiply by pi/180 to convert to radians
            radius = side_length_pixel * minus_included * np.pi / 180

            beam = np.array(hp.sphtfunc.bl2beam(bl=values, theta=radius))

            # Normalise function (y-axis)
            beam /= np.max(beam)

            # Clip < 0 values to 0
            beam = np.clip(beam, a_min=0, a_max=np.max(beam))

            psfs.append(beam)

    return psfs


def fit_point_source_psf(file_name: str) -> npt.NDArray[np.float64]:
    """Derives point spread function from gtmodel output that can be applied to point source expected counts map.

    Parameters
    ----------
    file_name : str
        Location of gtpsf output when applied to ROI in real Fermi LAT data.

    Returns
    -------
    ndarray
        Energy-binned PSF for application to point source infinite statistics maps.

    """
    function_params = []

    with fits.open(file_name) as hdul:

        # print(hdul.info())

        thetas = np.array([k[0] for k in hdul["THETA"].data])

        psf_data = hdul["PSF"].data

        num_bins = len(psf_data)

        for b in range(num_bins):
            # Lowest energy (MeV) of this bin
            energy_value = psf_data[b][0]

            # PSF values dP/dOmega - probability to find event in solid angle dOmega at offset r from point source
            psf_values = np.array(psf_data[b][2])

            # Scales out energy dependence
            # psf_values = scale_psf(psf_values, energy_value)

            probs = normalise_psf(thetas, psf_values)

            # # DIVIDE BY BIN WIDTHS
            # for k in range(len(probs) - 1):
            #     if energy_value != 0:
            #         probs[k] /= ((thetas[k] - thetas[k + 1]) / np.sqrt(((3.5 * ((energy_value / 100) ** (-0.8))) ** 2) + (0.15 ** 2)))
            #     else:
            #
            #         probs[k] /= ((thetas[k] - thetas[k + 1]) / np.sqrt((3.5 ** 2) + (0.15 ** 2)))

            # Fit King function (Moffat distribution to values to create a probability density function)
            popt, _ = curve_fit(dual_function, xdata=thetas, ydata=probs, maxfev=10000)

            function_params.append(popt)

    return np.array(function_params)

# REFERENCES

# Adjusting Curve Fit Iterations - https://stackoverflow.com/questions/15831763/scipy-curvefit-runtimeerroroptimal-
# parameters-not-found-number-of-calls-to-fun
# Approximation of Side Length -https://arxiv.org/html/2410.12951v1
# Custom PDFs - https://math.stackexchange.com/questions/3614107/how-do-you-create-a-custom-probability-density-function
# -from-a-discrete-distribu
# Function Fits - https://stackoverflow.com/questions/68523795/fit-a-custom-function-in-python
# FWHM - https://stackoverflow.com/questions/8914491/finding-the-nearest-value-and-return-the-index-of-array-in-python
# FWHM 2 - https://en.wikipedia.org/wiki/Full_width_at_half_maximum
# Normalising Functions - https://math.stackexchange.com/questions/4806473/forcing-a-function-to-integrate-to-1
# Normalise PDF - https://stackoverflow.com/questions/52223236/how-to-adjust-a-data-set-so-that-the-total-sum-is-equal-
# to-1-i-thought-i-kne
# PDF > 1 - https://math.stackexchange.com/questions/1720053/how-can-a-probability-density-function-pdf-be-greater-
# than-1
# PDF from Data - https://math.stackexchange.com/questions/2325565/is-it-possible-to-calculate-probability-density-
# function-from-a-data-set
# PSF Format - https://escholarship.org/content/qt723151vx/qt723151vx.pdf
# PSF Information - https://gamma-astro-data-formats.readthedocs.io/en/v0.1/irfs/psf/index.html#psf-pdf
# Radial Profiles - https://cxc.cfa.harvard.edu/ciao/why/radial_profile_correction.html
# Radial Profiles - https://stackoverflow.com/questions/34965275/radial-profile-from-a-fits-image
# Window Function - https://en.wikipedia.org/wiki/Window_function
