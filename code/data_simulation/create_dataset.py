from astropy.io import fits
# import healpy as hp
import numpy as np

from map_generation.visualisation import plot_all_sky_map
from map_generation.healpix_maps import create_expected_counts_map, create_exposure_map, create_infinite_statistics_map
from map_generation.utils import angle_to_healpix_pixels, get_nside, dual_function
from read_write_functions import xml_parser

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def fit_point_source_psf(file_name):
    function_params = []

    # These values were derived in https://iopscience.iop.org/article/10.1088/0004-637X/765/1/54/pdf
    c_0 = 3.5
    c_1 = 0.15
    beta = 0.8

    with fits.open(file_name) as hdul:

        thetas = np.array([k[0] for k in hdul["THETA"].data])

        num_bins = len(hdul["PSF"].data)

        for b in range(num_bins):
            # Lowest energy (MeV) of this bin
            energy_value = hdul["PSF"].data[b][0]

            # PSF values dP/dOmega - probability to find event in solid angle dOmega at offset r from point source
            psf_values = np.array(hdul["PSF"].data[b][2])

            # REFERENCES FOR THIS NORMALISATION AND WHY WE ARE MULTIPLYING PSF VALUES
            # BY 2, pi, thetas:
            # https://gamma-astro-data-formats.readthedocs.io/en/v0.1/irfs/psf/index.html#psf-pdf
            # https://gamma-astro-data-formats.readthedocs.io/en/v0.1/irfs/psf/psf_gtpsf/
            # Robust Neural paper
            # How to normalise - https://math.stackexchange.com/questions/4806473/forcing-a-function-to-integrate-to-1

            # Energy scale factor
            scale_factor = np.sqrt(((c_0 * (energy_value / 100) ** (-beta)) ** 2) + c_1)

            psf_values /= scale_factor

            probs = ((2 * np.pi * thetas) ** 2) * psf_values

            # Integrate over probs
            approx_integral = np.sum(np.array(
                [((probs[k + 1] + probs[k]) / 2) * (thetas[k + 1] - thetas[k]) for k in range(len(psf_values) - 1)]))

            # Normalise such that the integral is 1
            probs /= approx_integral

            plt.plot(thetas, probs, label="{}".format(energy_value))

            # Fit King function (Moffat distribution to values to create a probability density function)
            popt, _ = curve_fit(dual_function, thetas, probs, maxfev=10000)

            function_params.append(popt)

    return np.array(function_params)

            # plt.plot(thetas, dual_function(thetas, sigma_core=popt[0], gamma_core=popt[1], sigma_tail=popt[2],
            #                                gamma_tail=popt[3], f_core=popt[4]))
            #
            # plt.xlabel("$\\theta\ [\degree]$")
            # plt.ylabel("$\\frac{dP}{d\Omega(r)}$")
            #
            # plt.legend()
            #
            # plt.xlim(0, 4)
            #
            # plt.show()
            #
            # plt.close()


# MAIN PROGRAM

# # Prepare exposure maps
# exposure_maps, energy_bins = create_exposure_map(
#     exposure_file="/Volumes/T7/project_data/real_data/fermi_filtered_gti_exposure_map.fits")

# # Get NSIDE parameter from exposure map
# nside = get_nside(exposure_maps[0])
#
# # Plot exposure maps to verify correctness
# plot_all_sky_map(healpix_maps=exposure_maps, energy_bins=energy_bins, title="Exposure",
#                  directory="./plots/all_sky_maps/")
#
# coordinates, binned_fluxes = xml_parser(energy_bins=energy_bins, xml_file="./simulated_data/sources.xml")
#
# pixels = angle_to_healpix_pixels(coordinates, nside=nside)
#
# # Calculated source locations in lon-lat, the pixels in which they are situated in the healpix map, the binned exposure
# # maps of the sky, and their fluxes
# infinite_statistics_maps = create_infinite_statistics_map(exposure_maps, binned_fluxes, pixels)
#
# # Plot infinite counts maps
# plot_all_sky_map(healpix_maps=infinite_statistics_maps, energy_bins=energy_bins, title="Infinite Counts",
#                  directory="./plots/all_sky_maps/", logarithmic=True)
#
# # Sample infinite statistics maps to create expected counts maps
# count_maps = create_expected_counts_map(infinite_counts_map=infinite_statistics_maps)
#
# plot_all_sky_map(healpix_maps=count_maps, energy_bins=energy_bins, title="Expected Counts",
#                  directory="./plots/all_sky_maps/", logarithmic=True)

# Create and fit point spread function
binned_function_parameters = fit_point_source_psf(file_name="/Volumes/T7/project_data/real_data/pointsource_psf.fits")

print(binned_function_parameters)


# N.B. Do celestial (RA/Dec coords for PSF) - https://iopscience.iop.org/article/10.1088/0067-0049/203/1/4/pdf
