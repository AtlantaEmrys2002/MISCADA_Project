from astropy.io import fits
# import healpy as hp
import numpy as np

from map_generation.visualisation import plot_all_sky_map
from map_generation.healpix_maps import create_expected_counts_map, create_exposure_map, create_infinite_statistics_map
from map_generation.utils import angle_to_healpix_pixels, get_nside
from read_write_functions import xml_parser
from psfs.fit_psf import fit_point_source_psf
from psfs.visualisation import plot_fitted_point_source_psf


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

plot_fitted_point_source_psf(psf_file="/Volumes/T7/project_data/real_data/pointsource_psf.fits",
                             function_parameters=binned_function_parameters, directory="./plots/verification")


# N.B. Do celestial (RA/Dec coords for PSF) - https://iopscience.iop.org/article/10.1088/0067-0049/203/1/4/pdf
# WHEN REFERRING TO PDFs - USE THE TERM LIKELIHOOD INSTEAD OF PROBABILITY WHEN REFERRING TO THE Y-AXIS

# REFERENCES

# FITS Format of gtpsf Output - https://gamma-astro-data-formats.readthedocs.io/en/v0.1/irfs/psf/psf_gtpsf/
