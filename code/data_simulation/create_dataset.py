from map_generation.visualisation import plot_all_sky_map
from map_generation.healpix_maps import (create_background_counts_map, create_expected_counts_map, create_exposure_map,
                                         create_infinite_statistics_map, create_isotropic_background,
                                         create_point_source_map)
from map_generation.utils import angle_to_healpix_pixels, get_nside
from read_write_functions import save_count_maps, xml_parser
from psfs.fit_psf import fit_diffuse_source_psf, fit_point_source_psf
from psfs.visualisation import plot_fitted_point_source_psf


# MAIN PROGRAM

# POINT SOURCE MAPS

# Prepare exposure maps
# exposure_maps, energy_bins = create_exposure_map(
#     exposure_file="/Volumes/T7/project_data/real_data/fermi_filtered_gti_exposure_map.fits")

# Get NSIDE parameter from exposure map
# nside = get_nside(exposure_maps[0])

# Plot exposure maps to verify correctness
# plot_all_sky_map(healpix_maps=exposure_maps, energy_bins=energy_bins, title="Exposure",
#                  directory="./plots/all_sky_maps/")

# THIS IS WHERE TO START THE LOOP OVER THE DIFFERENT MOCK SOURCE CATALOGS

# coordinates, binned_fluxes = xml_parser(energy_bins=energy_bins, xml_file="./simulated_data/sources.xml")
#
# pixels = angle_to_healpix_pixels(coordinates, nside=nside)

# Calculated source locations in lon-lat, the pixels in which they are situated in the healpix map, the binned exposure
# maps of the sky, and their fluxes
# infinite_statistics_maps = create_infinite_statistics_map(exposure_maps, binned_fluxes, pixels)

# Plot infinite counts maps
# plot_all_sky_map(healpix_maps=infinite_statistics_maps, energy_bins=energy_bins, title="Infinite Counts",
#                  directory="./plots/all_sky_maps/", logarithmic=True)

# Sample infinite statistics maps to create expected counts maps
# count_maps = create_expected_counts_map(infinite_counts_map=infinite_statistics_maps)

# Plot infinite statistics count map
# plot_all_sky_map(healpix_maps=count_maps, energy_bins=energy_bins, title="Expected Counts",
#                  directory="./plots/all_sky_maps/", logarithmic=True)

# Create and fit point spread function
# binned_function_parameters = fit_point_source_psf(file_name="/Volumes/T7/project_data/real_data/pointsource_psf.fits")

# Plot PSF fit
# plot_fitted_point_source_psf(psf_file="/Volumes/T7/project_data/real_data/pointsource_psf.fits",
#                              function_parameters=binned_function_parameters, directory="./plots/verification")

# Convolve each PSF with our fitted LAT PSF - i.e. calculate new positions for each gamma ray to originate from - can do
# this directly from each infinite statistics count map
# point_source_maps = create_point_source_map(coordinates=coordinates, exposure_maps=exposure_maps,
#                                            psf_parameters=binned_function_parameters, fluxes=binned_fluxes, nside=nside)


# plot_all_sky_map(healpix_maps=point_source_maps, energy_bins=energy_bins, title="Point Source",
#                  directory="./plots/all_sky_maps/", logarithmic=True)

# Save results
# save_count_maps(count_maps=point_source_maps, directory="./plots/count_maps/", catalog_id=1)

# DIFFUSE SOURCE MAPS

# Create model backgrounds
create_isotropic_background(isotropic_background_file="/Volumes/T7/data/background_models/iso_P8R3_SOURCE_V3_v1.txt")

# diffuse_psf = fit_diffuse_source_psf(roi_count_map="/Volumes/T7/project_data/real_data/diffuse_psf_roi/count_map.fits")
#
# background = create_background_counts_map(6, 7)












# TEST CODE BELOW - DELETE
import healpy as hp

# point_source_read_in = []

# for x in range(5):
#
#     read_in = hp.fitsfunc.read_map("./plots/count_maps/map1/count_map_bin_{}.png".format(x))
#
#     point_source_read_in.append(read_in)
#
# point_source_read_in = np.array(point_source_read_in)
#
# print(point_source_read_in.shape)
#
# plot_all_sky_map(healpix_maps=point_source_read_in, energy_bins=[300, 400, 500, 600, 700, 800], title="TEST READ",
#                  directory="./tmp/", logarithmic=True)

# WHEN REFERRING TO PDFs - USE THE TERM LIKELIHOOD INSTEAD OF PROBABILITY WHEN REFERRING TO THE Y-AXIS

# REFERENCES

# FITS Format of gtpsf Output - https://gamma-astro-data-formats.readthedocs.io/en/v0.1/irfs/psf/psf_gtpsf/
