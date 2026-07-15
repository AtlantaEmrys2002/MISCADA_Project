import argparse
from map_generation.visualisation import plot_all_sky_map
from map_generation.healpix_maps import (create_diffuse_source_map, create_diffuse_background,
                                         create_expected_counts_map, create_exposure_map,
                                         create_infinite_statistics_map, create_isotropic_background,
                                         create_point_source_map)
from map_generation.utils import angle_to_healpix_pixels, get_nside
import numpy as np
import os
from read_write_functions import save_count_maps, xml_parser
from pathlib import Path
from psfs.fit_psf import fit_diffuse_source_psf, fit_point_source_psf
from psfs.visualisation import plot_fitted_point_source_psf
from read_write_functions import xml_parser


def visualise_maps(energy_bins, exposure_map, point_source_map, diffuse_source_map, catalog_id):
    # EXPOSURE MAP

    # Visualise exposure map to verify correctness - the exposure map will be the same for each simulated sky map, so
    # there is no need to consistently replot
    if not os.path.isfile("./plots/all_sky_maps/exposure_map.png"):
        # Create directory if it does not already exist
        Path("./plots/all_sky_maps/").mkdir(parents=True, exist_ok=True)

        plot_all_sky_map(healpix_maps=exposure_map, energy_bins=energy_bins, title="Exposure",
                         directory="./plots/all_sky_maps/".format(catalog_id))

    # Names of each plot
    titles = ["Point Source", "Diffuse Source"]

    maps = [point_source_map, diffuse_source_map]

    num_maps = len(maps)

    for m in range(num_maps):
        plot_all_sky_map(healpix_maps=maps[m], energy_bins=energy_bins, title=titles[m],
                         directory="./plots/all_sky_maps/catalog_{}/".format(catalog_id), logarithmic=True)


# def create_dataset():
#
#     # Creates a dataset from a single catalog

if __name__ == "__main__":

    # INPUT PARAMETER PARSING

    parser = argparse.ArgumentParser(description="Generate all-sky count maps for each specified source catalog, "
                                                 "dividing them into agn, pulsar, and background.")

    parser.add_argument("--exposure_fits", required=True, type=str, help="FITS-formatted file that stores"
                                                                         "Fermi-LAT telescope exposure, i.e. energy-"
                                                                         "binned file specifying how long telescope is"
                                                                         "pointed at specific area of the sky.")

    parser.add_argument("--number_skymaps", required=True, type=int, help="This specifies the number of "
                                                                          "catalogs from which to generate all-sky "
                                                                          "simulated source count maps.")

    parser.add_argument("--pointsource_psf", required=True, type=str, help="This is a FITS-formatted file consistent "
                                                                           "with that produced by gtpsf to describe the"
                                                                           "point spread function that \"blurs\" point "
                                                                           "sources")

    parser.add_argument("--diffuse_source_psf_roi", required=True, type=str, help="This is a "
                                                                                  "FITS-formatted file containing a "
                                                                                  "count map of the ROI from which the "
                                                                                  "diffuse PSF is derived.")

    parser.add_argument("--isotropic_background", required=True, type=str, help="File location of energy-"
                                                                                "binned isotropic gamma-ray background."
                                                                                "")

    args = parser.parse_args()

    exposure_fits_file = args.exposure_fits
    num_catalogs = args.number_skymaps
    point_source_psf_file = args.pointsource_psf
    diffuse_source_psf_roi_file = args.diffuse_source_psf_roi
    isotropic_background_file = args.isotropic_background

    # Create directory to store useful simulated data in if it does not already exist
    Path("./simulated_data/utils/").mkdir(parents=True, exist_ok=True)

    # EXPOSURE MAP

    # Prepare exposure maps
    exposure_maps, energy_bins = create_exposure_map(exposure_file=exposure_fits_file)

    # Save exposure map (it will be the same for every catalog/sky map)
    np.save("./simulated_data/utils/binned_healpix_exposure_maps.npy", exposure_maps)

    # Get NSIDE parameter from exposure map
    nside = get_nside(exposure_maps[0])

    # POINT SPREAD FUNCTIONS (PSF)

    # Create and fit point spread function for point sources
    binned_point_source_psf_parameters = fit_point_source_psf(file_name=point_source_psf_file)

    # Create and fit point spread function for diffuse sources (i.e. background sources)
    # binned_diffuse_source_psf = fit_diffuse_source_psf(roi_count_map="/Volumes/T7/project_data/real_data/diffuse_psf_roi/count_map.fits")
    binned_diffuse_source_psf = fit_diffuse_source_psf(roi_count_map=diffuse_source_psf_roi_file)

    # BACKGROUND

    # Infinite statistics map of isotropic background (energy binned)
    # infinite_statistis_isotropic_background = create_isotropic_background(isotropic_background_file=
    #                                                     "/Volumes/T7/data/background_models/iso_P8R3_SOURCE_V3_v1.txt",
    #                                                     nside=nside,
    #                                                     exposure_map=exposure_maps, energy_bins=energy_bins)

    infinite_statistics_isotropic_background = create_isotropic_background(isotropic_background_file=
                                                                          isotropic_background_file,
                                                                          nside=nside,
                                                                          exposure_map=exposure_maps,
                                                                          energy_bins=energy_bins)

    print(infinite_statistics_isotropic_background.shape)

    # Loop over catalogs

    # for c in range(num_catalogs):
    #     file_location = "./simulated_data/catalogs/catalog_{}".format(c + 1)
    #
    #     # Calculate coordinates and binned fluxes from each mock simulated catalog
    #     agn_coordinates, agn_binned_fluxes = xml_parser(energy_bins, xml_file=file_location + "/agns.xml")
    #     pulsar_coordinates, pulsar_binned_fluxes = xml_parser(energy_bins, xml_file=file_location + "/pulsars.xml")
    #
    #     # Convert galactic coordinates of each source to pixel location in HEALPIX-formatted map
    #     agn_pixels = angle_to_healpix_pixels(agn_coordinates, nside=nside)
    #     pulsar_pixels = angle_to_healpix_pixels(pulsar_coordinates, nside=nside)
    #
    #     # Convolve each PSF with our fitted LAT PSF - i.e. calculate new positions for each gamma ray to originate from
    #     # - can do this directly from each infinite statistics count map (instead of method described in Robust Neural
    #     # paper)
    #     agn_point_source_maps = create_point_source_map(coordinates=agn_coordinates, exposure_maps=exposure_maps,
    #                                                     psf_parameters=binned_point_source_psf_parameters,
    #                                                     fluxes=agn_binned_fluxes, nside=nside)
    #
    #     pulsar_point_source_maps = create_point_source_map(coordinates=pulsar_coordinates, exposure_maps=exposure_maps,
    #                                                        psf_parameters=binned_point_source_psf_parameters,
    #                                                        fluxes=pulsar_binned_fluxes, nside=nside)
    #




# # MAIN PROGRAM
#
# # POINT SOURCE MAPS
#
# # Prepare exposure maps
# exposure_maps, energy_bins = create_exposure_map(
#     exposure_file="/Volumes/T7/project_data/real_data/fermi_filtered_gti_exposure_map.fits")
#
# # # Get NSIDE parameter from exposure map
# nside = get_nside(exposure_maps[0])
#
# # THIS IS WHERE TO START THE LOOP OVER THE DIFFERENT MOCK SOURCE CATALOGS
#
# coordinates, binned_fluxes = xml_parser(energy_bins, xml_file="./simulated_data/sources.xml")
#
# pixels = angle_to_healpix_pixels(coordinates, nside=nside)
#
# # Calculated source locations in lon-lat, the pixels in which they are situated in the healpix map, the binned exposure
# # maps of the sky, and their fluxes
# # infinite_statistics_maps = create_infinite_statistics_map(exposure_maps, binned_fluxes, pixels)
#
# # # Plot infinite counts maps
# # plot_all_sky_map(healpix_maps=infinite_statistics_maps, energy_bins=energy_bins, title="Infinite Counts",
# #                  directory="./plots/all_sky_maps/", logarithmic=True)
#
# # Sample infinite statistics maps to create expected counts maps
# # count_maps = create_expected_counts_map(infinite_counts_map=infinite_statistics_maps)
#
# # Plot infinite statistics count map
# # plot_all_sky_map(healpix_maps=count_maps, energy_bins=energy_bins, title="Expected Counts",
# #                  directory="./plots/all_sky_maps/", logarithmic=True)
#
# # Create and fit point spread function
# binned_function_parameters = fit_point_source_psf(file_name="/Volumes/T7/project_data/real_data/pointsource_psf.fits")
#
# # Plot PSF fit
# plot_fitted_point_source_psf(psf_file="/Volumes/T7/project_data/real_data/pointsource_psf.fits",
#                              function_parameters=binned_function_parameters, directory="./plots/verification")
#
# # Convolve each PSF with our fitted LAT PSF - i.e. calculate new positions for each gamma ray to originate from - can do
# # this directly from each infinite statistics count map
# # point_source_maps = create_point_source_map(coordinates=coordinates, exposure_maps=exposure_maps,
# #                                            psf_parameters=binned_function_parameters, fluxes=binned_fluxes, nside=nside)
#
# # plot_all_sky_map(healpix_maps=point_source_maps, energy_bins=energy_bins, title="Point Source",
# #                  directory="./plots/all_sky_maps/", logarithmic=True)
#
# # THIS IS FOR TESTING
#
# import numpy as np
# point_source_maps = np.load("point_source_tmp.npy")
#
#
#
# # Save results
# # save_count_maps(count_maps=point_source_maps, directory="./simulated_data/count_maps/", catalog_id=1)
#
# # DIFFUSE SOURCE MAPS
#
# # Create backgrounds models
# isotropic_backgrounds = create_isotropic_background(isotropic_background_file=
#                                                     "/Volumes/T7/data/background_models/iso_P8R3_SOURCE_V3_v1.txt",
#                                                     nside=nside,
#                                                     exposure_map=exposure_maps, energy_bins=energy_bins)
#
# # plot_all_sky_map(healpix_maps=isotropic_backgrounds, energy_bins=energy_bins, title="Isotropic Background",
# #                  directory="./plots/all_sky_maps/", logarithmic=True)
#
# # Create infinite statistics map of diffuse background
# galactic_diffuse_backgrounds = create_diffuse_background(
#     diffuse_background_file="/Volumes/T7/data/background_models/gll_iem_v07.fits", exposure_map=exposure_maps,
#     nside=nside)
#
# # plot_all_sky_map(healpix_maps=galactic_diffuse_backgrounds, energy_bins=energy_bins, title="Expected Diffuse Background",
# #                  directory="./plots/all_sky_maps/", logarithmic=True)
#
# # Fit PSF for diffuse background
#
# diffuse_psf = fit_diffuse_source_psf(roi_count_map="/Volumes/T7/project_data/real_data/diffuse_psf_roi/count_map.fits")
#
# # CONVOLVE GALACTIC AND ISOTROPIC BACKGROUND MAPS WITH PSF IN BELOW FUNC - ADD IN ARGUMENT TO PASS THE PSFS
#
# diffuse_source_background = create_diffuse_source_map(expected_counts_diffuse_background=galactic_diffuse_backgrounds,
#                                        expected_counts_isotropic_background=isotropic_backgrounds, psfs=diffuse_psf)
#
# # plot_all_sky_map(healpix_maps=diffuse_source_background, energy_bins=energy_bins, title="Diffuse Source Background",
# #                  directory="./plots/all_sky_maps/", logarithmic=True)
#
#
# visualise_maps(energy_bins=energy_bins, exposure_map=exposure_maps, point_source_map=point_source_maps,
#                diffuse_source_map=diffuse_source_background, catalog_id=1)

# WHEN REFERRING TO PDFs - USE THE TERM LIKELIHOOD INSTEAD OF PROBABILITY WHEN REFERRING TO THE Y-AXIS

# REFERENCES

# FITS Format of gtpsf Output - https://gamma-astro-data-formats.readthedocs.io/en/v0.1/irfs/psf/psf_gtpsf/
