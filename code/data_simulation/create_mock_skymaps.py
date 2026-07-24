import argparse
import healpy as hp
from map_generation.visualisation import plot_all_sky_map
from map_generation.healpix_maps import (create_diffuse_source_map, create_diffuse_background,
                                         create_expected_counts_map, create_exposure_map,
                                         create_infinite_counts_maps, create_isotropic_background,
                                         create_point_source_map, create_count_map)
from map_generation.utils import angle_to_healpix_pixels, get_nside
import numpy as np
import os
from pathlib import Path
from psfs.fit_psf import fit_diffuse_source_psf, fit_point_source_psf
from psfs.visualisation import plot_fitted_point_source_psf
from read_write_functions import xml_parser


def save_count_maps(binned_maps, save_file, n_bins=5):
    for n in range(n_bins):
        hp.fitsfunc.write_map(filename=save_file.format(n), m=binned_maps[n], coord="G", dtype=np.float64,
                              overwrite=True)


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

    parser.add_argument("--galactic_background", required=True, type=str, help="Location of FITS-formatted"
                                                                               " energy-binned galactic gamma-ray "
                                                                               "background.")

    parser.add_argument("--num_maps_per_catalog", required=True, type=int, help="The number of independent"
                                                                                "sky maps to create per catalog"
                                                                                "provided via Poisson sampling.")

    args = parser.parse_args()

    exposure_fits_file = args.exposure_fits
    num_catalogs = args.number_skymaps
    point_source_psf_file = args.pointsource_psf
    diffuse_source_psf_roi_file = args.diffuse_source_psf_roi
    isotropic_background_file = args.isotropic_background
    galactic_background_file = args.galactic_background
    num_maps_per_catalog = args.num_maps_per_catalog

    # Create directory to store useful simulated data in if it does not already exist
    Path("./simulated_data/utils/").mkdir(parents=True, exist_ok=True)

    # EXPOSURE MAP

    # Prepare exposure maps
    exposure_maps, energy_bins = create_exposure_map(exposure_file=exposure_fits_file)

    # Number of energy bins
    num_bins = len(energy_bins) - 1

    # Save exposure map (it will be the same for every catalog/sky map)
    np.save("./simulated_data/utils/binned_healpix_exposure_maps.npy", exposure_maps)

    # Get NSIDE parameter from exposure map
    nside = get_nside(exposure_maps[0])

    # POINT SPREAD FUNCTIONS (PSF)

    # Create and fit point spread function for point sources
    binned_point_source_psf_parameters = fit_point_source_psf(file_name=point_source_psf_file)

    # Create and fit point spread function for diffuse sources (i.e. background sources)
    binned_diffuse_source_psf = fit_diffuse_source_psf(roi_count_map=diffuse_source_psf_roi_file)

    # BACKGROUND INFINITE STATISTICS MAPS

    # Will sample each time (to create unique backgrounds), but always sample from same infinite statistics map, so only
    # have to generate once

    # Infinite statistics map of isotropic background (energy binned)
    infinite_statistics_isotropic_background = create_isotropic_background(isotropic_background_file=
                                                                           isotropic_background_file,
                                                                           nside=nside,
                                                                           exposure_map=exposure_maps,
                                                                           energy_bins=energy_bins)

    np.save("./simulated_data/utils/binned_healpix_infinite_statistics_isotropic_background_maps.npy",
            infinite_statistics_isotropic_background)

    # Create infinite statistics map of diffuse background - can use for every count map
    infinite_statistics_galactic_diffuse_backgrounds = create_diffuse_background(
        diffuse_background_file=galactic_background_file,
        exposure_map=exposure_maps, nside=nside)

    np.save("./simulated_data/utils/binned_healpix_infinite_statistics_galactic_background_maps.npy",
            infinite_statistics_galactic_diffuse_backgrounds)

    # N.B. Can sample from these multiple times to create independent count maps (even though they have the same source
    # catalogs)

    # Create one infinite count map per catalog
    for c in range(num_catalogs):

        # CREATE INFINITE STATISTICS MAPS FOR POINT SOURCES

        print("Creating Infinite Statistics Count Map {}".format(c + 1))

        catalog_file_location = "./simulated_data/catalogs/catalog_{}".format(c + 1)

        infinite_statistics_directory = "./simulated_data/infinite_count_maps/infinite_count_map_{}".format(c + 1)

        # Create directory to store infinite statistics map for each catalog
        Path(infinite_statistics_directory).mkdir(parents=True, exist_ok=True)

        # Calculate coordinates and binned fluxes from each mock simulated catalog
        agn_coordinates, agn_binned_fluxes = xml_parser(energy_bins, xml_file=catalog_file_location + "/agns.xml")
        pulsar_coordinates, pulsar_binned_fluxes = xml_parser(energy_bins, xml_file=catalog_file_location +
                                                                                    "/pulsars.xml")

        # Convert galactic coordinates of each source to pixel location in HEALPIX-formatted map
        agn_pixels = angle_to_healpix_pixels(agn_coordinates, nside=nside)
        pulsar_pixels = angle_to_healpix_pixels(pulsar_coordinates, nside=nside)

        # Create binned infinite counts maps for each source type
        agn_infinite_counts_map = create_infinite_counts_maps(source_pixels=agn_pixels, exposure_maps=exposure_maps,
                                                              fluxes=agn_binned_fluxes)

        pulsar_infinite_counts_map = create_infinite_counts_maps(source_pixels=pulsar_pixels,
                                                                 exposure_maps=exposure_maps,
                                                                 fluxes=pulsar_binned_fluxes)

        save_count_maps(agn_infinite_counts_map, save_file=infinite_statistics_directory +
                                                           "/agn_infinite_counts_{}.fits")
        save_count_maps(pulsar_infinite_counts_map, save_file=infinite_statistics_directory
                                                              + "/pulsar_infinite_counts_{}.fits")

        # CREATE COUNT MAPS

        for m in range(num_maps_per_catalog):

            # i.e. sample infinite counts map and convolve with PSFs

            # Create background - convolve infinite statistics maps of isotropic and galactic backgrounds with diffuse
            # PSF, then scale with randomly-generated normalisation constant, and Poisson sample to create unique
            # background count map
            diffuse_source_background = create_diffuse_source_map(
                expected_counts_diffuse_background=infinite_statistics_galactic_diffuse_backgrounds,
                expected_counts_isotropic_background=infinite_statistics_isotropic_background,
                psfs=binned_diffuse_source_psf)

            # Create AGN count maps (with PSF convolution and Poisson sampling
            agn_point_source_map = create_count_map(coordinates=agn_coordinates, exposure_maps=exposure_maps,
                                                    psf_parameters=binned_point_source_psf_parameters,
                                                    fluxes=agn_binned_fluxes, nside=nside,
                                                    infinite_stats_file=infinite_statistics_directory +
                                                                       "/agn_infinite_counts_{}.fits")

            # Create pulsar count maps (with PSF convolution and Poisson sampling)
            pulsar_point_source_map = create_count_map(coordinates=pulsar_coordinates, exposure_maps=exposure_maps,
                                                    psf_parameters=binned_point_source_psf_parameters,
                                                    fluxes=pulsar_binned_fluxes, nside=nside,
                                                    infinite_stats_file=infinite_statistics_directory +
                                                                       "/pulsar_infinite_counts_{}.fits")

            # SAVE COUNT MAPS

            save_location = "./simulated_data/count_maps/skymap_{}".format((c * num_maps_per_catalog) + m + 1)

            # Create directory to store simulated count maps in if it does not already exist
            Path(save_location).mkdir(parents=True, exist_ok=True)

            save_count_maps(diffuse_source_background, save_file=save_location + "/background_{}.fits")
            save_count_maps(agn_point_source_map, save_file=save_location + "/agns_{}.fits")
            save_count_maps(pulsar_point_source_map, save_file=save_location + "/pulsars_{}.fits")

    #
    #
    # # Loop over catalogs
    #
    # for c in range(num_catalogs):
    #     print("Creating Sky Map {}".format(c + 1))
    #
    #     file_location = "./simulated_data/catalogs/catalog_{}".format(c + 1)
    #
    #     # Create directory to store useful simulated data in if it does not already exist
    #     Path(file_location).mkdir(parents=True, exist_ok=True)
    #
    #     # Create background - convolve infinite statistics maps of isotropic and galactic backgrounds with diffuse PSF,
    #     # scale with randomly-generated normalisation constant, and Poisson sample to create unique background count map
    #     diffuse_source_background = create_diffuse_source_map(
    #         expected_counts_diffuse_background=infinite_statistics_galactic_diffuse_backgrounds,
    #         expected_counts_isotropic_background=infinite_statistics_isotropic_background,
    #         psfs=binned_diffuse_source_psf)
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
    #     # SAVE COUNT MAPS
    #
    #     save_location = "./simulated_data/count_maps/skymap_{}".format(c + 1)
    #
    #     # Create directory to store simulated count maps in if it does not already exist
    #     Path(save_location).mkdir(parents=True, exist_ok=True)
    #
    #     save_count_maps(diffuse_source_background, save_file=save_location + "/background_{}.fits")
    #     save_count_maps(agn_point_source_maps, save_file=save_location + "/agns_{}.fits")
    #     save_count_maps(pulsar_point_source_maps, save_file=save_location + "/pulsars_{}.fits")
    #

# coordinates, binned_fluxes = xml_parser(energy_bins, xml_file="./simulated_data/sources.xml")
#
# pixels = angle_to_healpix_pixels(coordinates, nside=nside)
#
# # Calculated source locations in lon-lat, the pixels in which they are situated in the healpix map, the binned exposure
# # maps of the sky, and their fluxes
# # infinite_statistics_maps = create_infinite_statistics_map(exposure_maps, binned_fluxes, pixels)
#
# # Sample infinite statistics maps to create expected counts maps
# # count_maps = create_expected_counts_map(infinite_counts_map=infinite_statistics_maps)
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

# # THIS IS FOR TESTING

# visualise_maps(energy_bins=energy_bins, exposure_map=exposure_maps, point_source_map=point_source_maps,
#                diffuse_source_map=diffuse_source_background, catalog_id=1)

# REFERENCES

# FITS Format of gtpsf Output - https://gamma-astro-data-formats.readthedocs.io/en/v0.1/irfs/psf/psf_gtpsf/
