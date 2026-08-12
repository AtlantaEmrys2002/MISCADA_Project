"""
Main file for creating a series of real exposure maps and simulated infinite statistics expected counts and count maps
based on a series of simulated gamma-ray source catalogs.
"""

import argparse
import copy
from map_generation.healpix_maps import (create_diffuse_infinite_statistics_background,
                                         create_exposure_map, create_infinite_counts_maps,
                                         create_isotropic_infinite_statistics_background)
from map_generation.utils import angle_to_healpix_pixels, get_nside
from multiprocessing import Pool
import numpy as np
import os
from psfs.fit_psf import fit_diffuse_source_psf, fit_point_source_psf
from verification.visualisation import plot_fitted_point_source_psf
import time
from map_generation.healpix_maps import create_count_map_2, create_diffuse_source_map
from pathlib import Path
from read_write_functions import save_count_maps, xml_parser
from verification.visualisation import plot_all_sky_map


def create_all_count_maps(params: list) -> None:
    """
    Creates num_maps_per_catalog count maps for catalog catalog_id.
    """
    (catalog_id, num_maps_per_catalog, energy_bins, nside, exposure_maps, binned_point_source_psf_parameters,
     binned_diffuse_source_psf, infinite_statistics_galactic_diffuse_backgrounds,
     infinite_statistics_isotropic_background) = params

    catalog_file_location = "./simulated_data/catalogs/catalog_{}".format(catalog_id + 1)
    infinite_statistics_directory = "./simulated_data/infinite_count_maps_2/infinite_count_map_{}".format(
        catalog_id + 1)

    # Calculate coordinates and binned fluxes from each mock simulated catalog
    agn_coordinates, agn_binned_fluxes = xml_parser(energy_bins, xml_file=catalog_file_location + "/agns.xml")
    pulsar_coordinates, pulsar_binned_fluxes = (
        xml_parser(energy_bins, xml_file=catalog_file_location + "/pulsars.xml"))

    # CREATE COUNT MAPS

    for m in range(num_maps_per_catalog):

        print("Creating Count Map {}".format((catalog_id * num_maps_per_catalog) + m + 1))

        # i.e. sample infinite counts map and convolve with PSFs

        # Create background - convolve infinite statistics maps of isotropic and galactic backgrounds with diffuse
        # PSF, then scale with randomly-generated normalisation constant, and Poisson sample to create unique
        # background count map
        diffuse_source_background = create_diffuse_source_map(
            expected_counts_diffuse_background=infinite_statistics_galactic_diffuse_backgrounds,
            expected_counts_isotropic_background=infinite_statistics_isotropic_background,
            psfs=binned_diffuse_source_psf)

        # Create AGN count maps (with PSF convolution and Poisson sampling
        agn_point_source_map = (
            create_count_map_2(coordinates=agn_coordinates, exposure_maps=exposure_maps,
                               psf_parameters=binned_point_source_psf_parameters, fluxes=agn_binned_fluxes,
                               nside=nside, energy_bins=energy_bins, use_energy_bins=True,
                               infinite_stats_file=infinite_statistics_directory + "/agn_infinite_counts_{}.fits"))

        # Create pulsar count maps (with PSF convolution and Poisson sampling)
        pulsar_point_source_map = (
            create_count_map_2(coordinates=pulsar_coordinates, exposure_maps=exposure_maps,
                               psf_parameters=binned_point_source_psf_parameters, fluxes=pulsar_binned_fluxes,
                               nside=nside, energy_bins=energy_bins, use_energy_bins=True,
                               infinite_stats_file=infinite_statistics_directory + "/pulsar_infinite_counts_{}.fits"))

        # SAVE COUNT MAPS

        save_location = "./simulated_data/count_maps_2/skymap_{}".format((catalog_id * num_maps_per_catalog) + m + 1)

        # Create directory to store simulated count maps in if it does not already exist
        Path(save_location).mkdir(parents=True, exist_ok=True)

        save_count_maps(diffuse_source_background, save_file=save_location + "/background_{}.fits")
        save_count_maps(agn_point_source_map, save_file=save_location + "/agns_{}.fits")
        save_count_maps(pulsar_point_source_map, save_file=save_location + "/pulsars_{}.fits")

        # Save background
        if catalog_id == 0 and m == 0:
            plot_all_sky_map(healpix_maps=pulsar_point_source_map, energy_bins=energy_bins, title="Pulsar Count",
                             directory="./plots/all_sky_maps_2/", logarithmic=True)
            plot_all_sky_map(healpix_maps=agn_point_source_map, energy_bins=energy_bins, title="AGN Count",
                             directory="./plots/all_sky_maps_2/", logarithmic=True)
            plot_all_sky_map(healpix_maps=diffuse_source_background, energy_bins=energy_bins,
                             title="Background Count", directory="./plots/all_sky_maps_2/", logarithmic=True)

            actual_count_map = pulsar_point_source_map + agn_point_source_map + diffuse_source_background
            plot_all_sky_map(healpix_maps=actual_count_map, energy_bins=energy_bins,
                             title="Count", directory="./plots/all_sky_maps_2/", logarithmic=True)


if __name__ == "__main__":

    # INPUT PARAMETER PARSING

    parser = argparse.ArgumentParser(description="Generate all-sky count maps for each specified source catalog, "
                                                 "dividing them into agn, pulsar, and background.")

    parser.add_argument("--exposure_fits", required=True, type=str, help="FITS-formatted file that stores"
                                                                         "Fermi-LAT telescope exposure, i.e. energy-"
                                                                         "binned file specifying how long telescope is"
                                                                         "pointed at specific area of the sky.")

    parser.add_argument("--number_catalogs", required=True, type=int, help="This specifies the number of "
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
    num_catalogs = args.number_catalogs
    point_source_psf_file = args.pointsource_psf
    diffuse_source_psf_roi_file = args.diffuse_source_psf_roi
    isotropic_background_file = args.isotropic_background
    galactic_background_file = args.galactic_background
    num_maps_per_catalog = args.num_maps_per_catalog

    # Create directory to store useful simulated data in if it does not already exist
    Path("./simulated_data/utils_2/").mkdir(parents=True, exist_ok=True)

    # Create directory to store visualisations of each stage of the process if it does not already exist
    Path("./plots/all_sky_maps_2/").mkdir(parents=True, exist_ok=True)

    # EXPOSURE MAP

    start_exp_time = time.time()

    # Prepare exposure maps
    exposure_maps, energy_bins = create_exposure_map(exposure_file=exposure_fits_file)

    print("Time to create exposure maps: {} s".format(time.time() - start_exp_time))

    # Plot exposure maps
    plot_all_sky_map(healpix_maps=exposure_maps, energy_bins=energy_bins, title="Exposure",
                     directory="./plots/all_sky_maps_2/")

    # Number of energy bins
    num_bins = len(energy_bins) - 1

    # Save exposure map (it will be the same for every catalog/sky map)
    np.save("./simulated_data/utils_2/binned_healpix_exposure_maps.npy", exposure_maps)

    # Get NSIDE parameter from exposure map
    nside = get_nside(exposure_maps[0])

    # POINT SPREAD FUNCTIONS (PSF)

    start_psf_time = time.time()

    # Create and fit point spread function for point sources
    binned_point_source_psf_parameters = fit_point_source_psf(file_name=point_source_psf_file)

    # Create and fit point spread function for diffuse sources (i.e. background sources)
    binned_diffuse_source_psf = fit_diffuse_source_psf(roi_count_map=diffuse_source_psf_roi_file)

    print("Time to create diffuse and point source PSFs: {} s".format(time.time() - start_exp_time))

    plot_fitted_point_source_psf(psf_file=point_source_psf_file, function_parameters=binned_point_source_psf_parameters,
                                 directory="./plots/verification")

    # BACKGROUND INFINITE STATISTICS MAPS

    # Will sample each time (to create unique backgrounds), but always sample from same infinite statistics map, so only
    # have to generate once

    start_background_time = time.time()

    # Infinite statistics map of isotropic background (energy binned)
    infinite_statistics_isotropic_background = (
        create_isotropic_infinite_statistics_background(isotropic_background_file=isotropic_background_file,
                                                        nside=nside, exposure_map=exposure_maps,
                                                        energy_bins=energy_bins))

    # Create infinite statistics map of diffuse background - can use for every count map
    infinite_statistics_galactic_diffuse_backgrounds = create_diffuse_infinite_statistics_background(
        diffuse_background_file=galactic_background_file, exposure_map=exposure_maps, nside=nside)

    print("Time to create infinite statistics background maps: {} s".format(time.time() - start_background_time))

    np.save("./simulated_data/utils_2/binned_healpix_infinite_statistics_isotropic_background_maps.npy",
            infinite_statistics_isotropic_background)

    np.save("./simulated_data/utils_2/binned_healpix_infinite_statistics_galactic_background_maps.npy",
            infinite_statistics_galactic_diffuse_backgrounds)

    plot_all_sky_map(healpix_maps=infinite_statistics_galactic_diffuse_backgrounds, energy_bins=energy_bins,
                     title="Infinite Statistics Diffuse Background", directory="./plots/all_sky_maps_2/",
                     logarithmic=True)

    # N.B. Can sample from these multiple times to create independent count maps (even though they have the same source
    # catalogs)

    av_time_infinite_stats_maps = 0

    # Create one infinite count map per catalog
    for c in range(num_catalogs):

        # CREATE INFINITE STATISTICS MAPS FOR POINT SOURCES

        print("Creating Infinite Statistics Count Map {}".format(c + 1))

        catalog_file_location = "./simulated_data/catalogs/catalog_{}".format(c + 1)

        infinite_statistics_directory = "./simulated_data/infinite_count_maps_2/infinite_count_map_{}".format(c + 1)

        # Create directory to store infinite statistics map for each catalog
        Path(infinite_statistics_directory).mkdir(parents=True, exist_ok=True)

        start_infinite_time = time.time()

        # Calculate coordinates and binned fluxes from each mock simulated catalog
        agn_coordinates, agn_binned_fluxes = xml_parser(energy_bins, xml_file=catalog_file_location + "/agns.xml")
        pulsar_coordinates, pulsar_binned_fluxes = (
            xml_parser(energy_bins, xml_file=catalog_file_location + "/pulsars.xml"))

        # Convert galactic coordinates of each source to pixel location in HEALPIX-formatted map
        agn_pixels = angle_to_healpix_pixels(agn_coordinates, nside=nside)
        pulsar_pixels = angle_to_healpix_pixels(pulsar_coordinates, nside=nside)

        # Create binned infinite counts maps for each source type
        agn_infinite_counts_map = create_infinite_counts_maps(source_pixels=agn_pixels, exposure_maps=exposure_maps,
                                                              fluxes=agn_binned_fluxes)

        pulsar_infinite_counts_map = create_infinite_counts_maps(source_pixels=pulsar_pixels,
                                                                 exposure_maps=exposure_maps,
                                                                 fluxes=pulsar_binned_fluxes)

        av_time_infinite_stats_maps += (time.time() - start_infinite_time)

        # Save infinite statistics count maps for the catalogs - can then sample them many times to create independent
        # count maps
        save_count_maps(agn_infinite_counts_map,
                        save_file=infinite_statistics_directory + "/agn_infinite_counts_{}.fits")
        save_count_maps(pulsar_infinite_counts_map,
                        save_file=infinite_statistics_directory + "/pulsar_infinite_counts_{}.fits")

        # Plot these maps if this is the first catalog being analysed
        if c == 0:
            plot_all_sky_map(healpix_maps=agn_infinite_counts_map, energy_bins=energy_bins,
                             title="Infinite Statistics AGN", directory="./plots/all_sky_maps_2/", logarithmic=True)
            plot_all_sky_map(healpix_maps=pulsar_infinite_counts_map, energy_bins=energy_bins,
                             title="Infinite Statistics Pulsar", directory="./plots/all_sky_maps_2/", logarithmic=True)

    print("Total time to create infinite statistics maps: {} s".format(av_time_infinite_stats_maps))
    print("Average time to create an infinite statistics map: {} s".format(av_time_infinite_stats_maps / num_catalogs))

    # Set up arguments for passing to create_all_count_maps to calculate the actual counts for the diffuse, AGN, and
    # pulsar sources.
    arguments = [[c, num_maps_per_catalog, copy.deepcopy(energy_bins), nside, copy.deepcopy(exposure_maps),
                  copy.deepcopy(binned_point_source_psf_parameters), copy.deepcopy(binned_diffuse_source_psf),
                  copy.deepcopy(infinite_statistics_galactic_diffuse_backgrounds),
                  copy.deepcopy(infinite_statistics_isotropic_background)] for c in range(num_catalogs)]

    # Parallel computation of maps - split catalogs between CPUs
    start_count_time = time.time()

    pool = Pool(processes=os.cpu_count())

    pool.map(create_all_count_maps, arguments)

    pool.terminate()

    print("Total time to create count maps: {} s".format(time.time() - start_count_time))
    print("Average time to create a count map: {} s".format((time.time() - start_count_time) /
                                                            (num_catalogs * num_maps_per_catalog)))

# REFERENCES

# FITS Format of gtpsf Output - https://gamma-astro-data-formats.readthedocs.io/en/v0.1/irfs/psf/psf_gtpsf/
# HEALPIX - https://gamma-astro-data-formats.readthedocs.io/en/latest/skymaps/healpix/
# Info about PSF - https://docs.gammapy.org/0.20.1/tutorials/data/fermi_lat.html#PSF
# Python Parallel - https://www.reddit.com/r/learnpython/comments/8f2p4d/call_a_function_with_multiple_arguments_with/
# Visualisation Confirmation - https://nptfit.readthedocs.io/en/latest/Example1_Overview_of_the_Fermi_Data.html
