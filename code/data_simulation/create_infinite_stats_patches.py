"""
Create patches from infinite statistics maps - this is to compare and verify stages of dataset creation pipeline.
"""

import argparse
import copy
import healpy as hp
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import os
from pathlib import Path
import time
from verification.visualisation import plot_patch


def create_infinite_patches_for_catalog(params: list):
    """Creates patches from sky maps all created from the same simulated source catalog and returns information about
    those patches.

    Parameters
    ----------
    params : list
        Contains all the relevant information for creating these patches.

    Returns
    ------
    list
        Metadata about each patch created.

    """
    (save_directory, num_maps_per_catalog, catalog_id, energy_bins, max_patches_per_catalog, pix_sr, longitude,
     latitude, xsize_patch_generation, xsize_location, solid_area_ratio) = params

    num_bins = energy_bins.shape[0] - 1

    # To ensure 10 x 10 degree patches
    lb_range = [-5, 5]

    for m in range(num_maps_per_catalog):

        # READ IN MAPS AND SPATIAL/SPECTRAL PARAMETERS

        skymaps_directory = save_directory + "/count_maps/infinite_count_map_{}".format(catalog_id)

        binned_agn_map = []
        binned_pulsar_map = []
        binned_isotropic_background_map = []
        binned_galactic_background_map = []

        for b in range(num_bins):
            # Read in binned all-sky count maps for AGN, pulsars, and background
            agns = hp.fitsfunc.read_map(filename=skymaps_directory + "/agn_infinite_counts_{}.fits".format(b),
                                        field=None)
            pulsars = hp.fitsfunc.read_map(filename=skymaps_directory + "/pulsars_infinite_counts_{}.fits".format(b),
                                           field=None)

            isotropic_background = np.load(save_directory +
                                           "/utils/binned_healpix_infinite_statistics_isotropic_background_maps.npy")[b]
            galactic_background = np.load(save_directory +
                                          "/utils/binned_healpix_infinite_statistics_galactic_background_maps.npy")[b]

            binned_agn_map.append(agns / pix_sr)
            binned_pulsar_map.append(pulsars / pix_sr)
            binned_galactic_background_map.append(galactic_background / pix_sr)
            binned_isotropic_background_map.append(isotropic_background / pix_sr)

        # Convert to numpy array
        binned_agn_map = np.array(binned_agn_map)
        binned_pulsar_map = np.array(binned_pulsar_map)
        binned_galactic_background_map = np.array(binned_galactic_background_map)
        binned_isotropic_background_map = np.array(binned_isotropic_background_map)

        # Loop to generate each patch

        for p in range(max_patches_per_catalog):

            # Check if patch already exists

            # Unique identifier of patch being generated
            patch_id = (catalog_id * num_maps_per_catalog) + p

            print("PATCH: {}".format(patch_id))

            # Create directory where patches stored
            patch_directory = save_directory + "/infinite_stats_patches/patch_{}/".format(patch_id)

            # CENTRE OF PATCH p

            # transformation from 0-360 to -180-180 - healpy uses this coordinate system
            lon = (longitude[p] + 180) % 360 - 180

            lat = latitude[p]

            # GET NO. AGN AND PULSARS WITHIN PATCH, AS WELL AS THE CARTESIAN COORDINATES OF AGN AND PULSARS IN THE
            # PATCH

            # AGN and Pulsar coordinates must be in lon-lat format (l, b) with degree values

            Path(patch_directory).mkdir(parents=True, exist_ok=True)

            # PROJECT ROI OF COUNT MAP INTO CARTESIAN

            binned_agn_patch = []
            binned_pulsar_patch = []
            binned_isotropic_background_patch = []
            binned_galactic_background_patch = []

            plt.cla()
            plt.clf()
            plt.close("all")

            for b in range(num_bins):
                agn_patch_bin = hp.visufunc.cartview(binned_agn_map[b], rot=(lon, lat, 0.), coord='G',
                                                     xsize=xsize_patch_generation, lonra=lb_range, latra=lb_range,
                                                     return_projected_map=True)

                binned_agn_patch.append(np.array(agn_patch_bin) * solid_area_ratio)

                plt.cla()
                plt.clf()
                plt.close("all")

                pulsar_patch_bin = hp.visufunc.cartview(binned_pulsar_map[b], rot=(lon, lat, 0.), coord='G',
                                                        xsize=xsize_patch_generation, lonra=lb_range,
                                                        latra=lb_range, return_projected_map=True)

                binned_pulsar_patch.append(np.array(pulsar_patch_bin) * solid_area_ratio)
                plt.cla()
                plt.clf()
                plt.close("all")

                background_isotropic_patch_bin = hp.visufunc.cartview(binned_isotropic_background_map[b],
                                                                      rot=(lon, lat, 0.),
                                                                      coord='G',
                                                                      xsize=xsize_patch_generation, lonra=lb_range,
                                                                      latra=lb_range, return_projected_map=True)

                binned_isotropic_background_patch.append(np.array(background_isotropic_patch_bin) * solid_area_ratio)

                background_galactic_patch_bin = hp.visufunc.cartview(binned_galactic_background_map[b],
                                                                     rot=(lon, lat, 0.),
                                                                     coord='G',
                                                                     xsize=xsize_patch_generation, lonra=lb_range,
                                                                     latra=lb_range, return_projected_map=True)

                binned_galactic_background_patch.append(np.array(background_galactic_patch_bin) * solid_area_ratio)

                # Need these here (even though we are not showing the plots - this is because visufunc creates a
                # plot - we only need the 2D array)
                plt.cla()
                plt.clf()
                plt.close("all")

            # Convert to numpy
            binned_agn_patch = np.array(binned_agn_patch)
            binned_pulsar_patch = np.array(binned_pulsar_patch)
            binned_background_patch = np.array(binned_isotropic_background_patch) + np.array(
                binned_galactic_background_patch)

            # Sum together to create patch
            patch = binned_agn_patch + binned_pulsar_patch + binned_background_patch

            np.save(patch_directory + "/patch.npy", patch)

            # Plot all patches generated from the first skymap
            if catalog_id == 0 and m == 0:
                plot_patch(binned_patches=patch, mask=np.zeros((64, 64)), unformatted_energy_bins=energy_bins,
                           directory=patch_directory)


if __name__ == "__main__":
    # INPUT PARAMETERS AND DATA

    parser = argparse.ArgumentParser(description="Read in simulated all-sky maps and catalogs to create a dataset of "
                                                 "ROIs to be fed to source detection algorithms.")

    parser.add_argument("--num_catalogs", required=True, type=int, help="The number of simulated catalogs "
                                                                        "that have been generated and can be"
                                                                        "split into patches.")

    parser.add_argument("--num_skymaps_per_catalog", required=True, type=int, help="The number of sky maps derived "
                                                                                   "per catalog of sources.")

    parser.add_argument("--num_energy_bins", required=True, type=int, help="The number of energy bins that "
                                                                           "photons were binned into using fermitools.")

    parser.add_argument("--patch_directory", required=True, type=str, help="Directory in which to save "
                                                                           "generated patches")

    args = parser.parse_args()

    num_maps_per_catalog = args.num_skymaps_per_catalog
    num_catalogs = args.num_catalogs
    num_bins = args.num_energy_bins
    save_directory = args.patch_directory

    # FILE CREATION AND ORGANISATION

    energy_bins = np.logspace(np.log(300), np.log(200000), num=num_bins + 1, base=np.e)

    # Size of patch side when calculating Cartesian positions in image
    xsize_location = 128

    # Create template grid on which to project patch
    coord_range = np.linspace(-4.9609375, 4.9609375, 128)

    X, Y = np.meshgrid(coord_range, coord_range)
    lonlat_patch_template = list(zip(np.flip(X.flatten()), Y.flatten()))

    np.save("template.npy", lonlat_patch_template)

    ##################################################
    # global 64x64 correction
    ##################################################
    xsize_patch_generation = 64
    NSIDE = 256

    Npix = 12 * NSIDE * NSIDE
    pix_sr = 4.0 * np.pi / Npix

    lb_range = [-5, 5]
    # noinspection PyUnresolvedReferences
    proj = hp.projector.CartesianProj(lonra=lb_range, latra=lb_range, xsize=xsize_patch_generation)

    I, J = np.meshgrid(np.arange(xsize_patch_generation), np.arange(xsize_patch_generation))
    x, y = proj.ij2xy(I, J)

    # Used to correct the rotation smear
    dl = np.radians((lb_range[1] - lb_range[0]) / (xsize_patch_generation - 1))
    solid_area_ratio = dl * dl * np.cos(np.radians(y))

    # Gulli's approach to generate a more uniform coverage of the sky - specifies the longitudes at which to draw out
    # the slice of the sky to take
    longitude, latitude = hp.pix2ang(8, np.arange(hp.nside2npix(8)), lonlat=True)

    # The number of patches to generate per catalog
    max_patches_per_catalog = len(longitude)

    # CREATE CSV FILE FOR STORING PATCH INFORMATION

    patch_information = []

    # Num patches to generate
    patches = max_patches_per_catalog

    arguments = [[save_directory, num_maps_per_catalog, c, copy.deepcopy(energy_bins), max_patches_per_catalog, pix_sr,
                  copy.deepcopy(longitude), copy.deepcopy(latitude), xsize_patch_generation, xsize_location,
                  solid_area_ratio] for c in range(num_catalogs)]

    # Parallel computation of maps - split catalogs between CPUs
    start_count_time = time.time()

    pool = Pool(processes=os.cpu_count() // 8)

    pool.map(create_infinite_patches_for_catalog, arguments)

    pool.terminate()

    print("Time to create patches: {} s".format(time.time() - start_count_time))
    print("Average time to create patches per sky map {} s".format((time.time() - start_count_time) /
                                                                   (num_catalogs * num_maps_per_catalog)))
