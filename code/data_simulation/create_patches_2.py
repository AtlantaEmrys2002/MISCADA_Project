"""
Main function for creating patches from skymaps. Placed in separate file for ease of parallelization.
"""

from formatting.mask_creation import create_mask
from formatting.projection_tools import get_ps_info_128
import healpy as hp
from map_generation.utils import cartesian_patch
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from read_write_functions import xml_parser
import time
from verification.visualisation import plot_patch

import copy


def create_patches_for_catalog_2(params: list):
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

    catalogs_directory = save_directory + "/catalogs/catalog_{}".format(catalog_id + 1)

    num_bins = energy_bins.shape[0] - 1

    # To ensure 10 x 10 degree patches
    lb_range = [-5, 5]

    patch_information = []

    for m in range(num_maps_per_catalog):

        # READ IN MAPS AND SPATIAL/SPECTRAL PARAMETERS

        skymaps_directory = save_directory + "/count_maps_2/skymap_{}".format(
            (catalog_id * num_maps_per_catalog) + m + 1)

        # Read in the coordinates and associated integral photon fluxes of each source in AGN and pulsar maps

        agn_coordinates, _, agn_ids = xml_parser(energy_bins=energy_bins,
                                                 xml_file=catalogs_directory + "/agns.xml",
                                                 give_ids=True)
        pulsar_coordinates, _, pulsar_ids = xml_parser(energy_bins=energy_bins,
                                                       xml_file=catalogs_directory + "/pulsars.xml",
                                                       give_ids=True)

        binned_agn_map = []
        binned_pulsar_map = []
        binned_background_map = []

        for b in range(num_bins):
            # Read in binned all-sky count maps for AGN, pulsars, and background
            agns = hp.fitsfunc.read_map(filename=skymaps_directory + "/agns_{}.fits".format(b), field=None)
            pulsars = hp.fitsfunc.read_map(filename=skymaps_directory + "/pulsars_{}.fits".format(b), field=None)
            background = hp.fitsfunc.read_map(filename=skymaps_directory + "/background_{}.fits".format(b),
                                              field=None)

            binned_agn_map.append(agns / pix_sr)
            binned_pulsar_map.append(pulsars / pix_sr)
            binned_background_map.append(background / pix_sr)

        # Convert to numpy array
        binned_agn_map = np.array(binned_agn_map)
        binned_pulsar_map = np.array(binned_pulsar_map)
        binned_background_map = np.array(binned_background_map)

        # Loop to generate each patch

        for p in range(max_patches_per_catalog):

            start = time.time()

            # Check if patch already exists

            # Unique identifier of patch being generated
            patch_id = catalog_id * num_maps_per_catalog * max_patches_per_catalog + m * max_patches_per_catalog + p

            print("PATCH: {}".format(patch_id))

            # Create directory where patches stored
            patch_directory = save_directory + "/patches_2/patch_{}/".format(patch_id)

            # CENTRE OF PATCH p

            # transformation from 0-360 to -180-180 - healpy uses this coordinate system
            lon = (longitude[p] + 180) % 360 - 180

            lat = latitude[p]

            patch_centre = np.array([lon, lat])

            # GET NO. AGN AND PULSARS WITHIN PATCH, AS WELL AS THE CARTESIAN COORDINATES OF AGN AND PULSARS IN THE
            # PATCH

            # AGN and Pulsar coordinates must be in lon-lat format (l, b) with degree values

            # Get number of AGN and pulsar in given patch and their positions (in Cartesian coordinates and Galactic
            # coordinates_
            (nagn, npsr, agn_pos_list, psr_pos_list, agn_patch_ids,
             pulsar_patch_ids) = get_ps_info_128(patch_centre, agn_coordinates, pulsar_coordinates, agn_ids,
                                                 pulsar_ids)

            # If patch has already been created
            if (os.path.isfile(patch_directory + "patch.npy") and os.path.isfile(patch_directory + "mask.npy") and
                    os.path.isfile(patch_directory + "metadata.csv")):

                print("Patch already created")

                # Metadata for all patches
                patch_information.append(f"{patch_id},{catalog_id},{lat},{lon},{nagn},{npsr}\n")

            else:

                Path(patch_directory).mkdir(parents=True, exist_ok=True)

                # PROJECT ROI OF COUNT MAP INTO CARTESIAN

                binned_agn_patch = []
                binned_pulsar_patch = []
                binned_background_patch = []

                # binned_agn_patch_2 = []
                # binned_pulsar_patch_2 = []
                # binned_background_patch_2 = []

                plt.cla()
                plt.clf()
                plt.close("all")

                for b in range(num_bins):
                    agn_patch_bin = cartesian_patch(count_map=binned_agn_map[b], lon=lon, lat=lat,
                                                    xsize=xsize_patch_generation, lonra=lb_range, latra=lb_range)

                    # agn_patch_2 = copy.deepcopy(agn_patch_bin)
                    #
                    # binned_agn_patch.append(agn_patch_bin * solid_area_ratio)
                    #
                    # binned_agn_patch_2.append(agn_patch_2)

                    binned_agn_patch.append(agn_patch_bin)

                    plt.cla()
                    plt.clf()
                    plt.close("all")

                    pulsar_patch_bin = cartesian_patch(count_map=binned_pulsar_map[b], lon=lon, lat=lat,
                                                       xsize=xsize_patch_generation, lonra=lb_range, latra=lb_range)

                    # pulsar_patch_2 = copy.deepcopy(pulsar_patch_bin)
                    #
                    # binned_pulsar_patch_2.append(pulsar_patch_2)

                    # binned_pulsar_patch.append(pulsar_patch_bin * solid_area_ratio)

                    binned_pulsar_patch.append(pulsar_patch_bin)

                    plt.cla()
                    plt.clf()
                    plt.close("all")

                    background_patch_bin = cartesian_patch(count_map=binned_background_map[b], lon=lon, lat=lat,
                                                           xsize=xsize_patch_generation, lonra=lb_range, latra=lb_range)

                    # background_patch_2 = copy.deepcopy(background_patch_bin)
                    #
                    # binned_background_patch_2.append(background_patch_2)

                    # binned_background_patch.append(background_patch_bin * solid_area_ratio)

                    binned_background_patch.append(background_patch_bin)

                    # Need these here (even though we are not showing the plots - this is because visufunc creates a
                    # plot - we only need the 2D array)
                    plt.cla()
                    plt.clf()
                    plt.close("all")

                # Sum together to create patch
                # patch = np.array(binned_agn_patch) + np.array(binned_pulsar_patch) + np.array(binned_background_patch)

                patch = (solid_area_ratio * (np.array(binned_agn_patch) + np.array(binned_pulsar_patch) +
                                            np.array(binned_background_patch))).filled(0)

                # print(np.all(np.isclose(patch, patch_2)))


                np.save(patch_directory + "/patch.npy", patch)

                # CREATE MASKS

                grid2D_psf, ys, xs = create_mask(agn_pos_list, psr_pos_list, xsize_location)

                # Save masks
                np.save(patch_directory + "/mask.npy", grid2D_psf)

                # Plot the first 100 patches - for verification and inclusion in report
                if patch_id < 100:
                    plot_patch(binned_patches=patch, mask=grid2D_psf, unformatted_energy_bins=energy_bins,
                               directory=patch_directory)

                # SAVE METADATA CSV FILE FOR THIS PATCH AND THE MASK

                # Create file for storing individual patches' metadata
                source_info = (["source_id,source_type,cartesian_y,cartesian_x\n"] +
                               ([f"{agn_patch_ids[i]},AGN,{ys[i]},{xs[i]}\n" for i in range(nagn)] +
                                [f"{pulsar_patch_ids[i]},PSR,{ys[i + nagn]},{xs[i + nagn]}\n" for i in range(npsr)]))

                # Individual metadata for this patch
                f2 = open(os.path.join(patch_directory, "metadata.csv"), "w+")
                f2.writelines(source_info)
                f2.close()

                # Metadata for all patches
                patch_information.append(f"{patch_id},{catalog_id},{lat},{lon},{nagn},{npsr}\n")

            print("TIME TO CREATE PATCH: {}".format(time.time() - start))

    # SAVE PATCH METADATA

    # N.B. important to have no header - ease of combining files later.
    f1 = open(save_directory + "/patches_2/catalog_{}_patch_metadata_2.csv".format(catalog_id), "a")
    f1.writelines(patch_information)
    f1.close()
