"""
Main function for creating patches from skymaps. Placed in separate file for ease of parallelization.
"""

from formatting.mask_creation import psf_bck_mask
from formatting.projection_tools import get_ps_info_128
import healpy as hp
from map_generation.utils import cartesian_patch
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from read_write_functions import xml_parser
from verification.visualisation import plot_patch


def create_patches_for_catalog(params: list):
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

        skymaps_directory = save_directory + "/count_maps/skymap_{}".format((catalog_id * num_maps_per_catalog) + m + 1)

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

            # Check if patch already exists

            # Unique identifier of patch being generated
            patch_id = catalog_id * num_maps_per_catalog * max_patches_per_catalog + m * max_patches_per_catalog + p

            print("PATCH: {}".format(patch_id))

            # Create directory where patches stored
            patch_directory = save_directory + "/patches/patch_{}/".format(patch_id)

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

                # Create file for storing individual patches' metadata
                # list for the csv files - stores information about each patch
                individual_patch_header_line = "source_id,source_type,cartesian_y,cartesian_x\n"

                f2 = open(os.path.join(patch_directory, "metadata.csv"), "w+")
                f2.writelines(individual_patch_header_line)
                f2.close()

                # PROJECT ROI OF COUNT MAP INTO CARTESIAN

                binned_agn_patch = []
                binned_pulsar_patch = []
                binned_background_patch = []

                plt.cla()
                plt.clf()
                plt.close("all")

                for b in range(num_bins):
                    agn_patch_bin = cartesian_patch(count_map=binned_agn_map[b], lon=lon, lat=lat,
                                                    xsize=xsize_patch_generation, lonra=lb_range, latra=lb_range)

                    binned_agn_patch.append(np.array(agn_patch_bin) * solid_area_ratio)

                    plt.cla()
                    plt.clf()
                    plt.close("all")

                    pulsar_patch_bin = cartesian_patch(count_map=binned_pulsar_map[b], lon=lon, lat=lat,
                                                       xsize=xsize_patch_generation, lonra=lb_range, latra=lb_range)

                    binned_pulsar_patch.append(np.array(pulsar_patch_bin) * solid_area_ratio)
                    plt.cla()
                    plt.clf()
                    plt.close("all")

                    background_patch_bin = cartesian_patch(count_map=binned_background_map[b], lon=lon, lat=lat,
                                                           xsize=xsize_patch_generation, lonra=lb_range, latra=lb_range)

                    binned_background_patch.append(np.array(background_patch_bin) * solid_area_ratio)

                    # Need these here (even though we are not showing the plots - this is because visufunc creates a
                    # plot - we only need the 2D array)
                    plt.cla()
                    plt.clf()
                    plt.close("all")

                # Convert to numpy
                binned_agn_patch = np.array(binned_agn_patch)
                binned_pulsar_patch = np.array(binned_pulsar_patch)
                binned_background_patch = np.array(binned_background_patch)

                # Sum together to create patch
                patch = binned_agn_patch + binned_pulsar_patch + binned_background_patch

                np.save(patch_directory + "/patch.npy", patch)

                # CREATE MASKS

                # Create blank masks which can be added to
                grid2D_psf = np.zeros((xsize_patch_generation, xsize_patch_generation))

                source_info = []

                # Add AGN masks iteratively
                for i in range(nagn):
                    # Find the minimum of the position in the image and 127 (the maximum index of the image in the y or
                    # x-axis)
                    y = min(agn_pos_list[i][0], xsize_location - 1)
                    x = min(agn_pos_list[i][1], xsize_location - 1)

                    grid2D_psf = psf_bck_mask(y // 2, x // 2, radius=2.5, psf_mask=grid2D_psf)

                    source_info.append(f"{agn_patch_ids[i]},AGN,{y},{x}\n")

                for i in range(npsr):
                    y = min(psr_pos_list[i][0], xsize_location - 1)
                    x = min(psr_pos_list[i][1], xsize_location - 1)

                    grid2D_psf = psf_bck_mask(y // 2, x // 2, radius=2.5, psf_mask=grid2D_psf)

                    source_info.append(f"{pulsar_patch_ids[i]},PSR,{y},{x}\n")

                # Plot all patches generated from the first skymap
                if catalog_id == 0 and m == 0:
                    plot_patch(binned_patches=patch, mask=grid2D_psf, unformatted_energy_bins=energy_bins,
                               directory=patch_directory)

                # Save masks
                np.save(patch_directory + "/mask.npy", grid2D_psf)

                # SAVE METADATA CSV FILE FOR THIS PATCH AND THE MASK

                # Individual metadata for this patch
                f2 = open(os.path.join(patch_directory, "metadata.csv"), "a")
                f2.writelines(source_info)
                f2.close()

                # Metadata for all patches
                patch_information.append(f"{patch_id},{catalog_id},{lat},{lon},{nagn},{npsr}\n")

    # SAVE PATCH METADATA

    # N.B. important to have no header - ease of combining files later.
    f1 = open(save_directory + "/patches/catalog_{}_patch_metadata.csv".format(catalog_id), "a")
    f1.writelines(patch_information)
    f1.close()
