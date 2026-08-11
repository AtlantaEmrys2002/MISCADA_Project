"""
Main function for creating MASKS from skymaps - this is due to energy flux being TOO LOW at times for sources to be
detected. Placed in separate file for ease of parallelization.
"""

from formatting.mask_creation import psf_bck_mask
from formatting.projection_tools import get_ps_info_128
import numpy as np
import os
from read_write_functions import xml_parser


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
    (save_directory, num_maps_per_catalog, catalog_id, energy_bins, max_patches_per_catalog, longitude,
     latitude, xsize_patch_generation, xsize_location) = params

    catalogs_directory = save_directory + "/catalogs/catalog_{}".format(catalog_id + 1)

    patch_information = []

    for m in range(num_maps_per_catalog):

        # READ IN MAPS AND SPATIAL/SPECTRAL PARAMETERS

        # Read in the coordinates and associated integral photon fluxes of each source in AGN and pulsar maps

        agn_coordinates, _, agn_ids = xml_parser(energy_bins=energy_bins,
                                                 xml_file=catalogs_directory + "/agns.xml",
                                                 give_ids=True, energy_flux_limited=True)
        pulsar_coordinates, _, pulsar_ids = xml_parser(energy_bins=energy_bins,
                                                       xml_file=catalogs_directory + "/pulsars.xml",
                                                       give_ids=True, energy_flux_limited=True)

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

            # Create file for storing individual patches' metadata
            # list for the csv files - stores information about each patch
            individual_patch_header_line = "source_id,source_type,cartesian_y,cartesian_x\n"

            f2 = open(os.path.join(patch_directory, "metadata_2.csv"), "w+")
            f2.writelines(individual_patch_header_line)
            f2.close()

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

            # Save masks
            np.save(patch_directory + "/mask_2.npy", grid2D_psf)

            # SAVE METADATA CSV FILE FOR THIS PATCH AND THE MASK

            # Individual metadata for this patch
            f2 = open(os.path.join(patch_directory, "metadata_2.csv"), "a")
            f2.writelines(source_info)
            f2.close()

            # Metadata for all patches
            patch_information.append(f"{patch_id},{catalog_id},{lat},{lon},{nagn},{npsr}\n")

    # SAVE PATCH METADATA

    # N.B. important to have no header - ease of combining files later.
    f1 = open(save_directory + "/patches/catalog_{}_patch_metadata_2.csv".format(catalog_id), "a")
    f1.writelines(patch_information)
    f1.close()
