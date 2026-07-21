# This code is based on the code from ID8 - I have altered/optimised and tailored it to my project specifications.
# The link to this code can be found here: https://github.com/bapanes/AutoSourceID/blob/main/codes/from-cats-to-locnet-
# input.py


import argparse
from formatting.projection_tools import get_ps_info_128
from formatting.mask_creation import psf_bck_mask
import healpy as hp
from map_generation.visualisation import format_scientific_notation_label
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from read_write_functions import xml_parser


def plot_patch(binned_patches, mask, unformatted_energy_bins, directory):

    formatted_energy_bins = format_scientific_notation_label(unformatted_energy_bins)

    plt.rcParams["figure.figsize"] = (15, 10)

    fig, ax = plt.subplots(nrows=2, ncols=3)

    axes = ax.flatten()

    for a in range(len(axes) - 1):

        im = axes[a].imshow(binned_patches[a])

        axes[a].set_title("Count Map {} MeV - {} MeV".format(formatted_energy_bins[a], formatted_energy_bins[a + 1]))

        fig.colorbar(im, ax=axes[a])

    axes[-1].imshow(mask)

    fig.suptitle("Binned Patch Count Map")
    fig.tight_layout()

    plt.savefig(directory + "/patch_visualised.png")


if __name__ == "__main__":

    # INPUT PARAMETERS AND DATA

    parser = argparse.ArgumentParser(description="Read in simulated all-sky maps and catalogs to create a dataset of "
                                                 "ROIs to be fed to source dectection algorithms.")

    parser.add_argument("--num_catalogs", required=True, type=int, help="The number of simulated catalogs "
                                                                        "and skymaps that have been generated and can be"
                                                                        "split into patches.")

    parser.add_argument("--num_energy_bins", required=True, type=int, help="The number of energy bins that "
                                                                           "photons were binned into using fermitools.")

    args = parser.parse_args()

    num_catalogs = args.num_catalogs
    num_bins = args.num_energy_bins

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
    # noinspection PyUnresolvedReferences - HERE I PICKED coord = G to be safe
    proj = hp.projector.CartesianProj(lonra=lb_range, latra=lb_range, xsize=xsize_patch_generation)

    I, J = np.meshgrid(np.arange(xsize_patch_generation), np.arange(xsize_patch_generation))
    x, y = proj.ij2xy(I, J)

    # Used to correct the rotation smear - I THINK
    dl = np.radians((lb_range[1] - lb_range[0]) / (xsize_patch_generation - 1))
    solid_area_ratio = dl * dl * np.cos(np.radians(y))

    # Gulli's approach to generate a more uniform coverage of the sky - specifies the longitudes at which to draw out
    # the slice of the sky to take

    # ChANGED THIS TO 7

    longitude, latitude = hp.pix2ang(8, np.arange(hp.nside2npix(8)), lonlat=True)

    # longitude, latitude = hp.pix2ang(7, np.arange(hp.nside2npix(7)), lonlat=True)

    # The number of patches to generate per catalog
    max_patches_per_catalog = len(longitude)

    # CREATE CSV FILE FOR STORING PATCH INFORMATION

    # list for the csv files - stores information about each patch
    header_line = "patch_id,centre_lat,centre_lon,num_agn,num_psr\n"

    f1 = open(os.path.join("./simulated_data/patches/", "patch_metadata.csv"), "w+")
    f1.writelines(header_line)
    f1.close()

    patch_information = []

    for c in range(num_catalogs):

        # READ IN MAPS AND SPATIAL/SPECTRAL PARAMETERS

        # Num patches to generate
        patches = max_patches_per_catalog

        skymaps_directory = "./simulated_data/count_maps/skymap_{}".format(c + 1)
        catalogs_directory = "./simulated_data/catalogs/catalog_{}".format(c + 1)

        # Read in the coordinates and associated integral photon fluxes of each source in AGN and pulsar maps
        agn_coordinates, agn_photon_fluxes, agn_ids = xml_parser(energy_bins=energy_bins,
                                                                 xml_file=catalogs_directory + "/agns.xml",
                                                                 give_ids=True)
        pulsar_coordinates, pulsar_photon_fluxes, pulsar_ids = xml_parser(energy_bins=energy_bins,
                                                                          xml_file=catalogs_directory + "/pulsars.xml",
                                                                          give_ids=True)

        binned_agn_map = []
        binned_pulsar_map = []
        binned_background_map = []

        for b in range(num_bins):

            # Read in binned all-sky count maps for AGN, pulsars, and background

            agns = hp.fitsfunc.read_map(filename=skymaps_directory + "/agns_{}.fits".format(b), field=None)
            pulsars = hp.fitsfunc.read_map(filename=skymaps_directory + "/pulsars_{}.fits".format(b), field=None)
            background = hp.fitsfunc.read_map(filename=skymaps_directory + "/background_{}.fits".format(b), field=None)

            binned_agn_map.append(agns / pix_sr)
            binned_pulsar_map.append(pulsars / pix_sr)
            binned_background_map.append(background / pix_sr)

        # Convert to numpy array
        binned_agn_map = np.array(binned_agn_map)
        binned_pulsar_map = np.array(binned_pulsar_map)
        binned_background_map = np.array(binned_background_map)

        # Loop to generate each patch

        for p in range(patches):

            print("PATCH: {}".format(p + 1))

            # Unique identifier of patch being generated
            patch_id = c * max_patches_per_catalog + p

            # CENTRE OF PATCH p

            # transformation from 0-360 to -180-180 - healpy uses this coordinate system
            lon = (longitude[p] + 180) % 360 - 180

            lat = latitude[p]

            patch_centre = np.array([lon, lat])

            # PROJECT ROI OF COUNT MAP INTO CARTESIAN
            # N.B. in previous code, they performed a solid angle ratio correction here (assume not needed, as already
            # accounted for this in a previous step? - CHECK UNDERSTANDING)

            binned_agn_patch = []
            binned_pulsar_patch = []
            binned_background_patch = []

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
                                                        xsize=xsize_patch_generation, lonra=lb_range, latra=lb_range,
                                                        return_projected_map=True)

                binned_pulsar_patch.append(np.array(pulsar_patch_bin) * solid_area_ratio)
                plt.cla()
                plt.clf()
                plt.close("all")

                background_patch_bin = hp.visufunc.cartview(binned_background_map[b], rot=(lon, lat, 0.), coord='G',
                                                            xsize=xsize_patch_generation, lonra=lb_range,
                                                            latra=lb_range, return_projected_map=True)


                binned_background_patch.append(np.array(background_patch_bin) * solid_area_ratio)

                # Need these here (even though we are not showing the plots - this is because visufunc creates a plot -
                # we only need the 2D array)
                plt.cla()
                plt.clf()
                plt.close("all")

            # Convert to numpy
            binned_agn_patch = np.array(binned_agn_patch)
            binned_pulsar_patch = np.array(binned_pulsar_patch)
            binned_background_patch = np.array(binned_background_patch)

            # plot_patch(binned_patches=binned_agn_patch, mask=np.zeros((64, 64)), unformatted_energy_bins=energy_bins, directory='./')

            # Sum together to create patch
            patch = binned_agn_patch + binned_pulsar_patch + binned_background_patch

            # Create directory where patches stored
            patch_directory = "./simulated_data/patches/patch_{}/".format(patch_id)

            Path(patch_directory).mkdir(parents=True, exist_ok=True)

            # Create file for storing individual patches' metadata
            # list for the csv files - stores information about each patch
            individual_patch_header_line = "source_id,source_type,cartesian_y,cartesian_x\n"

            f2 = open(os.path.join(patch_directory, "metadata.csv"), "w+")
            f2.writelines(individual_patch_header_line)
            f2.close()

            np.save(patch_directory + "/patch.npy", patch)

            # GET NO. AGN AND PULSARS WITHIN PATCH, AS WELL AS THE CARTESIAN COORDINATES OF AGN AND PULSARS IN THE PATCH

            # AGN and Pulsar coordinates must be in lon-lat format (l, b) with degree values

            # Get number of AGN and pulsar in given patch and their positions (in Cartesian coordinates and Galactic
            # coordinates_
            (nagn, npsr, agn_pos_list, psr_pos_list, agn_patch_ids,
             pulsar_patch_ids) = get_ps_info_128(patch_centre, agn_coordinates, pulsar_coordinates, agn_ids, pulsar_ids)

            # CREATE MASKS

            # Create blank masks which can be added to
            grid2D_psf = np.zeros((xsize_patch_generation, xsize_patch_generation))
            grid2D_bck = np.ones((xsize_patch_generation, xsize_patch_generation))

            source_info = []

            # Add AGN masks iteratively
            for i in range(nagn):
                # Find the minimum of the position in the image and 127 (the maximum index of the image in the y or x-
                # axis)
                y = min(agn_pos_list[i][0], xsize_location - 1)
                x = min(agn_pos_list[i][1], xsize_location - 1)

                ltrue = agn_pos_list[i][2]
                btrue = agn_pos_list[i][3]

                # Further points of disc (in terms of indices) from location
                r = 5
                xmin, xmax, ymin, ymax = max(0, x - r), min(xsize_location - 1, x + r), max(0, y - r), min(
                    xsize_location - 1, y + r)

                # Divide by 2 to get the coordinates on the 64 x 64 image
                xmin //= 2
                xmax //= 2
                ymin //= 2
                ymax //= 2

                grid2D_psf = psf_bck_mask(y // 2, x // 2, radius=2.5, psf_mask=grid2D_psf)

                source_info.append(f"{agn_patch_ids[i]},AGN,{y},{x}\n")

            for i in range(npsr):

                y = min(psr_pos_list[i][0], xsize_location - 1)
                x = min(psr_pos_list[i][1], xsize_location - 1)

                ltrue = psr_pos_list[i][2]
                btrue = psr_pos_list[i][3]

                r = 5
                xmin, xmax, ymin, ymax = max(0, x - r), min(xsize_location - 1, x + r), max(0, y - r), min(
                    xsize_location - 1, y + r)

                xmin //= 2
                xmax //= 2
                ymin //= 2
                ymax //= 2

                grid2D_psf = psf_bck_mask(y // 2, x // 2, radius=2.5, psf_mask=grid2D_psf)

                source_info.append(f"{pulsar_patch_ids[i]},PSR,{y},{x}\n")

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
            patch_information.append(f"{patch_id},{lat},{lon},{nagn},{npsr}\n")

    # SAVE PATCH METADATA

    f1 = open("./simulated_data/patches/patch_metadata.csv", "a")
    f1.writelines(patch_information)
    f1.close()
