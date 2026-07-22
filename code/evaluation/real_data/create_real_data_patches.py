# This code is based on the code from ID8 - I have altered/optimised, tailored, and rewritten it for my project
# specifications (in this case to format real Fermi data into patches, as well as improve computation time, etc.).
# The link to this code can be found here: https://github.com/bapanes/AutoSourceID/blob/main/codes/from-cats-to-locnet-
# input.py

import argparse
import healpy.fitsfunc
from astropy.table import QTable
import healpy as hp
from ../../data_simulation/map_generation.visualisation import format_scientific_notation_label
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

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

    parser = argparse.ArgumentParser(description="Read in real all-sky maps to create a dataset of "
                                                 "ROIs to be fed to source detection algorithms.")

    parser.add_argument("--num_energy_bins", required=True, type=int, help="The number of energy bins that "
                                                                           "photons were binned into using fermitools.")

    parser.add_argument("--patch_directory", required=True, type=str, help="Directory in which to save "
                                                                           "generated patches")

    parser.add_argument("--fermi_catalog_location", required=True, type=str, help="File location of 4FGL catalog on"
                                                                                  "user's computer.")

    parser.add_argument("--binned_count_maps", required=True, type=str, help="File location of binned count maps"
                                                                             "of real Fermi LAT data.")

    args = parser.parse_args()

    num_bins = args.num_energy_bins
    save_directory = args.patch_directory
    catalog_4fgl = args.fermi_catalog_location
    binned_count_maps_file = args.binned_count_maps

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

    # Used to correct the rotation smear - I THINK
    dl = np.radians((lb_range[1] - lb_range[0]) / (xsize_patch_generation - 1))
    solid_area_ratio = dl * dl * np.cos(np.radians(y))

    # Gulli's approach to generate a more uniform coverage of the sky - specifies the longitudes at which to draw out
    # the slice of the sky to take

    longitude, latitude = hp.pix2ang(8, np.arange(hp.nside2npix(8)), lonlat=True)

    # The number of patches to generate per catalog
    max_patches_per_catalog = len(longitude)

    # CREATE CSV FILE FOR STORING PATCH INFORMATION

    # list for the csv files - stores information about each patch
    header_line = "patch_id,centre_lat,centre_lon,num_agn,num_psr\n"

    Path(save_directory + "/patches").mkdir(parents=True, exist_ok=True)

    f1 = open(save_directory + "/patches/patch_metadata.csv", "w+")
    f1.writelines(header_line)
    f1.close()

    patch_information = []

    # READ IN MAPS AND SPATIAL/SPECTRAL PARAMETERS

    # This is where we differ from the create_dataset.py function in data_simulation - we get information directly from
    # the 4FGL

    catalog = QTable.read(catalog_4fgl, format='fits', hdu=1)

    columns = ("Source_Name", "CLASS1", "GLAT", "GLON")

    catalog = catalog[columns]

    # Reformat CLASS1 column - remove empty spaces and make all lower case
    catalog["CLASS1"] = np.asarray([k.decode('utf-8').strip().lower() for k in catalog["CLASS1"].value.filled('-')])

    # Reformat CLASS1 column - remove empty spaces and make all lower case
    catalog["Source_Name"] = np.asarray([k.decode('utf-8').strip().lower()[5:] for k in catalog["Source_Name"].value])

    # Select all rows that describe pulsars
    pulsar_mask = (catalog["CLASS1"] == "psr")

    # Select all rows that describe AGN
    agn_mask = np.isin(catalog["CLASS1"].data, np.array(["bcu", "sey", "ssrq", "bll", "fsrq", "rdg", "nlsy1", "agn"]))

    agn_data = catalog[agn_mask].copy()
    pulsar_data = catalog[pulsar_mask].copy()

    # Convert to numpy arrays and select coordinates and IDS

    agn_coordinates = np.array([agn_data["GLON"].value, agn_data["GLAT"].value]).T
    pulsar_coordinates = np.array([pulsar_data["GLON"].value, pulsar_data["GLAT"].value]).T

    agn_ids = agn_data["Source_Name"].value

    pulsar_ids = pulsar_data["Source_Name"].value

    # READ IN BINNED ALL-SKY COUNT MAPS

    # N.B. We make pixel/solid angle correction here

    binned_count_map = [healpy.fitsfunc.read_map(binned_count_maps_file, field=b, hdu="SKYMAP") / pix_sr for b in range(num_bins)]

    for p in range(max_patches_per_catalog):

        print("PATCH: {}".format(p + 1))

        patch_id = p

        # transformation from 0-360 to -180-180 - healpy uses this coordinate system
        lon = (longitude[p] + 180) % 360 - 180

        lat = latitude[p]

        patch_centre = np.array([lon, lat])

        # PROJECT ROI OF COUNT MAP INTO CARTESIAN
        # N.B. in previous code, they performed a solid angle ratio correction here (assume not needed, as already
        # accounted for this in a previous step? - CHECK UNDERSTANDING)

        plt.cla()
        plt.clf()
        plt.close("all")

        binned_patch = []

        for b in range(num_bins):
            patch_bin = hp.visufunc.cartview(binned_count_map[b], rot=(lon, lat, 0.), coord='G',
                                             xsize=xsize_patch_generation, lonra=lb_range, latra=lb_range,
                                             return_projected_map=True)

            binned_patch.append(np.array(patch_bin) * solid_area_ratio)

            # Need these here (even though we are not showing the plots - this is because visufunc creates a plot -
            # we only need the 2D array)
            plt.cla()
            plt.clf()
            plt.close("all")

        # Convert to numpy
        binned_patch = np.array(binned_patch)

        # SAVE PATCH, MASK, AND METADATA

        patch_directory = save_directory + "/patches/patch_{}/".format(patch_id)

        Path(patch_directory).mkdir(parents=True, exist_ok=True)

        # Create file for storing individual patches' metadata
        # list for the csv files - stores information about each patch
        individual_patch_header_line = "source_id,source_type,cartesian_y,cartesian_x\n"

        f2 = open(os.path.join(patch_directory, "metadata.csv"), "w+")
        f2.writelines(individual_patch_header_line)
        f2.close()

        np.save(patch_directory + "/patch.npy", binned_patch)

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

    f1 = open(save_directory + "/patches/patch_metadata.csv", "a")
    f1.writelines(patch_information)
    f1.close()

