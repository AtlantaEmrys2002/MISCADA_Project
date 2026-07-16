# ALL FUNCTIONS EXCEPT FOR main() WHERE TAKEN DIRECTLY FROM ID8'S GITHUB CODE
# The link to this code can be found here: https://github.com/bapanes/AutoSourceID/blob/main/codes/from-cats-to-locnet-
# input.py


import argparse
from formatting.projection_tools import get_ps_info_128
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from read_write_functions import xml_parser


if __name__ == "__main__":

    # INPUT PARAMETERS AND DATA

    parser = argparse.ArgumentParser(description="Read in simulated all-sky maps and catalogs to create a dataset of "
                                                 "ROIs to be fed to source dectection algorithms.")

    parser.add_argument("--num_catalogs", required=True, type=int, help="The number of simulated catalogs "
                                                                        "and skymaps that have been generated and can be"
                                                                        "split into patches.")

    parser.add_argument("--num_energy_bins", required=True, type=int, help="The number of energy bins that "
                                                                           "photons were binned into using fermitools.")

    parser.add_argument("--num_patches_per_catalog", required=False, type=int, help="The number of patches"
                                                                                    "to generate per sky-map provided.")

    args = parser.parse_args()

    num_catalogs = args.num_catalogs
    num_bins = args.num_energy_bins
    # n = args.num_patches_per_catalog

    # FILE CREATION AND ORGANISATION

    # list for the csv files - stores information about each patch (including true source locations)
    header_line = "filename,xmin,xmax,ymin,ymax,class,lon_c,lat_c,flux_1000,lon_p,lat_p,catalog,flux_10000\n"

    f1 = open(os.path.join("./simulated_data/patches/", "metadata.csv"), "w+")
    f1.writelines(header_line)
    f1.close()

    energy_bins = np.logspace(np.log(300), np.log(200000), num=num_bins + 1, base=np.e)

    xsize_location = 128

    ##################################################
    # global 64x64 correction
    ##################################################
    xsize_patch_generation = 64
    # NSIDE = 256

    # Npix = 12 * NSIDE * NSIDE
    # pix_sr = 4.0 * np.pi / Npix

    lb_range = [-5, 5]
    proj = hp.projector.CartesianProj(lonra=lb_range, latra=lb_range, xsize=xsize_patch_generation)

    I, J = np.meshgrid(np.arange(xsize_patch_generation), np.arange(xsize_patch_generation))
    x, y = proj.ij2xy(I, J)

    # Used to correct the rotation smear
    dl = np.radians((lb_range[1] - lb_range[0])/ (xsize_patch_generation - 1))
    solid_area_ratio = dl * dl * np.cos(np.radians(y))

    # Gulli's approach to generate a more uniform coverage of the sky - specifies the longitudes at which to draw out
    # the slice of the sky to take
    longitude, latitude = hp.pix2ang(8, np.arange(hp.nside2npix(8)), lonlat=True)

    # The number of patches to generate per catalog
    max_patches_per_catalog = len(longitude)

    for c in range(num_catalogs):

        # READ IN MAPS AND SPATIAL/SPECTRAL PARAMETERS

        # Num patches to generate
        patches = max_patches_per_catalog

        skymaps_directory = "./simulated_data/count_maps/skymap_{}".format(c + 1)
        catalogs_directory = "./simulated_data/catalogs/catalog_{}".format(c + 1)

        # Read in the coordinates and associated integral photon fluxes of each source in AGN and pulsar maps
        agn_coordinates, agn_photon_fluxes = xml_parser(energy_bins=energy_bins,
                                                        xml_file=catalogs_directory + "/agns.xml")
        pulsar_coordinates, pulsar_photon_fluxes = xml_parser(energy_bins=energy_bins,
                                                              xml_file=catalogs_directory + "/pulsars.xml")

        binned_agn_map = []
        binned_pulsar_map = []
        binned_background_map = []

        for b in range(num_bins):

            # Read in binned all-sky count maps for AGN, pulsars, and background

            agns = hp.fitsfunc.read_map(filename=skymaps_directory + "/agns_{}.fits".format(b))
            pulsars = hp.fitsfunc.read_map(filename=skymaps_directory + "/pulsars_{}.fits".format(b))
            background = hp.fitsfunc.read_map(filename=skymaps_directory + "/background_{}.fits".format(b))

            binned_agn_map.append(agns)
            binned_pulsar_map.append(pulsars)
            binned_background_map.append(background)

        # Convert to numpy array
        binned_agn_map = np.array(binned_agn_map)
        binned_pulsar_map = np.array(binned_pulsar_map)
        binned_background_map = np.array(binned_background_map)

        # Loop to generate each patch

        for p in range(patches):

            # CENTRE OF PATCH p

            # transformation from 0-360 to -180-180 - healpy uses this coordinate system
            lon = (longitude[p] + 180) % 360 - 180

            lat = latitude[p]

            print(lon, lat)

            patch_centre = np.array([lon, lat])

            # PROJECT ROI OF COUNT MAP INTO CARTESIAN
            # N.B. in previous code, they performed a solid angle ratio correction here (assume not needed, as already
            # accounted for this in a previous step? - CHECK UNDERSTANDING)

            binned_agn_patch = []
            binned_pulsar_patch = []
            binned_background_patch = []

            for b in range(num_bins):

                agn_patch_bin = hp.visufunc.cartview(binned_agn_map[b], rot=(lon, lat, 0.), coord='G',
                                                       xsize=xsize_patch_generation, lonra=lb_range, latra=lb_range,
                                                       return_projected_map=True)

                pulsar_patch_bin = hp.visufunc.cartview(binned_pulsar_map[b], rot=(lon, lat, 0.), coord='G',
                                                       xsize=xsize_patch_generation, lonra=lb_range, latra=lb_range,
                                                       return_projected_map=True)

                background_patch_bin = hp.visufunc.cartview(binned_background_map[b], rot=(lon, lat, 0.), coord='G',
                                                       xsize=xsize_patch_generation, lonra=lb_range, latra=lb_range,
                                                       return_projected_map=True)

                binned_agn_patch.append(agn_patch_bin)
                binned_pulsar_patch.append(pulsar_patch_bin)
                binned_background_patch.append(background_patch_bin)

                # Need these here (even though we are not showing the plots - this is because visufunc creates a plot - we
                # only need the 2D array)
                plt.cla()
                plt.clf()
                plt.close("all")

            # Convert to numpy
            binned_agn_patch = np.array(binned_agn_patch)
            binned_pulsar_patch = np.array(binned_pulsar_patch)
            binned_background_patch = np.array(binned_background_patch)

            # Sum together to create patch
            patch = binned_agn_patch + binned_pulsar_patch + binned_background_patch

            # Create directory where patches stored
            patch_directory = "./simulated_data/patches/patch_{}/".format(p)

            Path(patch_directory).mkdir(parents=True, exist_ok=True)

            np.save(patch_directory + "/patch.npy".format(c * max_patches_per_catalog + p), patch)

            # SAVE METADATA CSV FILE FOR THIS PATCH AND THE MASK







            # GET NO. AGN AND PULSARS WITHIN PATCH, AS WELL AS THE CARTESIAN COORDINATES OF AGN AND PULSARS IN THE PATCH

            # we recover the info using 128x128 patch dimensions
            # this remains from our initial approach. it does not affect at all the image and mask generation
            nagn, npsr, agn_pos_list, psr_pos_list = get_ps_info_128(patch_centre, agn_coordinates,
                                                                     pulsar_coordinates, xsize_location)





