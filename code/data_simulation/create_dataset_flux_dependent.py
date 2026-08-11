"""
This code performs the same functionality as that of ID8's. However, I have rewritten my own version and optimised,
tailored, added to it to fit my specifications and improve its performance. ID8's code can be found here:
https://github.com/bapanes/AutoSourceID/blob/main/codes/from-cats-to-locnet-input.py.

UNLIKE create_dataset.py, THIS VERSION INCORPORATES ENERGY FLUXES OF SOURCES (EXCLUDING SOURCES THAT ARE TOO FAINT).
"""

import argparse
import copy
from create_patches_flux_dependent import create_patches_for_catalog_flux_dependent
import healpy as hp
from multiprocessing import Pool
import numpy as np
import os
import time

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

    # Actual dimension of square patch
    xsize_patch_generation = 64

    # Gulli's approach to generate a more uniform coverage of the sky - specifies the longitudes at which to draw out
    # the slice of the sky to take
    longitude, latitude = hp.pix2ang(8, np.arange(hp.nside2npix(8)), lonlat=True)

    # The number of patches to generate per catalog
    max_patches_per_catalog = len(longitude)

    # CREATE CSV FILE FOR STORING PATCH INFORMATION

    # list for the csv files - stores information about each patch
    header_line = "patch_id,catalog_id,centre_lat,centre_lon,num_agn,num_psr\n"

    f1 = open(save_directory + "/patches/patch_metadata_2.csv", "w+")
    f1.writelines(header_line)
    f1.close()

    patch_information = []

    # Num patches to generate
    patches = max_patches_per_catalog

    arguments = [[save_directory, num_maps_per_catalog, c, copy.deepcopy(energy_bins), max_patches_per_catalog,
                  copy.deepcopy(longitude), copy.deepcopy(latitude), xsize_patch_generation, xsize_location] for c in
                 range(num_catalogs)]

    # Parallel computation of maps - split catalogs between CPUs
    start_count_time = time.time()

    pool = Pool(processes=os.cpu_count() // 2)

    pool.map(create_patches_for_catalog_flux_dependent, arguments)

    pool.terminate()

    print("Time to create new masks: {} s".format(time.time() - start_count_time))
    print("Average time to create masks per sky map {} s".format((time.time() - start_count_time) /
                                                                 (num_catalogs * num_maps_per_catalog)))

    # SAVE PATCH METADATA

    new_lines = []

    for c in range(num_catalogs):
        with open(save_directory + "/patches/catalog_{}_patch_metadata_2.csv".format(c)) as f3:
            lines = f3.readlines()

            new_lines += lines

    f1 = open(save_directory + "/patches/patch_metadata_2.csv", "a")
    f1.writelines(new_lines)
    f1.close()

# REFERENCES

# Indexing 3D Arrays stored as 1D - https://cplusplus.com/forum/general/137677/
# Readlines - https://stackoverflow.com/questions/16222956/reading-a-file-line-by-line-into-elements-of-an-array-in-
# python
