import argparse
from benchmarks.uneb import uneb_algorithm
from benchmarks.unek import unek_algorithm
import numpy as np
import pandas as pd
from read_write_functions import read_patches, save_predictions

if __name__ == "__main__":
    # PROCESS USER INPUT

    parser = argparse.ArgumentParser(description="Reads in a collection of specified patches and formats them to be "
                                                 "passed to benchmark algorithms. These benchmarks are then applied and"
                                                 "the parameters of the 'best' versions of them are saved.")

    parser.add_argument("--patch_location", required=True, type=str, help="The directory in which the"
                                                                          "patch data is stored.")

    parser.add_argument("--num_patches", required=True, type=int, help="The number of patches to read from"
                                                                       "the specified patch directory.")

    parser.add_argument("--save_directory", required=True, type=str, help="Location in which to save the "
                                                                          "predictions and results for each source"
                                                                          "extraction method.")

    args = parser.parse_args()

    patches_directory = args.patch_location
    num_patches = args.num_patches
    save_directory = args.save_directory

    # READ IN PATCHES CORRECTLY

    print("Reading in and formatting patches...")

    # WARNING - REMEMBER THAT PREDICTED LOCATIONS ARE NOT IN THE SAME ORDER AS ACTUAL LOCATIONS - HAVE TO FIND WHICH ONES ARE CLOSEST
    # REMEMBER ACTUAL LOCATIONS ARE GIVEN AS y,x AND NOT x, y - READ tHE METADATA CAREFULLY!!!!!!! LOOK BACK AT NOTES IN read_write_functions

    train, valid, test = read_patches(num_patches=num_patches, directory=patches_directory)

    # GET INFORMATION FOR TEST PATCHES

    # FETCH PATCH INFORMATION

    # Fetch the actual locations of each source
    # file_structure = "./../data_simulation/simulated_data/patches/patch_{}/metadata.csv"
    #
    # patch_information = [pd.read_csv(file_structure.format(k)) for k in test_patch_ids]
    #
    # types = [df["source_type"].to_numpy() for df in patch_information]
    #
    # actual_source_locations = [np.array(list(zip(df["cartesian_x"].to_numpy(), df["cartesian_y"].to_numpy()))) for df
    #                            in patch_information]
    #
    # source_ids = [df["source_id"].to_numpy() for df in patch_information]

    # NEED INFINITE COUNTS PATCH SO IT GOES PATCH ID, SOURCE ID, CARTESIAN LOCATION, RA/DEC LOCATION IN SKY, TYPE, FLUX IN RANGE 300-200000
    # FLUX IS THE ONE IN PHOTON cm^-2 s^-1 (I.E. PHOTON FLUX), NO. PHOTONS FROM SOURCE VS NO. PHOTONS FROM THAT PIXEL IN THE MAP




    # WE NEED TO TRAIN ON OUTPUT WITH TRAIN DATA!!! THIS IS IMPORTANT - SAVE TEST DATA TILL THE END - NEED PATCH INFO FOR ALL - NOT JUST THE TEST PATCHES!!!!!!!
    # SO RETURN PATCH IDS OF VALIDATION AND TEST DATA TOO








    print("Complete")

    # UNEK




    # CHANGE BACK tO NOt PRETRAINED AFTER GOT CLASSIFIER WORKING



    # Results when applied to the TEST data (not the train or validation data)
    unek_predicted_segmentations, unek_predicted_locations, unek_classifier_predictions, actual_classes, test_patch_ids = unek_algorithm(train, valid, test,
                                                                                            use_pretrained_detector=
                                                                                            True)

    # Save predictions for test patches
    save_predictions(patch_ids=test_patch_ids, predicted_segmentations=unek_predicted_segmentations,
                     predicted_locations=unek_predicted_locations,
                     predicted_classes=unek_classifier_predictions, actual_classes=actual_classes, directory=save_directory, method="UNEK")

    # UNEB

    # # N.B. use pre-trained, as exactly the same data is being used as validation and test data
    # uneb_predicted_segmentations, uneb_prediction_locations, test_patch_ids = uneb_algorithm(train, valid, test,
    #                                                                                          use_pretrained_detector=
    #                                                                                          True)

    save_predictions(patch_ids=test_patch_ids, predicted_segmentations=uneb_predicted_segmentations,
                     predicted_locations=uneb_prediction_locations, directory=save_directory, method="UNEB")

# REFERENCES

# Argparse Errors - https://stackoverflow.com/questions/10900617/getting-syntax-error-near-unexpected-token-in-python
# Interpolation in Imshow - https://stackoverflow.com/questions/55121294/imshow-plot-with-no-data-values-excluded-from-interpolation
