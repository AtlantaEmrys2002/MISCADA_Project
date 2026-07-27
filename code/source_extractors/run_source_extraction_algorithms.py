import argparse
from benchmarks.uneb import uneb_algorithm
from benchmarks.unek import unek_algorithm
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

    # NEED INFINITE COUNTS PATCH SO IT GOES PATCH ID, SOURCE ID, CARTESIAN LOCATION, RA/DEC LOCATION IN SKY, TYPE, FLUX IN RANGE 300-200000
    # FLUX IS THE ONE IN PHOTON cm^-2 s^-1 (I.E. PHOTON FLUX), NO. PHOTONS FROM SOURCE VS NO. PHOTONS FROM THAT PIXEL IN THE MAP

    print("Complete")

    # UNEK




    # CHANGE BACK tO NOt PRETRAINED AFTER GOT CLASSIFIER WORKING



    # Results when applied to the TEST data (not the train or validation data)

    successful_algorithms = []
    unsuccessful_algorithms = []

    try:

        (unek_predicted_segmentations, unek_predicted_locations, unek_classifier_predictions, actual_classes,
         test_patch_ids) = unek_algorithm(train, valid, test, use_pretrained_detector=True)

    except RuntimeError:

        print("Not enough data to train classifier for " + "UNEK")
        unsuccessful_algorithms.append("UNEK")

    else:

        # Save predictions for test patches
        save_predictions(patch_ids=test_patch_ids, predicted_segmentations=unek_predicted_segmentations,
                         predicted_locations=unek_predicted_locations,
                         predicted_classes=unek_classifier_predictions, actual_classes=actual_classes,
                         directory=save_directory, method="UNEK")

        successful_algorithms.append("UNEK")

    # UNEB

    try:

        # # N.B. use pre-trained, as exactly the same data is being used as validation and test data
        (uneb_predicted_segmentations, uneb_prediction_locations, uneb_classifier_predictions, actual_class,
         test_patch_ids) = uneb_algorithm(train, valid, test, use_pretrained_detector=True)

    except RuntimeError:

        print("Not enough data to train classifier for " + "UNEB")
        unsuccessful_algorithms.append("UNEB")

    else:

        # N.B. although we use the same U-Net as UNEK, we do not use the same classifier (although the architectures are the
        # same, the output localisations (K-means vs blob detection) will vary, so need to train classifier on the
        # predictions of the localise algorithm.
        save_predictions(patch_ids=test_patch_ids, predicted_segmentations=uneb_predicted_segmentations,
                         predicted_locations=uneb_prediction_locations, predicted_classes=uneb_classifier_predictions,
                         actual_classes=actual_class, directory=save_directory, method="UNEB")

        successful_algorithms.append("UNEB")

    print("Successfully Trained Algorithms: {}".format(successful_algorithms))
    print("Unsucessfully Trained Algorithms: {}".format(unsuccessful_algorithms))

# REFERENCES

# Argparse Errors - https://stackoverflow.com/questions/10900617/getting-syntax-error-near-unexpected-token-in-python
# Interpolation in Imshow - https://stackoverflow.com/questions/55121294/imshow-plot-with-no-data-values-excluded-from-interpolation
