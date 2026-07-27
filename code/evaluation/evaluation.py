from metrics.utils import image_cartesian_coordinates_to_physical_coordinates
from metrics.localisation_metrics import chamfer_separation, num_sources_correctly_detected
from metrics.segmentation_metrics import (binary_balanced_accuracy, dice_coefficient, segmentation_precision,
                                          segmentation_recall)
import numpy as np
import pickle
from read_write_functions import get_patch_centres, localisation_metadata


def evaluate_classifiers():

    pass


def evaluate_localisation(actual_source_centers, predicted_source_centers):

    num_patches = len(actual_source_centers)

    average_chamfer_distance = sum([chamfer_separation(actual_source_centers[p], predicted_source_centers[p]) for p in
                                    range(num_patches)]) / num_patches

    average_percentage_of_sources_detected = (
            sum([num_sources_correctly_detected(actual_source_locations[p], predicted_locations_celestial[p]) for p in
                 range(num_patches)]) / num_patches)

    return average_chamfer_distance, average_percentage_of_sources_detected


def evaluate_detection(actual_segmentations, predicted_segmentations):

    # N.B. actual segmentations and predicted segmentations should be numpy arrays

    num_patches = actual_segmentations.shape[0]

    average_binary_balanced_accuracy = sum([binary_balanced_accuracy(actual_segmentations[p],
                                                                     predicted_segmentations[p])
                                            for p in range(num_patches)]) / num_patches

    average_dice_coefficient = sum([dice_coefficient(actual_segmentations[p], predicted_segmentations[p]) for p in
                                    range(num_patches)]) / num_patches

    average_precision = sum([segmentation_precision(actual_segmentations[p], predicted_segmentations[p]) for p in
                             range(num_patches)]) / num_patches

    average_recall = sum([segmentation_recall(actual_segmentations[p], predicted_segmentations[p]) for p in
                          range(num_patches)]) / num_patches

    return average_binary_balanced_accuracy, average_dice_coefficient, average_precision, average_recall


if __name__ == "__main__":

    # TAKE INPUTS (RECOMMENDED READ IN FILE)

    # Add name of models here
    # models = ["UNEK", "UNEB"]
    models = ["UNEK"]

    detectors = {"UNEK": "U-NET", "UNEB": "U-NET"}
    localisers = {"UNEK": "K-Means", "UNEB": "Blob Detection"}
    classifiers = {"UNEK": "CNN", "UNEB": "CNN"}

    results = []

    csv_headers = ("id,detection_algorithm,localisation_algorithm,classification_algorithm,"
                   "segmentation_balanced_binary_accuracy,segmentation_dive_coefficient,segmentation_precision,"
                   "segmentation_recall,chamfer_separation,frac_sources_detected\n")

    # Set up file
    file = open("./results.csv", "w+")
    file.writelines(csv_headers)
    file.close()

    model_id = 0

    # # READ IN ACTUAL COORDINATES OF EACH SOURCE IN EACH CATALOG

    patch_centres = get_patch_centres(patches_metadata_file="./../data_simulation/simulated_data/patches/patch_metadata.csv")

    # CALCULATE METRICS FOR EACH SOURCE EXTRACTION ALGORITHM

    for m in models:

        # IDs of patches used to test model
        patch_ids = np.load("./../results/{}/patch_ids.npy".format(m))

        # DETECTION (SEGMENTATION) EVALUATION

        segments = np.load("./../results/{}/segmentations.npy".format(m))

        actual_segments = segments[:, 0]

        predicted_segments = segments[:, 1]

        av_bin_balanced_acc, av_dice, av_prec, av_rec = evaluate_detection(actual_segmentations=actual_segments,
                                                                           predicted_segmentations=predicted_segments)

        # LOCALISATION EVALUATION

        # Get predicted locations from model (in x, y coordinates in 64 x 64 image)
        with open("./../results/{}/predicted_locations.data".format(m), 'rb') as f:

            predicted_locations = pickle.load(f)

        # Convert predicted locations to celestial RA/DEC coordinates

        predicted_locations_celestial = [
            image_cartesian_coordinates_to_physical_coordinates(coordinates=predicted_locations[p],
                                                                patch_centre=patch_centres[patch_ids[p]],
                                                                coordinate_system='C') for p in range(len(patch_ids))]

        # Get locations of actual sources (in celestial coordinates) within each patch
        actual_agn_locations_celestial, actual_psr_locations_celestial = localisation_metadata(patch_ids=patch_ids)

        actual_source_locations = []

        for n in range(actual_segments.shape[0]):

            if actual_agn_locations_celestial[n].size != 0 and actual_psr_locations_celestial[n].size != 0:
                actual_source_locations_for_patch = np.vstack((actual_agn_locations_celestial[n],
                                                               actual_psr_locations_celestial[n]))
            elif actual_psr_locations_celestial[n].size == 0:
                actual_source_locations_for_patch = actual_agn_locations_celestial[n]
            else:
                actual_source_locations_for_patch = actual_psr_locations_celestial[n]

            actual_source_locations.append(actual_source_locations_for_patch)

        # Evaluate localisation
        av_chamfer_distance, av_frac_sources_detected = (
            evaluate_localisation(actual_source_centers=actual_source_locations, predicted_source_centers=
            predicted_locations_celestial))

        # SAVE RESULTS

        results.append(f"{model_id},{detectors[m]},{localisers[m]},{classifiers[m]},{av_bin_balanced_acc},{av_dice},"
                       f"{av_prec},{av_rec},{av_chamfer_distance},{av_frac_sources_detected}\n")

        model_id += 1

    # EVALUATE DIFFERENT STAGES FOR EACH MODEL WITH METRICS

    # SAVE RESULTS TO FILE

    file = open("./results.csv", "a")
    file.writelines(results)
    file.close()


# NEED TO MAKE SURE PLOTS DIRECTORY EXISTS AND CONFUSION MATRIX FILE EXISTS (SEE CLASSIFICATION METRICS FILE)

# FOR s90 - figure out the number of degrees (suggested is 0.3 degrees) before reject as true source representation and
# convert to pixels on image

# REFERENCES

# Cartesian Products - https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-
# single-array-of-2d-points
