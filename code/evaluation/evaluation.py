from metrics.localisation_metrics import chamfer_distance
from metrics.segmentation_metrics import (binary_balanced_accuracy, dice_coefficient, segmentation_precision,
                                          segmentation_recall)
import numpy as np
import os


def evaluate_classifiers():

    pass


def evaluate_localisation(actual_source_centers, predicted_source_centers):

    num_patches = actual_source_centers.shape[0]

    average_chamfer_distance = sum([chamfer_distance(actual_source_centers[p], predicted_source_centers[p]) for p in
                                    range(num_patches)]) / num_patches

    return average_chamfer_distance


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
    models = ["UNEK", "UNEB"]

    detectors = {"UNEK": "U-NET", "UNEB": "U-NET"}
    localisers = {"UNEK": "K-Means", "UNEB": "Blob Detection"}
    classifiers = {"UNEK": "CNN", "UNEB": "CNN"}

    results = []

    csv_headers = ("id,detection_algorithm,localisation_algorithm,classification_algorithm,"
                   "segmentation_balanced_binary_accuracy,segmentation_dive_coefficient,segmentation_precision,"
                   "segmentation_recall\n")

    # Set up file
    file = open("./results.csv", "w+")
    file.writelines(csv_headers)
    file.close()

    id = 0

    for m in models:

        segments = np.load("./../results/{}/segmentations.npy".format(m))

        actual = segments[:, 0]

        predicted = segments[:, 1]

        av_bin_balanced_acc, av_dice, av_prec, av_rec = evaluate_detection(actual_segmentations=actual,
                                                                           predicted_segmentations=predicted)

        results.append(f"{id},{detectors[m]},{localisers[m]},{classifiers[m]},{av_bin_balanced_acc},{av_dice},"
                       f"{av_prec},{av_rec}\n")

        id += 1

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
