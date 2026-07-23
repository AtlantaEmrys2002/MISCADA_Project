from metrics.localisation_metrics import chamfer_distance
from metrics.segmentation_metrics import (binary_balanced_accuracy, dice_coefficient, segmentation_precision,
                                          segmentation_recall)
import numpy as np


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
    models = ["UNEK"]

    results = []

    csv_headers = ("id,detection_algorithm,localisation_algorithm,classification_algorithm,"
                   "segmentation_balanced_binary_accuracy,segmentation_dive_coefficient,segmentation_precision,"
                   "segmentation_recall")

    for m in models:

        segments = np.load("./../results/{}/segmentations.npy".format(m))

        # num_patches = segments.shape[0]

        actual = segments[:, 0]

        predicted = segments[:, 1]

        av_bin_balanced_acc, av_dice, av_prec, av_rec = evaluate_detection(actual_segmentations=actual,
                                                                           predicted_segmentations=predicted)

        results.append()



    # EVALUATE DIFFERENT STAGES FOR EACH MODEL WITH METRICS

    # For each model, get metrics for all TEST PATCHES (not training/validation patches)

    # for m in models:
    #
    #




    # SAVE RESULTS TO FILE - RECOMMEND ROWS WITH SEGMENTATION, LOCALISATION, AND CLASSIFICATION METRICS LISTED FIRST FOLLOWED BY METRICS







# NEED TO MAKE SURE PLOTS DIRECTORY EXISTS AND CONFUSION MATRIX FILE EXISTS (SEE CLASSIFICATION METRICS FILE)

# FOR s90 - figure out the number of degrees (suggested is 0.3 degrees) before reject as true source representation and
# convert to pixels on image

# REFERENCES

# Cartesian Products - https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-
# single-array-of-2d-points
