import numpy as np


def binary_balanced_accuracy(actual_segmentation, predicted_segmentation):

    # Expect two binary arrays

    true_positives = np.sum(np.logical_and(actual_segmentation == 1, predicted_segmentation == 1))

    true_negatives = np.sum(np.logical_and(actual_segmentation == 0, predicted_segmentation == 0))

    false_positives = np.sum(np.logical_and(actual_segmentation == 0, predicted_segmentation == 1))

    false_negatives = np.sum(np.logical_and(actual_segmentation == 1, predicted_segmentation == 0))

    if (true_positives == 0 and false_negatives == 0) or (true_negatives == 0 and false_positives == 0):

        return 0

    else:

        first_term = true_positives / (true_positives + false_negatives)
        second_term = true_negatives / (true_negatives + false_positives)

        balanced_accuracy = (first_term + second_term) / 2

        return balanced_accuracy


def dice_coefficient(actual_segmentation, predicted_segmentation):

    # See ID27 and ID29.

    # We do not assume data is Boolean (see ID27) - if we did then Dice is same as F1 score. However, our data is
    # probabilistic and we may introduce Gaussian blurs in mask (instead of single point representing correctness).
    # Therefore, we introduce binary parameter - it says whether to convert predicted segmentation to Boolean (1s and 0s)
    # and calculate equivalent of F1-score, our keep probabilistic data.

    # Assume that we will be using Boolean data most of the time, but made sure to keep general.

    # Input format assumptions:
    # Actual segmentation - binary mask, indicating where actual sources are
    # Predicted segmentation - probability in each pixel that pixel contains source vs background

    # We are looking at the cardinality of the logical and
    numerator = np.sum(2 * np.logical_and(actual_segmentation, predicted_segmentation))

    denominator = 2 * actual_segmentation.size

    return 0 if denominator == 0 else numerator / denominator


def segmentation_precision(actual_segmentation, predicted_segmentation):

    true_positives = np.sum(np.logical_and(actual_segmentation == 1, predicted_segmentation == 1))

    false_positives = np.sum(np.logical_and(actual_segmentation == 0, predicted_segmentation == 1))

    return 0 if (true_positives == 0 and false_positives == 0) else true_positives / (true_positives + false_positives)


def segmentation_recall(actual_segmentation, predicted_segmentation):

    true_positives = np.sum(np.logical_and(actual_segmentation == 1, predicted_segmentation == 1))

    false_negatives = np.sum(np.logical_and(actual_segmentation == 1, predicted_segmentation == 0))

    return 0 if (true_positives == 0 and false_negatives == 0) else true_positives / (true_positives + false_negatives)

# REFERENCES
# Dice-Sorenson Coefficient - https://en.wikipedia.org/wiki/Dice-Sørensen_coefficient
# Precision and Recall - https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-
# recall
