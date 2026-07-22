import numpy as np


def binary_balanced_accuracy(actual_segmentation, predicted_segmentation):

    # Expect two binary arrays

    true_positives = np.sum(np.logical_and(actual_segmentation == 1, predicted_segmentation == 1))

    true_negatives = np.sum(np.logical_and(actual_segmentation == 0, predicted_segmentation == 0))

    false_positives = np.sum(np.logical_and(actual_segmentation == 0, predicted_segmentation == 1))

    false_negatives = np.sum(np.logical_and(actual_segmentation == 1, predicted_segmentation == 0))

    first_term = true_positives / (true_positives + false_negatives)
    second_term = true_negatives / (true_negatives + false_positives)

    balanced_accuracy = (first_term + second_term) / 2

    return balanced_accuracy


def dice_coefficient(actual_segmentation, predicted_segmentation, threshold=0.5, binary=True):

    # See ID27 and ID29.

    # We do not assume data is Boolean (see ID27) - if we did then Dice is same as F1 score. However, our data is
    # probabilistic and we may introduce Gaussian blurs in mask (instead of single point representing correctness).
    # Therefore, we introduce binary parameter - it says whether to convert predicted segmentation to Boolean (1s and 0s)
    # and calculate equivalent of F1-score, our keep probabilistic data.

    # Assume that we will be using Boolean data most of the time, but made sure to keep general.

    # Input format assumptions:
    # Actual segmentation - binary mask, indicating where actual sources are
    # Predicted segmentation - probability in each pixel that pixel contains source vs background

    if binary is True:
        predicted_segmentation = (predicted_segmentation > threshold).astype(np.uint8)

    # We are looking at the cardinality of the logical and
    numerator = np.sum(2 * np.logical_and(actual_segmentation, predicted_segmentation))

    denominator = 2 * actual_segmentation.size

    dice = numerator / denominator

    return dice


def segmentation_precision(actual_segmentation, predicted_segmentation):

    true_positives = np.sum(np.logical_and(actual_segmentation == 1, predicted_segmentation == 1))

    false_positives = np.sum(np.logical_and(actual_segmentation == 0, predicted_segmentation == 1))

    precision = true_positives / (true_positives / false_positives)

    return precision


def segmentation_recall(actual_segmentation, predicted_segmentation):

    true_positives = np.sum(np.logical_and(actual_segmentation == 1, predicted_segmentation == 1))

    false_negatives = np.sum(np.logical_and(actual_segmentation == 1, predicted_segmentation == 0))

    return true_positives / (true_positives + false_negatives)


# REFERENCES
# Dice-Sorenson Coefficient - https://en.wikipedia.org/wiki/Dice-Sørensen_coefficient
