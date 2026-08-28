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

        return (first_term + second_term) / 2


def dice_coefficient(actual_segmentation, predicted_segmentation):

    # See ID27 and ID29.

    true_positives = np.sum(np.logical_and(actual_segmentation == 1, predicted_segmentation == 1))

    false_positives = np.sum(np.logical_and(actual_segmentation == 0, predicted_segmentation == 1))

    false_negatives = np.sum(np.logical_and(actual_segmentation == 1, predicted_segmentation == 0))

    numerator = 2 * true_positives

    denominator = numerator + false_positives + false_negatives

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
