import numpy as np

def balanced_binary_accuracy(predicted, actual):

    # N.B. Used for tuning of random forest classifier for segmentation - predicted is predicted masks and actual are
    # actual masks

    predicted = predicted.reshape(actual.shape[0], 64, 64)

    # Make so classes are 0 and 1 rather than 1 and 2
    actual -= 1
    predicted -= 1

    true_positives = np.sum(np.logical_and(actual == 1, predicted == 1))

    true_negatives = np.sum(np.logical_and(actual == 0, predicted == 0))

    false_positives = np.sum(np.logical_and(actual == 0, predicted == 1))

    false_negatives = np.sum(np.logical_and(actual == 1, predicted == 0))

    if (true_positives + false_negatives == 0) or (true_negatives + false_positives == 0):

        bba = 0

    else:

        first_term = true_positives / (true_positives + false_negatives)
        second_term = true_negatives / (true_negatives + false_positives)

        bba = (first_term + second_term) / 2

    return bba


