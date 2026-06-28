import numpy as np
from scipy.spatial.distance import cdist
import warnings


def precision_recall_distances(actual_sources, predicted_sources, distance_threshold):

    # Actual sources and predicted sources are subset of below which have flux greater than given value

    # UTILITY FUNCTION FOR s90

    distances = cdist(predicted_sources, actual_sources)

    # Find the closest actual source to each predicted source
    closest_predicted_source_to_each_ground_source = np.argmin(distances, axis=0)

    # Find the closest predicted source to each actual source
    closest_ground_source_to_each_actual_predicted_source = np.argmin(distances, axis=1)

    # Find number of true positive sources (predicted source is within a given distance of an actual source that is its
    # nearest neigbour)

    true_positives = np.sum(np.array([distances[closest_predicted_source_to_each_ground_source[k], k] for k in
                                      range(distances.shape[0])]) <= distance_threshold)

    false_negatives = np.sum(np.array([distances[closest_predicted_source_to_each_ground_source[k], k] for k in
                                       range(distances.shape[0])]) >= distance_threshold)

    false_positives = np.sum(np.array([distances[k, closest_ground_source_to_each_actual_predicted_source[k]] for k in
                                       range(distances.shape[0])]) >= distance_threshold)

    precision = true_positives / (true_positives + false_positives)

    recall = true_positives / (true_positives + false_negatives)

    return precision, recall


def s90(actual_source_locations, predicted_source_locations, signal_to_noise_ratio, distance_threshold=10):
    # From ID8 - assume energy fluxes in MeV and locations are in Cartesian coordinates (i.e. for patch 64 x 64, gives
    # locations of both in format (x, y) where x and y are in range [0, 63] (0-indexed)

    # The distance threshold is how far away a predicted source location can be (in pixels) before it is considered not
    # close enough to likely be the actual source it is closest to

    increasing_order = np.argsort(signal_to_noise_ratio)

    for f in increasing_order:

        # Take all sources with flux above the flux of source f

        greater_snr = np.argwhere(signal_to_noise_ratio > signal_to_noise_ratio[f]).flatten()

        if (greater_snr.size != 0) and (greater_snr.size != 1):

            subset_actual_sources = actual_source_locations[greater_snr]
            subset_predicted_sources = predicted_source_locations[greater_snr]

            precision, recall = precision_recall_distances(subset_actual_sources, subset_predicted_sources,
                                                           distance_threshold)

            if (precision > 0.9) and (recall > 0.9):
                return greater_snr

    warnings.warn("There is never a signal to noise ratio above which both the precision and recall of this algorithm "
                  "is 90%", RuntimeWarning)

    return signal_to_noise_ratio[increasing_order[-1]]


# REFERENCES

# Python Documentation - https://docs.python.org/3/library/exceptions.html#RuntimeWarning
# Warnings - https://stackoverflow.com/questions/3891804/raise-warning-in-python-without-interrupting-program
