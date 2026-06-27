import numpy as np
from scipy.spatial.distance import cdist


def precision_recall_distances(actual_sources, predicted_sources, distance_threshold):

    # Actual sources and predicted sources are subset of below which have flux greater than given value

    # UTILITY FUNCTION FOR s90

    distances = cdist(predicted_sources, actual_sources, metric="euclidean")

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


def s90(actual_source_locations, predicted_source_locations, actual_source_energy_fluxes, distance_threshold=10 ):

    # From ID8 - assume energy fluxes in MeV and locations are in Cartesian coordinates (i.e. for patch 64 x 64, gives
    # locations of both in format (x, y) where x and y are in range [0, 63] (0-indexed)

    # The distance threshold is how far away a prediced source location can be (in pixels) before it is considered not
    # close enough to likely be the actual source it is closest to

    increasing_order = np.argsort(actual_source_energy_fluxes)

    for f in increasing_order:

        # Take all sources with flux above the flux of source f

        greater_flux = np.argwhere(actual_source_energy_fluxes > actual_source_energy_fluxes[f]).flatten()

        if (greater_flux.size != 0) and (greater_flux.size != 1):

            subset_actual_sources = actual_source_locations[greater_flux]
            subset_predicted_sources = predicted_source_locations[greater_flux]

            precision, recall = precision_recall_distances(subset_actual_sources, subset_predicted_sources,
                                                           distance_threshold)






test_ground = np.array([[0, 2], [64, 7], [9, 31], [6, 8], [23, 18]])
test_predicted = np.array([[61, 3], [0, 5], [2, 2], [48, 1], [19, 55]])
test_fluxes = np.array([0, 10000, 80000, 150000])  # np.random.choice(np.arange(100, 100000), size=4)

# print(test_fluxes)

s90(test_ground, test_predicted, actual_source_energy_fluxes=test_fluxes)
