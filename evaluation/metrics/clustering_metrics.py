import numpy as np
from scipy.spatial.distance import cdist

test_ground = np.array([[0, 2], [64, 7], [9, 31], [6, 8], [23, 18]])
test_predicted = np.array([[61, 3], [0, 5], [2, 2], [48, 1], [19, 55]])


def chamfer_distance(predicted_source_centres, true_source_centres):

    # ID11 - evaluates localisation power of faint source detection (how close predicted point sources are to actual
    # point source)

    # A = Predicted source centres
    # B = Ground truth source centres

    distances = cdist(predicted_source_centres, true_source_centres, metric="euclidean")

    dist_ab = np.sum(np.min(distances, axis=1))

    dist_ba = np.sum(np.min(distances, axis=0))

    dist_chamfer = dist_ab + dist_ba

    return dist_chamfer



# REFERENCES

# Chamfer Distance - https://medium.com/@sim30217/chamfer-distance-4207955e8612
# Scipy Documentation - https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist
