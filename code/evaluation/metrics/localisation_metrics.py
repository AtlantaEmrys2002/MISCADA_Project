import numpy as np
from scipy.spatial.distance import cdist


def chamfer_distance(actual_source_centres, predicted_source_centres):
    # ID11 - evaluates localisation power of faint source detection (how close predicted point sources are to actual
    # point source)

    # A = Predicted source centres
    # B = Ground truth source centres

    distances = cdist(predicted_source_centres, actual_source_centres)

    dist_ab = np.sum(np.min(distances, axis=1))

    dist_ba = np.sum(np.min(distances, axis=0))

    dist_chamfer = dist_ab + dist_ba

    return dist_chamfer


# REFERENCES

# Chamfer Distance - https://medium.com/@sim30217/chamfer-distance-4207955e8612
# Distance Calculations - https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-
# numpy
# Mesh grids - https://www.geeksforgeeks.org/python/numpy-meshgrid-function/
# Scipy Documentation - https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.
# spatial.distance.cdist
