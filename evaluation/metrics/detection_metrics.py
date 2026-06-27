import numpy as np
from scipy.spatial.distance import cdist

def s90(actual_source_locations, predicted_source_locations, actual_source_energy_fluxes, distance_threshold=10 ):

    # From ID8 - assume energy fluxes in MeV and locations are in Cartesian coordinates (i.e. for patch 64 x 64, gives
    # locations of both in format (x, y) where x and y are in range [0, 63] (0-indexed)

    # The distance threshold is how far away a prediced source location can be (in pixels) before it is considered not
    # close enough to likely be the actual source it is closest to

    increasing_order = np.argsort(actual_source_energy_fluxes)

    distances = cdist(predicted_source_locations, actual_source_locations, metric="euclidean")

    # Find the closest actual source to each predicted source



    # FILL IN




    # Find number of sources





    for f in increasing_order:

        # Take all sources with flux above the flux of source f

        greater_flux = np.argwhere(actual_source_energy_fluxes > actual_source_energy_fluxes[f])



test_ground = np.array([[0, 2], [64, 7], [9, 31], [6, 8], [23, 18]])
test_predicted = np.array([[61, 3], [0, 5], [2, 2], [48, 1], [19, 55]])