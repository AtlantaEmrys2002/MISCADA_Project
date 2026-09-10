import copy
from metrics.utils import im_cartesian_to_physical
from metrics.detection_metrics import s90
import numpy as np
from pathlib import Path
import pickle
from read_write_functions import get_patch_centres, localisation_metadata, vector_labels_to_str


def get_locations_per_catalog(actual_source_centers, predicted_source_centers,
                              patch_ids, patch_to_catalog_ids):

    # Separates locations into per catalog (celestial coordinates)

    num_catalogs = max(patch_to_catalog_ids.items())[1] + 1

    num_patches = len(actual_source_centers)

    catalog_separated_actual_locations = []

    catalog_separated_predicted_locations = []

    for b in range(num_catalogs):

        a_catalog = [actual_source_centers[p] for p in range(num_patches) if int(patch_to_catalog_ids[patch_ids[p]]) == b]
        p_catalog = [predicted_source_centers[p] for p in range(num_patches) if int(patch_to_catalog_ids[patch_ids[p]]) == b]

        catalog_separated_actual_locations.append(a_catalog)
        catalog_separated_predicted_locations.append(p_catalog)

    return catalog_separated_actual_locations, catalog_separated_predicted_locations



def evaluate_localisation(actual_source_centers, predicted_source_centers, ids, catalog_ids):


    actual_loc, predicted_loc = get_locations_per_catalog(actual_source_centers=actual_source_centers,
                                                                      predicted_source_centers=predicted_source_centers,
                                                                      patch_ids=ids, patch_to_catalog_ids=catalog_ids)

    s90_value = s90(predicted_source_locations=predicted_loc, test_patch_ids = ids, patch_in_catalog=catalog_ids)

    print(s90_value)

    return s90_value


if __name__ == "__main__":

    # TAKE INPUTS (RECOMMENDED READ IN FILE)

    # segmentation_algorithms = ["unet", "pspnet", "random_forest"]
    #
    # localisation_algorithms = ["dbscan", "blob_detection", "kmeans", "spectral"]
    #
    # classification_algorithms = ["random_forest", "cnn"]

    segmentation_algorithms = ["random_forest", "unet", "pspnet"]

    localisation_algorithms = ["dbscan", "blob_detection", "kmeans", "spectral"]

    classification_algorithms = ["random_forest", "cnn"]

    models = ["{}_{}_{}".format(i, j, k) for i in segmentation_algorithms for j in localisation_algorithms
              for k in classification_algorithms]

    model_lists = [[i, j, k] for i in segmentation_algorithms for j in localisation_algorithms
              for k in classification_algorithms]

    results = []

    # Create directory in which to save plots
    plot_directory = "./../results/plots"
    Path(plot_directory).mkdir(parents=True, exist_ok=True)

    model_id = 0

    # # READ IN ACTUAL COORDINATES OF EACH SOURCE IN EACH CATALOG

    patch_centres, catalog_of_each_patch = get_patch_centres(
        patches_metadata_file="./../data_simulation/simulated_data/patches/patch_metadata.csv")

    # CALCULATE METRICS FOR EACH SOURCE EXTRACTION ALGORITHM

    for m in models:

        print(m)

        # IDs of patches used to test model
        patch_ids = np.load("./../results/{}/patch_ids.npy".format(m))

        # LOCALISATION EVALUATION

        # Get predicted locations from model (in x, y coordinates in 64 x 64 image)
        with open("./../results/{}/predicted_locations.data".format(m), 'rb') as f:
            predicted_locations = pickle.load(f)

        # Convert predicted locations to celestial RA/DEC coordinates

        predicted_locations_celestial = [
            im_cartesian_to_physical(coordinates=copy.deepcopy(predicted_locations[p]),
                                                                patch_centre=copy.deepcopy(patch_centres[patch_ids[p]]),
                                                                coordinate_system='C') for p in range(patch_ids.shape[0])]

        # Get locations of actual sources (in celestial coordinates) within each patch
        actual_agn_locations_celestial, actual_psr_locations_celestial = localisation_metadata(patch_ids=copy.deepcopy(patch_ids))

        actual_source_locations = []

        for n in range(patch_ids.shape[0]):

            if actual_agn_locations_celestial[n].size != 0 and actual_psr_locations_celestial[n].size != 0:
                actual_source_locations_for_patch = np.vstack((actual_agn_locations_celestial[n],
                                                               actual_psr_locations_celestial[n]))
            elif actual_psr_locations_celestial[n].size == 0:
                actual_source_locations_for_patch = actual_agn_locations_celestial[n]
            else:
                actual_source_locations_for_patch = actual_psr_locations_celestial[n]

            actual_source_locations.append(actual_source_locations_for_patch)

        # Evaluate localisation
        full_s90 = (
            evaluate_localisation(actual_source_centers=copy.deepcopy(actual_source_locations), predicted_source_centers=
            copy.deepcopy(predicted_locations_celestial), ids=copy.deepcopy(patch_ids), catalog_ids=copy.deepcopy(catalog_of_each_patch)))

        print("S90: {}".format(full_s90))

        # SAVE RESULTS

        model_id += 1
