import copy
from utils import get_catalog_data
from metrics.utils import im_cartesian_to_physical
from metrics.real_data_application_metrics import (percentage_of_4fgl_sources_detected, plot_predictions_actual,
                                                   percentage_of_4fgl_source_correctly_classified,
                                                   save_candidate_sources, compare_candidate_sources)
import numpy as np
from pathlib import Path
import pickle
from read_write_functions import get_patch_centres
from utils import get_classified_patches


def evaluate_on_real_data(file_4fgl, model, label):

    patch_centres, _ = get_patch_centres(patches_metadata_file="./../source_extractors/real_data/real_patches/patches/"
                                                            "patch_metadata.csv", real=True)

    # RESULTS

    # Get the positions of each of the detected sources in the real data

    # Get predicted locations from model (in x, y coordinates in 64 x 64 image)
    with open("./../results/real/{}/predicted_locations.data".format(model), 'rb') as f:
        predicted_locations_raw = pickle.load(f)

    predicted_locations_real = [
        im_cartesian_to_physical(coordinates=predicted_locations_raw[p],
                                                            patch_centre=patch_centres[p],
                                                            coordinate_system='C') for p in range(768) if
        predicted_locations_raw[p].shape != 0]

    predicted_locations_real = (
        np.unique(np.array([p for x in range(768) for p in predicted_locations_real[x]]), axis=0))

    # EVALUATE NUM oF 4FGL SOURCES DETECTED

    # Read in locations of 4FGL sources
    actual_source_locations_4fgl, actual_source_types = get_catalog_data(file_4fgl)

    frac_of_4fgl_sources_detected = percentage_of_4fgl_sources_detected(
        actual_source_locations=actual_source_locations_4fgl,
        predicted_source_locations=predicted_locations_real)

    Path("./../results/plots/source_discoveries_all_sky/").mkdir(parents=True, exist_ok=True)

    plot_predictions_actual(actual_coordinates=copy.deepcopy(actual_source_locations_4fgl),
                            predicted_coordinates=copy.deepcopy(predicted_locations_real), model=model, label=label)

    # THIS FRACTION IS THE NUMBER OF SOURCES CORRECTLY CLASSIFIED OF THE NUMBER OF SOURCES CORRECTLY DETECTED

    predicted_locations_in_real_data = get_classified_patches(predicted_locations_raw)

    predicted_locations_in_real_data_celestial = [
        im_cartesian_to_physical(coordinates=predicted_locations_in_real_data[p],
                                                            patch_centre=patch_centres[p],
                                                            coordinate_system='C') for p in range(768) if
        predicted_locations_in_real_data[p].shape != 0]

    predicted_locations_in_real_data_celestial = np.array([p for x in range(768)
                                                           for p in predicted_locations_in_real_data_celestial[x]])

    classifications_raw = np.load("./../results/real/{}/classifications.npy".format(model))

    # One-hot encoding of PREDICTED SOURCE LABELS - some classifications are based on probabilities

    classifications = np.zeros((classifications_raw.shape[0], 3))

    classifications[np.arange(classifications_raw.shape[0]), np.argmax(classifications_raw[:, 1], axis=1)] = 1

    frac_correct_classed_sources = (
        percentage_of_4fgl_source_correctly_classified(actual_source_locations=actual_source_locations_4fgl,
                                                       predicted_source_locations=
                                                       predicted_locations_in_real_data_celestial, classifications=
                                                       classifications, actual_classifications=actual_source_types))

    save_candidate_sources(actual_source_locations=actual_source_locations_4fgl,
                           predicted_source_locations=predicted_locations_in_real_data_celestial,
                           classifications=classifications, model=model)

    print(frac_of_4fgl_sources_detected, frac_correct_classed_sources)

    return frac_of_4fgl_sources_detected, frac_correct_classed_sources


if __name__ == "__main__":

    # TAKE INPUTS (RECOMMENDED READ IN FILE)

    segmentation_algorithms = ["unet", "pspnet", "random_forest"]
    localisation_algorithms = ["dbscan", "blob_detection", "kmeans", "spectral"]
    classification_algorithms = ["random_forest", "cnn"]

    segment_labels = ["UNET", "PSPNet", "Random Forest"]
    local_labels = ["DBSCAN", "Blob Detection", "$k$-Means", "Spectral"]
    classifier_labels = ["Random Forest", "CNN"]

    models = ["{}_{}_{}".format(i, j, k) for i in segmentation_algorithms for j in localisation_algorithms
              for k in classification_algorithms]

    # Labels for plots
    labels = [[i, j, k] for i in segment_labels for j in local_labels for k in classifier_labels]

    # model_lists = [[i, j, k] for i in segmentation_algorithms for j in localisation_algorithms
    #           for k in classification_algorithms]

    results = []

    # Create directory in which to save plots
    plot_directory = "./../results/plots"
    Path(plot_directory).mkdir(parents=True, exist_ok=True)

    model_id = 0

    # CALCULATE METRICS FOR EACH SOURCE EXTRACTION ALGORITHM

    compare_candidate_sources()

    for m in models:

        print(m)

        (frac_of_4fgl_sources_detected,
         frac_correct_classed_4fgl_sources) = evaluate_on_real_data(file_4fgl="/Volumes/T7/data/catalog/4FGL_DR4.fit", model=m, label=labels[model_id])

        # SAVE RESULTS

        model_id += 1

    # EVALUATE DIFFERENT STAGES FOR EACH MODEL WITH METRICS


# NEED TO MAKE SURE PLOTS DIRECTORY EXISTS AND CONFUSION MATRIX FILE EXISTS (SEE CLASSIFICATION METRICS FILE)

# FOR s90 - figure out the number of degrees (suggested is 0.3 degrees) before reject as true source representation and
# convert to pixels on image

# REFERENCES

# Array of Arrays - https://stackoverflow.com/questions/12020872/array-of-arrays-python-numpy
# Cartesian Products - https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-
# single-array-of-2d-points
# Is Finite - https://stackoverflow.com/questions/2831516/isnotnan-functionality-in-numpy-can-this-be-more-pythonic
# Number of Occurrences - https://stackoverflow.com/questions/28663856/how-do-i-count-the-occurrence-of-a-certain-item-
# in-an-ndarray
# Max Value in Dictionary - https://www.reddit.com/r/learnpython/comments/o4qksz/find_a_maximum_value_in_dictionary/
# One-Hot Encoding - https://stackoverflow.com/questions/20295046/numpy-change-max-in-each-row-to-1-all-other-numbers-to
# -0
# Pandas to Numpy - https://stackoverflow.com/questions/49734441/converting-pandas-dataframe-to-numpy-array-with-headers
# -and-dtypes
