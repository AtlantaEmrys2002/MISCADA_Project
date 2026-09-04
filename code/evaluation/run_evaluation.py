import copy
from utils import get_catalog_data
from metrics.utils import im_cartesian_to_physical
from metrics.classification_metrics import classification_confusion_matrix, classification_precision_recall
from metrics.detection_metrics import s90
from metrics.localisation_metrics import chamfer_separation, num_sources_correctly_detected
from metrics.real_data_application_metrics import (percentage_of_4fgl_sources_detected, plot_predictions_actual,
                                                   percentage_of_4fgl_source_correctly_classified,
                                                   save_candidate_sources)
from metrics.segmentation_metrics import (binary_balanced_accuracy, dice_coefficient, segmentation_precision,
                                          segmentation_recall)
import numpy as np
from pathlib import Path
import pickle
from read_write_functions import get_patch_centres, localisation_metadata, vector_labels_to_str
from utils import get_classified_patches
import warnings


def evaluate_classifiers(actual_class, predicted_class, method_name, directory):

    # Plot confusion matrices
    classification_confusion_matrix(ground_truth=actual_class, predicted=predicted_class, classifier_name=method_name,
                                    directory=directory)

    classification_precision, classification_recall = classification_precision_recall(ground_truth=np.array(actual_class), predicted=np.array(predicted_class))

    return classification_precision, classification_recall



def get_locations_per_catalog(actual_source_centers, predicted_source_centers,
                              patch_ids, patch_to_catalog_ids):

    # Separates locations into per catalog (celestial coordinates)

    num_catalogs = max(patch_to_catalog_ids.items())[1] + 1

    num_patches = len(actual_source_centers)

    catalog_separated_actual_locations = []

    catalog_separated_predicted_locations = []

    # catalog_separated_patch_ids = []

    for b in range(num_catalogs):

        a_catalog = [actual_source_centers[p] for p in range(num_patches) if int(patch_to_catalog_ids[patch_ids[p]]) == b]
        p_catalog = [predicted_source_centers[p] for p in range(num_patches) if int(patch_to_catalog_ids[patch_ids[p]]) == b]

        # a_catalog = np.unique(
        #     np.array([a_catalog[i][j] for i in range(len(a_catalog)) for j in range(a_catalog[i].shape[0])]), axis=0)
        #
        # p_catalog = np.unique(
        #     np.array([p_catalog[i][j] for i in range(len(p_catalog)) for j in range(p_catalog[i].shape[0])]), axis=0)

        catalog_separated_actual_locations.append(a_catalog)
        catalog_separated_predicted_locations.append(p_catalog)

    return catalog_separated_actual_locations, catalog_separated_predicted_locations


def evaluate_localisation(actual_source_centers, predicted_source_centers, ids, catalog_ids):


    actual_loc, predicted_loc = get_locations_per_catalog(actual_source_centers=actual_source_centers,
                                                                      predicted_source_centers=predicted_source_centers,
                                                                      patch_ids=ids, patch_to_catalog_ids=catalog_ids)

    num_catalogs = len(actual_loc)
    #
    # s90_value = s90(predicted_source_locations=predicted_loc, test_patch_ids = ids, patch_in_catalog=catalog_ids)
    #
    # print(s90_value)

    actual_loc = [np.unique(
        np.array([actual_loc[b][i][j] for i in range(len(actual_loc[b])) for j in range(actual_loc[b][i].shape[0])]), axis=0) for b in range(num_catalogs)]

    predicted_loc = [np.unique(
        np.array([predicted_loc[b][i][j] for i in range(len(predicted_loc[b])) for j in range(predicted_loc[b][i].shape[0])]), axis=0) for b in range(num_catalogs)]

    if np.any(np.array([len(k) for k in actual_loc]) == 0):

        warnings.warn("Catalogs {} do not have any predicted locations".format(
            str([str(k) + " " for k in range(num_catalogs) if k != num_catalogs - 1 and len(actual_loc[k]) == 0])))

    chamfer_seps = [chamfer_separation(actual_loc[b], predicted_loc[b]) for b in range(num_catalogs)]

    chamfer_distance_val = sum([chamfer_seps[b] for b in range(num_catalogs) if np.isfinite(chamfer_seps[b])])

    percentage_of_sources_detected = num_sources_correctly_detected(actual_loc, predicted_loc)

    print(chamfer_distance_val, percentage_of_sources_detected)

    print("END")

    return chamfer_distance_val, percentage_of_sources_detected # , s90_value


def evaluate_detection(actual_segmentations, predicted_segmentations):
    # N.B. actual segmentations and predicted segmentations should be numpy arrays

    num_patches = actual_segmentations.shape[0]

    average_binary_balanced_accuracy = sum([binary_balanced_accuracy(actual_segmentations[p],
                                                                     predicted_segmentations[p] > 0.5)
                                            for p in range(num_patches)]) / num_patches

    average_dice_coefficient = sum([dice_coefficient(actual_segmentations[p], predicted_segmentations[p] > 0.5) for p in
                                    range(num_patches)]) / num_patches

    average_precision = sum([segmentation_precision(actual_segmentations[p], predicted_segmentations[p] > 0.5) for p in
                             range(num_patches)]) / num_patches

    average_recall = sum([segmentation_recall(actual_segmentations[p], predicted_segmentations[p] > 0.5) for p in
                          range(num_patches)]) / num_patches

    return average_binary_balanced_accuracy, average_dice_coefficient, average_precision, average_recall


def evaluate_on_real_data(file_4fgl, model):

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
                            predicted_coordinates=copy.deepcopy(predicted_locations_real), model=model)

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

    return frac_of_4fgl_sources_detected, frac_correct_classed_sources



if __name__ == "__main__":

    # TAKE INPUTS (RECOMMENDED READ IN FILE)

    segmentation_algorithms = ["unet", "pspnet", "random_forest"]

    # localisation_algorithms = ["dbscan", "blob_detection", "kmeans", "spectral"]
    #
    # classification_algorithms = ["random_forest", "cnn", "svm"]

    localisation_algorithms = ["dbscan"]

    classification_algorithms = ["random_forest"] # ["cnn"] # , "random_forest"] # , "cnn"]

    models = ["{}_{}_{}".format(i, j, k) for i in segmentation_algorithms for j in localisation_algorithms
              for k in classification_algorithms]

    model_lists = [[i, j, k] for i in segmentation_algorithms for j in localisation_algorithms
              for k in classification_algorithms]

    results = []

    csv_headers = ("id,detection_algorithm,localisation_algorithm,classification_algorithm,"
                   "segmentation_balanced_binary_accuracy,segmentation_dice_coefficient,segmentation_precision,"
                   "segmentation_recall,chamfer_separation,frac_sources_detected,classification_precision,"
                   "classification_recall,frac_4fgl_detected,frac_4fgl_detected_and_classified\n")

    # Set up file
    file = open("./../results/results.csv", "w+")
    file.writelines(csv_headers)
    file.close()

    # Create directory in which to save plots
    plot_directory = "./../results/plots"
    Path(plot_directory).mkdir(parents=True, exist_ok=True)

    model_id = 0

    # # READ IN ACTUAL COORDINATES OF EACH SOURCE IN EACH CATALOG

    patch_centres, catalog_of_each_patch = get_patch_centres(
        patches_metadata_file="./../data_simulation/simulated_data/patches/patch_metadata.csv")

    # CALCULATE METRICS FOR EACH SOURCE EXTRACTION ALGORITHM

    for m in models:

        # IDs of patches used to test model
        patch_ids = np.load("./../results/{}/patch_ids.npy".format(m))

        # DETECTION (SEGMENTATION) EVALUATION

        segments = np.load("./../results/{}/segmentations.npy".format(m))

        actual_segments = segments[:, 0]

        predicted_segments = segments[:, 1]

        av_bin_balanced_acc, av_dice, av_prec, av_rec = evaluate_detection(actual_segmentations=actual_segments,
                                                                           predicted_segmentations=predicted_segments)

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

        for n in range(actual_segments.shape[0]):

            if actual_agn_locations_celestial[n].size != 0 and actual_psr_locations_celestial[n].size != 0:
                actual_source_locations_for_patch = np.vstack((actual_agn_locations_celestial[n],
                                                               actual_psr_locations_celestial[n]))
            elif actual_psr_locations_celestial[n].size == 0:
                actual_source_locations_for_patch = actual_agn_locations_celestial[n]
            else:
                actual_source_locations_for_patch = actual_psr_locations_celestial[n]

            actual_source_locations.append(actual_source_locations_for_patch)

        # Evaluate localisation
        # chamfer_distance, av_frac_sources_detected, full_s90 = (
        #     evaluate_localisation(actual_source_centers=copy.deepcopy(actual_source_locations), predicted_source_centers=
        #     copy.deepcopy(predicted_locations_celestial), ids=copy.deepcopy(patch_ids), catalog_ids=copy.deepcopy(catalog_of_each_patch)))

        chamfer_distance, av_frac_sources_detected = (
            evaluate_localisation(actual_source_centers=copy.deepcopy(actual_source_locations), predicted_source_centers=
            copy.deepcopy(predicted_locations_celestial), ids=copy.deepcopy(patch_ids), catalog_ids=copy.deepcopy(catalog_of_each_patch)))

        # CLASSIFICATION EVALUATION

        classifications = np.load("./../results/{}/classifications.npy".format(m))

        actual_classes, predicted_classes = vector_labels_to_str(classifications[:, 0]), vector_labels_to_str(
            classifications[:, 1])

        # This does not take into account any spatial distributions (at the moment!!!!!!)
        classification_precision, classification_recall = evaluate_classifiers(actual_class=copy.deepcopy(actual_classes),
                                                                               predicted_class=copy.deepcopy(predicted_classes),
                                                                               method_name=copy.deepcopy(m), directory=plot_directory)

        (frac_of_4fgl_sources_detected,
         frac_correct_classed_4fgl_sources) = evaluate_on_real_data(file_4fgl="/Volumes/T7/data/catalog/4FGL_DR4.fit", model=m)

        # SAVE RESULTS

        results.append(f"{model_id},{model_lists[model_id][0]},{model_lists[model_id][1]},{model_lists[model_id][2]},{av_bin_balanced_acc},{av_dice},"
                       f"{av_prec},{av_rec},{chamfer_distance},{av_frac_sources_detected},{classification_precision},{classification_recall},{frac_of_4fgl_sources_detected},{frac_correct_classed_4fgl_sources}\n")

        model_id += 1

    # EVALUATE DIFFERENT STAGES FOR EACH MODEL WITH METRICS

    # SAVE RESULTS TO FILE

    file = open("./../results/results.csv", "a")
    file.writelines(results)
    file.close()

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
