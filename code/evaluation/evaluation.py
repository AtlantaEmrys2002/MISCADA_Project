from astropy.table import QTable
import copy
from utils import integral_photon_flux_agn, integral_photon_flux_pulsar, get_catalog_data
from metrics.utils import image_cartesian_coordinates_to_physical_coordinates
from metrics.classification_metrics import classification_confusion_matrix
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


def evaluate_classifiers(actual_class, predicted_class, method_name, directory):
    # Plot confusion matrices
    classification_confusion_matrix(ground_truth=actual_class, predicted=predicted_class, classifier_name=method_name,
                                    directory=directory)


def evaluate_localisation(actual_source_centers, predicted_source_centers):
    num_patches = len(actual_source_centers)

    average_chamfer_distance = sum([chamfer_separation(actual_source_centers[p], predicted_source_centers[p]) for p in
                                    range(num_patches)]) / num_patches

    average_percentage_of_sources_detected = (
            sum([num_sources_correctly_detected(actual_source_locations[p], predicted_locations_celestial[p]) for p in
                 range(num_patches)]) / num_patches)

    return average_chamfer_distance, average_percentage_of_sources_detected


def evaluate_detection(actual_segmentations, predicted_segmentations):
    # N.B. actual segmentations and predicted segmentations should be numpy arrays

    num_patches = actual_segmentations.shape[0]

    average_binary_balanced_accuracy = sum([binary_balanced_accuracy(actual_segmentations[p],
                                                                     predicted_segmentations[p])
                                            for p in range(num_patches)]) / num_patches

    average_dice_coefficient = sum([dice_coefficient(actual_segmentations[p], predicted_segmentations[p]) for p in
                                    range(num_patches)]) / num_patches

    average_precision = sum([segmentation_precision(actual_segmentations[p], predicted_segmentations[p]) for p in
                             range(num_patches)]) / num_patches

    average_recall = sum([segmentation_recall(actual_segmentations[p], predicted_segmentations[p]) for p in
                          range(num_patches)]) / num_patches

    return average_binary_balanced_accuracy, average_dice_coefficient, average_precision, average_recall


def get_classified_patches(predicted_locations_in_real_data_raw):
    # see if it would have been classified or not (i.e. if it was too close to the edge. If it was too close to
    # edge of 64 x 64 image, then remove.

    predicted_locations_in_real_data = []

    for p in range(len(predicted_locations_in_real_data_raw)):

        loc_in_patch = predicted_locations_in_real_data_raw[p]

        if loc_in_patch.shape[0] != 0:

            xs = loc_in_patch[:, 0]
            ys = loc_in_patch[:, 1]

            mask = np.logical_not(((xs - 3) < 0) | ((xs + 4) > 63) | ((ys - 3) < 0) | ((ys + 4) > 63))

            predicted_locations_in_real_data.append(copy.deepcopy(predicted_locations_in_real_data_raw[p][mask]))

        else:

            predicted_locations_in_real_data.append(np.array([]))

    return predicted_locations_in_real_data


def evaluate_on_real_data(file_4fgl, model):

    print(model)

    patch_centres = get_patch_centres(patches_metadata_file="./../source_extractors/real_data/real_patches/patches/"
                                                            "patch_metadata.csv")

    # RESULTS

    # Get the positions of each of the detected sources in the real data

    # Get predicted locations from model (in x, y coordinates in 64 x 64 image)
    with open("./../results/real/{}/predicted_locations.data".format(model), 'rb') as f:
        predicted_locations_in_real_data_raw = pickle.load(f)

    predicted_locations_in_real_data_celestial = [
        image_cartesian_coordinates_to_physical_coordinates(coordinates=copy.deepcopy(predicted_locations_in_real_data_raw[p]),
                                                            patch_centre=patch_centres[p],
                                                            coordinate_system='C') for p in range(768) if
        predicted_locations_in_real_data_raw[p].shape != 0]

    new = []

    for x in range(768):
        for p in predicted_locations_in_real_data_celestial[x]:
            new.append(p)

    predicted_locations_in_real_data_celestial = np.array(new)

    # EVALUATE NUM oF 4FGL SOURCES DETECTED

    # Read in locations of 4FGL sources
    actual_source_locations_4fgl, actual_source_types = get_catalog_data(file_4fgl)

    frac_of_4fgl_sources_detected = percentage_of_4fgl_sources_detected(
        actual_source_locations=actual_source_locations_4fgl,
        predicted_source_locations=copy.deepcopy(predicted_locations_in_real_data_celestial))

    Path("./../results/plots/source_discoveries_all_sky/").mkdir(parents=True, exist_ok=True)

    plot_predictions_actual(actual_coordinates=actual_source_locations_4fgl,
                            predicted_coordinates=predicted_locations_in_real_data_celestial, model=model)

    # THIS FRACTION IS THE NUMBER OF SOURCES CORRECTLY CLASSIFIED OF THE NUMBER OF SOURCES CORRECTLY DETECTED

    predicted_locations_in_real_data = get_classified_patches(predicted_locations_in_real_data_raw)

    predicted_locations_in_real_data_celestial = [
        image_cartesian_coordinates_to_physical_coordinates(coordinates=predicted_locations_in_real_data[p],
                                                            patch_centre=patch_centres[p],
                                                            coordinate_system='C') for p in range(768) if
        predicted_locations_in_real_data[p].shape != 0]

    new = []

    for x in range(768):
        for p in predicted_locations_in_real_data_celestial[x]:
            new.append(p)

    predicted_locations_in_real_data_celestial = np.array(new)

    classifications = np.load("./../results/real/{}/classifications.npy".format(model))

    print(len(predicted_locations_in_real_data_celestial), len(classifications))

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

    # Add name of models here
    # models = ["UNEK", "UNEB"]
    # models = ["UNEK"]

    # detectors = {"UNEK": "U-NET", "UNEB": "U-NET"}
    # localisers = {"UNEK": "K-Means", "UNEB": "Blob Detection"}
    # classifiers = {"UNEK": "CNN", "UNEB": "CNN"}

    segmentation_algorithms = ["random_forest", "unet"]

    localisation_algorithms = ["dbscan", "blob_detection", "kmeans", "spectral"]

    classification_algorithms = ["random_forest", "cnn", "svm"]

    models = []

    for i in segmentation_algorithms:

        for j in localisation_algorithms:

            for k in classification_algorithms:

                models.append("{}_{}_{}".format(i, j, k))

    results = []

    csv_headers = ("id,detection_algorithm,localisation_algorithm,classification_algorithm,"
                   "segmentation_balanced_binary_accuracy,segmentation_dive_coefficient,segmentation_precision,"
                   "segmentation_recall,chamfer_separation,frac_sources_detected\n")

    # Set up file
    file = open("./../results/results.csv", "w+")
    file.writelines(csv_headers)
    file.close()

    # Create directory in which to save plots
    plot_directory = "./../results/plots"
    Path(plot_directory).mkdir(parents=True, exist_ok=True)

    model_id = 0

    # # READ IN ACTUAL COORDINATES OF EACH SOURCE IN EACH CATALOG

    patch_centres = get_patch_centres(
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
            image_cartesian_coordinates_to_physical_coordinates(coordinates=predicted_locations[p],
                                                                patch_centre=patch_centres[patch_ids[p]],
                                                                coordinate_system='C') for p in range(len(patch_ids))]

        # Get locations of actual sources (in celestial coordinates) within each patch
        actual_agn_locations_celestial, actual_psr_locations_celestial = localisation_metadata(patch_ids=patch_ids)

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
        av_chamfer_distance, av_frac_sources_detected = (
            evaluate_localisation(actual_source_centers=actual_source_locations, predicted_source_centers=
            predicted_locations_celestial))

        # CLASSIFICATION EVALUATION

        classifications = np.load("./../results/{}/classifications.npy".format(m))

        actual_classes, predicted_classes = vector_labels_to_str(classifications[:, 0]), vector_labels_to_str(
            classifications[:, 1])

        # This does not take into account any spatial distributions (at the moment!!!!!!)
        evaluate_classifiers(actual_class=actual_classes, predicted_class=predicted_classes, method_name=m,
                             directory=plot_directory)

        evaluate_on_real_data(file_4fgl="/Volumes/T7/data/catalog/4FGL_DR4.fit", model=m)

        # SAVE RESULTS

        results.append(f"{model_id},{m},{av_bin_balanced_acc},{av_dice},"
                       f"{av_prec},{av_rec},{av_chamfer_distance},{av_frac_sources_detected}\n")

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

# Cartesian Products - https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-
# single-array-of-2d-points
