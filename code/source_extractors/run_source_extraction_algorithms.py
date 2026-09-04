"""
This creates each novel extraction pipeline (each novel extraction pipeline is a combination of 1 segmentation
algorithm, 1 localisation algorithm, and 1 classification algorithm). It also ensures that algorithms are only trained
as needed - i.e. each segmentation algorithm requires training only once, whereas the localisation algorithms need to be
trained on the outputs of each segmentation algorithm and the classification algorithms need to be trained on the
outputs of each combination of segmentation and localisation algorithm (this is done through nested for loops).
"""

import argparse
from algorithms.components.classification_algorithms import classification_neural_network
from algorithms.components.clustering_algorithms import (blob_detection, dbscan_clustering, k_means_clustering,
                                                         spectral_clustering)
from algorithms.components.machine_learning_classification_algorithms import random_forest_classifier, svm_classifier
from algorithms.components.pspnet import pspnet
from algorithms.components.random_classifier_segmentation import random_forest_segmentation
from algorithms.components.unet import unet
from algorithms.components.data_preparation import prepare_classifier_data
import copy
import csv
import numpy as np
from pathlib import Path
from read_write_functions import read_patches, save_predictions
import time


if __name__ == "__main__":
    # PROCESS USER INPUT

    parser = argparse.ArgumentParser(description="Reads in a collection of specified patches and formats them to be "
                                                 "passed to benchmark algorithms. These algorithms are then applied and"
                                                 "the parameters of the 'best' versions of them are saved.")

    parser.add_argument("--patch_location", required=True, type=str, help="The directory in which the"
                                                                          "patch data is stored.")

    parser.add_argument("--num_patches", required=True, type=int, help="The number of patches to read from"
                                                                       "the specified patch directory.")

    parser.add_argument("--save_directory", required=True, type=str, help="Location in which to save the "
                                                                          "predictions and results for each source"
                                                                          "extraction method.")

    args = parser.parse_args()

    patches_directory = args.patch_location
    num_patches = args.num_patches
    save_directory = args.save_directory

    real_data_save_directory = save_directory + "/real"

    Path(real_data_save_directory).mkdir(parents=True, exist_ok=True)

    # READ IN PATCHES CORRECTLY

    print("Reading in and formatting patches...")

    # CHANGE BACK FOR FINAL RUN

    (train_patch_ids, training_maps, training_masks,
    validation_patch_ids, validation_maps, validation_masks, test_patch_ids, testing_maps, testing_masks) = (
        read_patches(num_patches=76800, directory=patches_directory))

    real_patch_ids, real_maps, real_masks = read_patches(num_patches=768, directory="./real_data/real_patches/patches",
                                                         split=False)

    # FILE INITIALISATION

    # Make directory to save results on simulated data
    Path(save_directory).mkdir(parents=True, exist_ok=True)

    # Make directory to save results on real data
    real_data_save_directory = save_directory + "/real"

    Path(real_data_save_directory).mkdir(parents=True, exist_ok=True)

    # ALGORITHMS

    segmentation_algorithms = ["pspnet", "random_forest", "unet"]

    # localisation_algorithms = ["dbscan", "kmeans", "spectral", "blob_detection"]
    #
    # classification_algorithms = ["cnn", "random_forest", "svm"]

    # CHANGE BACK

    localisation_algorithms = ["dbscan"] #, "kmeans", "blob_detection", "spectral"]

    classification_algorithms = ["random_forest"] # ["cnn"] # , "random_forest"]

    algorithm_count = 1

    execution_times = []






    # MAKE SURE NONE NONE OF Pretrained parameters are set for hand in - we are training the segmentation algorithms now and using the results repeatedly

    for segment in segmentation_algorithms:

        print("Segmentation Algorithm: {}".format(segment))

        start_segmentation_time = time.time()

        match segment:

            case "unet":

                (train_segmentation_predictions, validation_segmentation_predictions, test_segmentation_predictions) = (
                    unet(training_maps=copy.deepcopy(training_maps), training_masks=copy.deepcopy(training_masks),
                         validation_maps=copy.deepcopy(validation_maps),
                         validation_masks=copy.deepcopy(validation_masks), testing_maps=copy.deepcopy(testing_maps), pretrained=True))

                _, _, real_segmentation_predictions = unet(training_maps=np.array([]), training_masks=np.array([]),
                                                           validation_maps=np.array([]), validation_masks=np.array([]),
                                                           testing_maps=copy.deepcopy(real_maps), pretrained=True,
                                                           real=True)

            case "random_forest":

                (train_segmentation_predictions, validation_segmentation_predictions, test_segmentation_predictions) = \
                    (random_forest_segmentation(training_maps=copy.deepcopy(training_maps),
                                                training_masks=copy.deepcopy(training_masks),
                                                validation_maps=copy.deepcopy(validation_maps),
                                                validation_masks=copy.deepcopy(validation_masks),
                                                testing_maps=copy.deepcopy(testing_maps), pretrained=True))

                _, _, real_segmentation_predictions = random_forest_segmentation(training_maps=np.array([]),
                                                                                 training_masks=np.array([]),
                                                                                 validation_maps=np.array([]),
                                                                                 testing_maps=real_maps,
                                                                                 pretrained=True, real=True)

            case "pspnet":

                (train_segmentation_predictions, validation_segmentation_predictions, test_segmentation_predictions) = (
                    pspnet(training_maps=copy.deepcopy(training_maps), training_masks=copy.deepcopy(training_masks),
                         validation_maps=copy.deepcopy(validation_maps),
                         validation_masks=copy.deepcopy(validation_masks), testing_maps=copy.deepcopy(testing_maps), pretrained=True))

                _, _, real_segmentation_predictions = pspnet(training_maps=np.array([]), training_masks=np.array([]),
                                                           validation_maps=np.array([]), validation_masks=np.array([]),
                                                           testing_maps=copy.deepcopy(real_maps), pretrained=True,
                                                           real=True)

            case _:

                raise NameError("Segmentation algorithm {} could not be found.".format(segment))

        segmentation_time = time.time() - start_segmentation_time

        for local in localisation_algorithms:

            start_localisation_time = time.time()

            match local:

                case "kmeans":

                    if segment != "unet":

                        train_source_locations = k_means_clustering(train_segmentation_predictions,
                                                                    max_num_centroids=20, threshold=0.5)
                        validation_source_locations = k_means_clustering(validation_segmentation_predictions,
                                                                         max_num_centroids=20, threshold=0.5)
                        test_source_locations = k_means_clustering(test_segmentation_predictions, max_num_centroids=20,
                                                                   threshold=0.5)

                        real_source_locations = k_means_clustering(real_segmentation_predictions, max_num_centroids=20,
                                                                   threshold=0.5)

                    else:
                        train_source_locations = k_means_clustering(train_segmentation_predictions, threshold=0.5)
                        validation_source_locations = k_means_clustering(validation_segmentation_predictions,
                                                                         threshold=0.5)
                        test_source_locations = k_means_clustering(test_segmentation_predictions, threshold=0.5)
                        real_source_locations = k_means_clustering(real_segmentation_predictions, threshold=0.5)

                case "dbscan":

                    train_source_locations = dbscan_clustering(train_segmentation_predictions, threshold=0.5)
                    validation_source_locations = dbscan_clustering(validation_segmentation_predictions,
                                                                    threshold=0.5)
                    test_source_locations = dbscan_clustering(test_segmentation_predictions, threshold=0.5)
                    real_source_locations = dbscan_clustering(real_segmentation_predictions, threshold=0.5)

                case "blob_detection":

                    train_source_locations = blob_detection(train_segmentation_predictions)
                    validation_source_locations = blob_detection(validation_segmentation_predictions)
                    test_source_locations = blob_detection(test_segmentation_predictions)
                    real_source_locations = blob_detection(real_segmentation_predictions)

                case "spectral":

                    train_source_locations = spectral_clustering(train_segmentation_predictions, threshold=0.5)
                    validation_source_locations = spectral_clustering(validation_segmentation_predictions,
                                                                      threshold=0.5)
                    test_source_locations = spectral_clustering(test_segmentation_predictions, threshold=0.5)
                    real_source_locations = spectral_clustering(real_segmentation_predictions, threshold=0.5)

                case _:
                    raise NameError("Localisation algorithm {} could not be found.".format(local))

            localisation_time = time.time() - start_localisation_time

            # # Prepare detected sources for classification (effectively, data prep)
            # train_batches_class = (
            #     prepare_classifier_data(patches=copy.deepcopy(training_maps),
            #                             predicted_locations=train_source_locations,
            #                             patch_ids=copy.deepcopy(train_patch_ids)))
            #
            # validation_batches_class = (
            #     prepare_classifier_data(patches=copy.deepcopy(validation_maps),
            #                             predicted_locations=validation_source_locations,
            #                             patch_ids=copy.deepcopy(validation_patch_ids)))
            #
            # test_batches_class = prepare_classifier_data(patches=copy.deepcopy(testing_maps),
            #                                              predicted_locations=test_source_locations,
            #                                              patch_ids=copy.deepcopy(test_patch_ids), test=True)
            #
            # real_batches_class = prepare_classifier_data(patches=real_maps, predicted_locations=real_source_locations,
            #                                              patch_ids=real_patch_ids, test=True, real_data=True)

            for classifier in classification_algorithms:

                start_classification_time = time.time()

                match classifier:

                    case "cnn":

                        # Prepare detected sources for classification (effectively, data prep)
                        train_batches_class = (
                            prepare_classifier_data(patches=copy.deepcopy(training_maps),
                                                    predicted_locations=train_source_locations,
                                                    patch_ids=copy.deepcopy(train_patch_ids)))

                        validation_batches_class = (
                            prepare_classifier_data(patches=copy.deepcopy(validation_maps),
                                                    predicted_locations=validation_source_locations,
                                                    patch_ids=copy.deepcopy(validation_patch_ids)))

                        test_batches_class = prepare_classifier_data(patches=copy.deepcopy(testing_maps),
                                                                     predicted_locations=test_source_locations,
                                                                     patch_ids=copy.deepcopy(test_patch_ids), test=True)

                        real_batches_class = prepare_classifier_data(patches=real_maps,
                                                                     predicted_locations=real_source_locations,
                                                                     patch_ids=real_patch_ids, test=True,
                                                                     real_data=True)

                        save_file_classifier = ("./algorithms/pre_trained_models/cnn_classifier_for_{}_and_{}.pt".
                                                format(segment, local))

                        # Train classifier on data
                        actual_labels, classifier_predictions = (
                            classification_neural_network(copy.deepcopy(train_batches_class),
                                                          copy.deepcopy(validation_batches_class),
                                                          copy.deepcopy(test_batches_class),
                                                          save_file=save_file_classifier))

                        real_actual_labels, real_classifier_predictions = (
                            classification_neural_network(np.array([]), np.array([]),
                                                          copy.deepcopy(real_batches_class), pretrained=True,
                                                          save_file=save_file_classifier))

                    case "random_forest":

                        # N.B. Here, we prepare classifier data as if it were all classifier data (to prevent in being
                        # sorted into batch loaders - that is a process only for data that will be fed to DL algorithms)

                        # Prepare detected sources for classification (effectively, data prep)
                        train_batches_class = (
                            prepare_classifier_data(patches=copy.deepcopy(training_maps),
                                                    predicted_locations=train_source_locations,
                                                    patch_ids=copy.deepcopy(train_patch_ids), ml=True))

                        test_batches_class = prepare_classifier_data(patches=copy.deepcopy(testing_maps),
                                                                     predicted_locations=test_source_locations,
                                                                     patch_ids=copy.deepcopy(test_patch_ids), test=True)

                        real_batches_class = prepare_classifier_data(patches=real_maps,
                                                                     predicted_locations=real_source_locations,
                                                                     patch_ids=real_patch_ids, test=True,
                                                                     real_data=True)

                        save_file_classifier = ("./algorithms/pre_trained_models/rf_classifier_for_{}_and_{}.pt".
                                                format(segment, local))

                        actual_labels, classifier_predictions = (
                            random_forest_classifier(train_data=copy.deepcopy(train_batches_class),
                                                     test_data=copy.deepcopy(test_batches_class),
                                                     save_file=save_file_classifier, tune=True))

                        real_actual_labels, real_classifier_predictions = (
                            random_forest_classifier(train_data=np.array([]),
                                                     test_data=copy.deepcopy(real_batches_class),
                                                     save_file=save_file_classifier, pretrained=True))

                    case "svm":

                        save_file_classifier = ("./algorithms/pre_trained_models/svm_classifier_for_{}_and_{}.pt".
                                                format(segment, local))

                        actual_labels, classifier_predictions = (
                            svm_classifier(train_data=copy.deepcopy(train_batches_class),
                                           test_data=copy.deepcopy(test_batches_class),
                                           save_file=save_file_classifier))

                        real_actual_labels, real_classifier_predictions = (
                            svm_classifier(train_data=np.array([]), test_data=copy.deepcopy(real_batches_class),
                                           save_file=save_file_classifier,
                                           pretrained=True))

                    case _:
                        raise NameError("Classification algorithm {} could not be found.".format(classifier))

                classification_time = time.time() - start_classification_time

                # SAVE RESULTS FOR SOURCE EXTRACTION ALGORITHM

                # Save predictions for test patches
                save_predictions(patch_ids=copy.deepcopy(test_patch_ids),
                                 predicted_segmentations=test_segmentation_predictions,
                                 predicted_locations=test_source_locations,
                                 predicted_classes=classifier_predictions, actual_classes=actual_labels,
                                 directory=save_directory, method=f"{segment}_{local}_{classifier}")

                # Save predictions for real data
                save_predictions(patch_ids=copy.deepcopy(real_patch_ids),
                                 predicted_segmentations=real_segmentation_predictions,
                                 predicted_locations=real_source_locations,
                                 predicted_classes=real_classifier_predictions, actual_classes=real_actual_labels,
                                 directory=real_data_save_directory, method=f"{segment}_{local}_{classifier}")

                print("Completed Novel Source Extraction Pipeline {}: {} + {} + {}".format(algorithm_count,
                                                                                           segment, local, classifier))

                execution_times.append([f"{segment}_{local}_{classifier}", segmentation_time, localisation_time, classification_time])

                algorithm_count += 1

    # Save execution times

    fields = ["algorithm", "segmentation_time (s)", "localisation_time(s)", "classification_time (s)"]

    with open("./../results/execution_times.csv", mode="w", newline='') as f:

        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(execution_times)

    print("Complete")

# REFERENCES

# Argparse Errors - https://stackoverflow.com/questions/10900617/getting-syntax-error-near-unexpected-token-in-python
# Double List Comprehension - https://stackoverflow.com/questions/1198777/double-iteration-in-list-comprehension
# Interpolation in imshow - https://stackoverflow.com/questions/55121294/imshow-plot-with-no-data-values-excluded-from-
# interpolation
# Save CSV Results - https://www.geeksforgeeks.org/python/python-save-list-to-csv/
