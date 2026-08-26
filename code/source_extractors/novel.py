"""
This creates each novel extraction pipeline (each novel extraction pipeline is a combination of 1 segmentation
algorithm, 1 localisation algorithm, and 1 classification algorithm). It also ensures that algorithms are only trained
as needed - i.e. each segmentation algorithm requires training only once, whereas the localisation algorithms need to be
trained on the outputs of each segmentation algorithm and the classification algorithms need to be trained on the
outputs of each combination of segmentation and localisation algorithm (this is done through nested for loops).
"""

from algorithms.components.classification_algorithms import classification_neural_network
from algorithms.components.clustering_algorithms import (blob_detection, dbscan_clustering, k_means_clustering,
                                                         spectral_clustering)
from algorithms.components.machine_learning_classification_algorithms import random_forest_classifier, svm_classifier
from algorithms.components.machine_learning_segmentation_algorithms import random_forest_segmentation
from algorithms.components.segmentation_algorithms import unet
from algorithms.components.data_preparation import ml_segmentation_data_prep, prepare_classifier_data
import copy
import numpy as np
from pathlib import Path
from read_write_functions import save_predictions


# SAVE EACH TRAINED CLASSIFIER AS [Name of classifier]_classifier_trained_on_[name of segmentation and localisation]_
# detection

# BALANCE DATASETS!!!!!!!!!!!!!!!!!!


def novel_source_extraction_algorithms(training_data, validation_data, testing_data, real_data, save_directory):
    # FILE INITIALISATION

    # Make directory to save results on simulated data
    Path(save_directory).mkdir(parents=True, exist_ok=True)

    # Make directory to save results on real data
    real_data_save_directory = save_directory + "/real"

    Path(real_data_save_directory).mkdir(parents=True, exist_ok=True)

    # ALGORITHMS

    segmentation_algorithms = ["unet", "random_forest"]

    localisation_algorithms = ["dbscan", "kmeans", "spectral", "blob_detection"]

    classification_algorithms = ["cnn", "random_forest", "svm"]

    algorithm_count = 1

    # GET DATA IN USEFUL FORMAT - MAYBE MAKE SURE ONLY FORMAT FOR SEGMENTATION CLASSIFIER

    (train_patch_ids, training_maps, training_masks,
     validation_patch_ids, validation_maps, validation_masks,
     test_patch_ids, testing_maps, testing_masks) = (
        ml_segmentation_data_prep(train_data=copy.deepcopy(training_data),
                                  validation_data=copy.deepcopy(validation_data),
                                  test_data=copy.deepcopy(testing_data)))

    _, _, _, _, _, _, real_patch_ids, real_maps, real_masks = (
        ml_segmentation_data_prep(train_data=np.array([]), validation_data=np.array([]),
                                  test_data=copy.deepcopy(real_data)))

    for segment in segmentation_algorithms:

        match segment:

            case "unet":

                (train_segmentation_predictions, validation_segmentation_predictions, test_segmentation_predictions) = (
                    unet(training_maps=copy.deepcopy(training_maps), training_masks=copy.deepcopy(training_masks), validation_maps=copy.deepcopy(validation_maps), validation_masks=copy.deepcopy(validation_masks), testing_maps=copy.deepcopy(testing_maps)))

                _, _, real_segmentation_predictions = unet(training_maps=np.array([]), training_masks=np.array([]), validation_maps=np.array([]), validation_masks=np.array([]),
                                                           testing_maps=copy.deepcopy(real_maps), pretrained=True, real=True)

            case "random_forest":

                (train_segmentation_predictions, validation_segmentation_predictions, test_segmentation_predictions) = \
                    (random_forest_segmentation(training_maps=training_maps, training_masks=training_masks,
                                                validation_maps=validation_maps, testing_maps=testing_maps))

                _, _, real_segmentation_predictions = random_forest_segmentation(training_maps=np.array([]),
                                                                                 training_masks=np.array([]),
                                                                                 validation_maps=np.array([]),
                                                                                 testing_maps=real_maps,
                                                                                 pretrained=True, real=True)

            case _:

                raise NameError("Segmentation algorithm {} could not be found.".format(segment))

        for local in localisation_algorithms:

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

                    if segment != "unet":

                        train_source_locations = dbscan_clustering(train_segmentation_predictions, threshold=0.5)
                        validation_source_locations = dbscan_clustering(validation_segmentation_predictions,
                                                                        threshold=0.5)
                        test_source_locations = dbscan_clustering(test_segmentation_predictions, threshold=0.5)
                        real_source_locations = dbscan_clustering(real_segmentation_predictions, threshold=0.5)

                    else:
                        train_source_locations = dbscan_clustering(train_segmentation_predictions, threshold=0.5)
                        validation_source_locations = dbscan_clustering(validation_segmentation_predictions,
                                                                        threshold=0.5)
                        test_source_locations = dbscan_clustering(test_segmentation_predictions, threshold=0.5)
                        real_source_locations = dbscan_clustering(real_segmentation_predictions, threshold=0.5)

                case "blob_detection":

                    if segment != "unet":

                        train_source_locations = blob_detection(train_segmentation_predictions)
                        validation_source_locations = blob_detection(validation_segmentation_predictions)
                        test_source_locations = blob_detection(test_segmentation_predictions)
                        real_source_locations = blob_detection(real_segmentation_predictions)

                    else:
                        train_source_locations = blob_detection(train_segmentation_predictions)
                        validation_source_locations = blob_detection(validation_segmentation_predictions)
                        test_source_locations = blob_detection(test_segmentation_predictions)
                        real_source_locations = blob_detection(real_segmentation_predictions)

                case "spectral":

                    if segment != "unet":

                        train_source_locations = spectral_clustering(train_segmentation_predictions, threshold=0.5)
                        validation_source_locations = spectral_clustering(validation_segmentation_predictions,
                                                                          threshold=0.5)
                        test_source_locations = spectral_clustering(test_segmentation_predictions, threshold=0.5)

                        real_source_locations = spectral_clustering(real_segmentation_predictions, threshold=0.5)

                    else:
                        train_source_locations = spectral_clustering(train_segmentation_predictions, threshold=0.5)
                        validation_source_locations = spectral_clustering(validation_segmentation_predictions,
                                                                          threshold=0.5)
                        test_source_locations = spectral_clustering(test_segmentation_predictions, threshold=0.5)
                        real_source_locations = spectral_clustering(real_segmentation_predictions, threshold=0.5)

                case _:
                    raise NameError("Localisation algorithm {} could not be found.".format(local))

            for classifier in classification_algorithms:

                match classifier:

                    case "cnn":
                        # Prepare detected sources for classification (effectively, data prep)
                        train_batches_class = (
                            prepare_classifier_data(patches=copy.deepcopy(training_maps),
                                                    predicted_locations=copy.deepcopy(train_source_locations),
                                                    patch_ids=copy.deepcopy(train_patch_ids)))

                        validation_batches_class = (
                            prepare_classifier_data(patches=copy.deepcopy(validation_maps),
                                                    predicted_locations=copy.deepcopy(validation_source_locations),
                                                    patch_ids=copy.deepcopy(validation_patch_ids), shuffle_data=False))

                        test_batches_class = prepare_classifier_data(patches=copy.deepcopy(testing_maps),
                                                                     predicted_locations=copy.deepcopy(test_source_locations),
                                                                     patch_ids=copy.deepcopy(test_patch_ids), test=True)

                        real_batches_class = prepare_classifier_data(patches=copy.deepcopy(real_maps),
                                                                     predicted_locations=copy.deepcopy(real_source_locations),
                                                                     patch_ids=copy.deepcopy(real_patch_ids), test=True, real_data=True)

                        save_file_classifier = ("./algorithms/pre_trained_models/cnn_classifier_for_{}_and_{}.pt".
                                                format(segment, local))

                        # Train classifier on data
                        actual_labels, classifier_predictions = (
                            classification_neural_network(train_batches_class, validation_batches_class,
                                                          test_batches_class, save_file=save_file_classifier))

                        real_actual_labels, real_classifier_predictions = (
                            classification_neural_network(np.array([]), np.array([]),
                                                          real_batches_class, pretrained=True,
                                                          save_file=save_file_classifier))

                    case "random_forest":

                        # N.B. Here, we prepare classifier data as if it were all classifier data (to prevent in being
                        # sorted into batch loaders - that is a process only for data that will be fed to DL algorithms)

                        train_batches_class = (
                            prepare_classifier_data(patches=copy.deepcopy(training_maps),
                                                    predicted_locations=copy.deepcopy(train_source_locations),
                                                    patch_ids=copy.deepcopy(train_patch_ids), test=False, ml_data=True))

                        test_batches_class = prepare_classifier_data(patches=copy.deepcopy(testing_maps),
                                                                     predicted_locations=copy.deepcopy(test_source_locations),
                                                                     patch_ids=copy.deepcopy(test_patch_ids), test=True)

                        real_batches_class = prepare_classifier_data(patches=copy.deepcopy(real_maps),
                                                                     predicted_locations=copy.deepcopy(real_source_locations),
                                                                     patch_ids=copy.deepcopy(real_patch_ids), test=True, real_data=True)

                        save_file_classifier = ("./algorithms/pre_trained_models/rf_classifier_for_{}_and_{}.pt".
                                                format(segment, local))

                        actual_labels, classifier_predictions = (
                            random_forest_classifier(train_data=train_batches_class, test_data=test_batches_class,
                                                     save_file=save_file_classifier))

                        real_actual_labels, real_classifier_predictions = (
                            random_forest_classifier(train_data=np.array([]), test_data=real_batches_class,
                                                     save_file=save_file_classifier,
                                                     pretrained=True))

                    case "svm":

                        train_batches_class = (
                            prepare_classifier_data(patches=copy.deepcopy(training_maps),
                                                    predicted_locations=copy.deepcopy(train_source_locations),
                                                    patch_ids=copy.deepcopy(train_patch_ids), test=False, ml_data=True))

                        test_batches_class = prepare_classifier_data(patches=copy.deepcopy(testing_maps),
                                                                     predicted_locations=copy.deepcopy(test_source_locations),
                                                                     patch_ids=copy.deepcopy(test_patch_ids), test=True)

                        real_batches_class = prepare_classifier_data(patches=copy.deepcopy(real_maps),
                                                                     predicted_locations=copy.deepcopy(real_source_locations),
                                                                     patch_ids=copy.deepcopy(real_patch_ids), test=True, real_data=True)

                        save_file_classifier = ("./algorithms/pre_trained_models/svm_classifier_for_{}_and_{}.pt".
                                                format(segment, local))

                        actual_labels, classifier_predictions = (
                            svm_classifier(train_data=train_batches_class, test_data=test_batches_class,
                                           save_file=save_file_classifier))

                        real_actual_labels, real_classifier_predictions = (
                            svm_classifier(train_data=np.array([]), test_data=real_batches_class,
                                           save_file=save_file_classifier,
                                           pretrained=True))

                    case _:
                        raise NameError("Classification algorithm {} could not be found.".format(classifier))

                # SAVE RESULTS FOR SOURCE EXTRACTION ALGORITHM

                # Save predictions for test patches
                save_predictions(patch_ids=test_patch_ids, predicted_segmentations=test_segmentation_predictions,
                                 predicted_locations=test_source_locations,
                                 predicted_classes=classifier_predictions, actual_classes=actual_labels,
                                 directory=save_directory, method=f"{segment}_{local}_{classifier}")

                # Save predictions for real data
                save_predictions(patch_ids=real_patch_ids, predicted_segmentations=real_segmentation_predictions,
                                 predicted_locations=real_source_locations,
                                 predicted_classes=real_classifier_predictions, actual_classes=real_actual_labels,
                                 directory=real_data_save_directory, method=f"{segment}_{local}_{classifier}")

                print("Completed Novel Source Extraction Pipeline {}: {} + {} + {}".format(algorithm_count,
                                                                                           segment, local, classifier))

                algorithm_count += 1

# REFERENCES

# Double List Comprehension - https://stackoverflow.com/questions/1198777/double-iteration-in-list-comprehension
