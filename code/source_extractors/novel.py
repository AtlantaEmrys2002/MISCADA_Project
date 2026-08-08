"""
This creates each novel extraction pipeline (each novel extraction pipeline is a combination of 1 segmentation
algorithm, 1 localisation algorithm, and 1 classification algorithm). It also ensures that algorithms are only trained
as needed - i.e. each segmentation algorithm requires training only once, whereas the localisation algorithms need to be
trained on the outputs of each segmentation algorithm and the classification algorithms need to be trained on the
outputs of each combination of segmentation and localisation algorithm (this is done through nested for loops).
"""

from algorithms.components.classification_algorithms import classification_neural_network
from algorithms.components.clustering_algorithms import blob_detection, dbscan_clustering, k_means_clustering
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

    segmentation_algorithms = ["random_forest", "unet"]

    localisation_algorithms = ["dbscan", "blob_detection", "kmeans"]

    classification_algorithms = ["svm", "random_forest", "cnn"]

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
                    unet(training_maps=training_data, validation_maps=validation_data, testing_maps=testing_data))

                _, _, real_segmentation_predictions = unet(training_maps=np.array([]), validation_maps=np.array([]),
                                                           testing_maps=real_data, pretrained=True, real=True)

                import matplotlib.pyplot as plt

                # print(np.sum(train_segmentation_predictions[0] > 0.5))

                plt.imshow((train_segmentation_predictions[0] > 0.5) * 200)

                plt.show()

                plt.imshow((train_segmentation_predictions[1] > 0.5) * 200)

                plt.show()

            case "random_forest":

                # (train_patch_ids, training_maps, training_masks,
                #  validation_patch_ids, validation_maps, validation_masks,
                #  test_patch_ids, testing_maps, testing_masks) = (
                #     ml_segmentation_data_prep(train_data=copy.deepcopy(training_data),
                #                               validation_data=copy.deepcopy(validation_data),
                #                               test_data=copy.deepcopy(testing_data)))

                (train_segmentation_predictions, validation_segmentation_predictions, test_segmentation_predictions) = \
                    (random_forest_segmentation(training_maps=training_maps, training_masks=training_masks,
                                                validation_maps=validation_maps, testing_maps=testing_maps))

                # _, _, _, _, _, _, real_patch_ids, real_maps, real_masks = (
                #     ml_segmentation_data_prep(train_data=np.array([]), validation_data=np.array([]),
                #                               test_data=copy.deepcopy(real_data)))

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

                    # print("Localisation Algorithm Applied")

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

                    # print("Localisation Algorithm Applied")

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

                case _:
                    raise NameError("Localisation algorithm {} could not be found.".format(segment))

            for classifier in classification_algorithms:

                match classifier:

                    case "cnn":
                        # Prepare detected sources for classification (effectively, data prep)
                        train_batches_class = (
                            prepare_classifier_data(patches=training_maps,
                                                    predicted_locations=train_source_locations,
                                                    patch_ids=train_patch_ids))

                        validation_batches_class = (
                            prepare_classifier_data(patches=validation_maps,
                                                    predicted_locations=validation_source_locations,
                                                    patch_ids=validation_patch_ids, shuffle_data=False))

                        test_batches_class = prepare_classifier_data(patches=testing_maps,
                                                                     predicted_locations=test_source_locations,
                                                                     patch_ids=test_patch_ids, test=True)

                        real_batches_class = prepare_classifier_data(patches=real_maps,
                                                                     predicted_locations=real_source_locations,
                                                                     patch_ids=real_patch_ids, test=True)

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

                        # print("Classification Done.")

                    case "random_forest":

                        # N.B. Here, we prepare classifier data as if it were all classifier data (to prevent in being
                        # sorted into batch loaders - that is a process only for data that will be fed to DL algorithms)

                        train_batches_class = (
                            prepare_classifier_data(patches=training_maps,
                                                    predicted_locations=train_source_locations,
                                                    patch_ids=train_patch_ids, test=True))

                        test_batches_class = prepare_classifier_data(patches=testing_maps,
                                                                     predicted_locations=test_source_locations,
                                                                     patch_ids=test_patch_ids, test=True)

                        real_batches_class = prepare_classifier_data(patches=real_maps,
                                                                     predicted_locations=real_source_locations,
                                                                     patch_ids=real_patch_ids, test=True)

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
                            prepare_classifier_data(patches=training_maps,
                                                    predicted_locations=train_source_locations,
                                                    patch_ids=train_patch_ids, test=True))

                        test_batches_class = prepare_classifier_data(patches=testing_maps,
                                                                     predicted_locations=test_source_locations,
                                                                     patch_ids=test_patch_ids, test=True)

                        real_batches_class = prepare_classifier_data(patches=real_maps,
                                                                     predicted_locations=real_source_locations,
                                                                     patch_ids=real_patch_ids, test=True)

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
                        raise NameError("Classification algorithm {} could not be found.".format(segment))

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
