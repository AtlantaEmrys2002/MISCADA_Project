from algorithms.components.classification_algorithms import classification_neural_network
from algorithms.components.clustering_algorithms import k_means_clustering
from algorithms.components.machine_learning_segmentation_algorithms import random_forest_segmentation
from algorithms.components.data_preparation import ml_segmentation_data_prep, prepare_classifier_data
import copy
import numpy as np
from pathlib import Path
from read_write_functions import save_predictions


# NEED TO CHECK IF THIS THE FIRST TIME TRAINING EACH SEGMENTATION ALGORITHM - IF IT IS THEN TRAIN IF NOT USE PRETRAINED
# SAVE EACH TRAINED CLASSIFIER AS [Name of classifier]_classifier_trained_on_[name of segmentation and localisation]_
# detection

# BALANCE DATASETS


def novel_source_extraction_algorithms(training_data, validation_data, testing_data, real_data, save_directory):
    # FILE INITIALISATION

    # Make directory to save results on simulated data
    Path(save_directory).mkdir(parents=True, exist_ok=True)

    # Make directory to save results on real data
    real_data_save_directory = save_directory + "/real"

    Path(real_data_save_directory).mkdir(parents=True, exist_ok=True)

    # ALGORITHMS

    segmentation_algorithms = ["random_forest"]

    localisation_algorithms = ["kmeans"]

    classification_algorithms = ["cnn"]

    for segment in segmentation_algorithms:

        match segment:

            case "random_forest":

                (train_patch_ids, training_maps, training_masks,
                 validation_patch_ids, validation_maps, validation_masks,
                 test_patch_ids, testing_maps, testing_masks) = (
                    ml_segmentation_data_prep(train_data=copy.deepcopy(training_data),
                                              validation_data=copy.deepcopy(validation_data),
                                              test_data=copy.deepcopy(testing_data)))

                (train_segmentation_predictions, validation_segmentation_predictions, test_segmentation_predictions) = \
                    (random_forest_segmentation(training_maps=training_maps, training_masks=training_masks,
                                                validation_maps=validation_maps, testing_maps=testing_maps))

                _, _, _, _, _, _, real_patch_ids, real_maps, real_masks = (
                    ml_segmentation_data_prep(train_data=np.array([]), validation_data=np.array([]),
                                              test_data=copy.deepcopy(real_data)))

                _, _, real_segmentation_predictions = random_forest_segmentation(training_maps=np.array([]),
                                                                                 training_masks=np.array([]),
                                                                                 validation_maps=np.array([]),
                                                                                 testing_maps=real_maps,
                                                                                 pretrained=True, real=True)

                print("Segmentation Algorithm Trained")

            case _:

                raise NameError("Segmentation algorithm {} could not be found.".format(segment))

        for local in localisation_algorithms:

            match local:

                case "kmeans":

                    if segment != "unet":

                        train_source_locations = k_means_clustering(train_segmentation_predictions,
                                                                    max_num_centroids=20)
                        validation_source_locations = k_means_clustering(validation_segmentation_predictions,
                                                                         max_num_centroids=20)
                        test_source_locations = k_means_clustering(test_segmentation_predictions, max_num_centroids=20)

                        real_source_locations = k_means_clustering(real_segmentation_predictions, max_num_centroids=20)

                    else:
                        train_source_locations = k_means_clustering(train_segmentation_predictions)
                        validation_source_locations = k_means_clustering(validation_segmentation_predictions)
                        test_source_locations = k_means_clustering(test_segmentation_predictions)
                        real_source_locations = k_means_clustering(real_segmentation_predictions)

                    print("Localisation Algorithm Applied")

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

                        # Train classifier on data
                        actual_labels, classifier_predictions = (
                            classification_neural_network(train_batches_class, validation_batches_class,
                                                          test_batches_class,
                                                          save_file="./algorithms/pre_trained_models/classifier_for{}_"
                                                                    "and_{}.pt".format(segment, local)))

                        real_actual_labels, real_classifier_predictions = (
                            classification_neural_network(np.array([]), np.array([]),
                                                          real_batches_class, pretrained=True,
                                                          save_file="./algorithms/pre_trained_models/classifier_for{}_"
                                                                    "and_{}.pt".format(segment, local)))

                        print("Classification Done.")

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
