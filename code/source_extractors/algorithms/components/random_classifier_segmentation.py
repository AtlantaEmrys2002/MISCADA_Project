"""
Conducts map segmentation using traditional ML (as opposed to DL) approaches - the Random Forest Segmentor was adapted
from a tutorial included in scikit-image documentation (the tutorial can be found here - https://scikit-image.org/docs/
stable/auto_examples/segmentation/plot_trainable_segmentation.html#id4)
"""

import copy
from .data_preparation import extract_features
import numpy as np
from pickle import dump, load
from skimage import future
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
import time
from .utils import balanced_binary_accuracy


def random_forest_segmentation(training_maps, training_masks, validation_maps, testing_maps, validation_masks=None,
                               pretrained=False, real=False, tune=False,
                               save_file="./algorithms/pre_trained_models/random_forest_segmentation.pt"):
    """
    Consulted this heavily when implementing - https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_
    trainable_segmentation.html

    Parameters
    ----------
    training_maps
    training_masks
    validation_maps
    testing_maps
    validation_masks
    sigma_min
    sigma_max
    pretrained
    real
    save_file
    tune

    Returns
    -------

    """

    # PREPARE DATA

    # N.B. That we add 1 here to indicate classes
    training_masks = np.array(training_masks) + 1

    if isinstance(validation_masks, np.ndarray):

        validation_masks = np.array(validation_masks) + 1

    if not pretrained:

        # TRAIN RANDOM FOREST CLASSIFIER ON DATA

        if not tune:

            # Found model overfits if number of patches used to train is greater than 250
            if training_maps.shape[0] > 250:

                random_indices = np.random.choice(a=training_maps.shape[0], size=250, replace=False)

                training_maps_subset = training_maps[random_indices]
                training_masks_subset = training_masks[random_indices]

            else:

                training_maps_subset = training_maps
                training_masks_subset = training_masks

            start = time.time()

            # Extract local features from training data and stitch training images together to create giant image upon
            # which to train
            training_features, training_labels = extract_features(training_maps_subset, training_masks_subset)

            # Best values found during tuning
            clf = RandomForestClassifier(n_estimators=150, n_jobs=5, max_depth=20, max_samples=0.05, max_features=None,
                                         oob_score=balanced_accuracy_score)

            # GET PREDICTIONS
            clf = future.fit_segmenter(training_labels, training_features, clf)

            end = time.time() - start

            print(f"Training Time: {end}")

            # SAVE TRAINED SEGMENTATION ALGORITHM
            with open(save_file, "wb") as f:
                dump(clf, f)

        else:

            # Best BBA found when only fitting 250 patches
            best_bba = 0
            best_time = 0

            best_parameters = [0, 0, 0, 0, 0, 0]

            # N.B. cannot use grid search as not implemented for multi-dimensional targets
            for n in [30, 50, 60, 70, 90, 100, 150]:

                for m in [5, 7, 9, 10, 15, 20]:

                    for ms in [0.01, 0.025, 0.05]:

                        for mf in [None, "sqrt", "log2"]:

                            for sm in [1]:

                                for sx in [4, 8, 16]:

                                    start = time.time()

                                    training_features, training_labels = extract_features(maps=training_maps,
                                                                                          masks=training_masks,
                                                                                          sigma_min=sm, sigma_max=sx)

                                    validation_features, validation_labels = extract_features(maps=validation_maps,
                                                                                              masks=validation_masks,
                                                                                              sigma_min=sm, sigma_max=sx)

                                    clf = RandomForestClassifier(n_estimators=n, max_depth=m, max_samples=ms, max_features=mf,
                                                                 n_jobs=-1, oob_score=balanced_accuracy_score)

                                    clf = future.fit_segmenter(training_labels, training_features, clf)

                                    predicted_tmp = future.predict_segmenter(validation_features, clf)

                                    # Evaluate on validation data (similar to deep learning techniques
                                    bba = balanced_binary_accuracy(predicted=predicted_tmp,
                                                                   actual=copy.deepcopy(validation_masks))

                                    end = time.time() - start

                                    if (bba > best_bba) or (bba == best_bba and best_time > end):
                                        best_bba = bba
                                        best_time = end
                                        best_parameters = [n, m, ms, mf, sm, sx]

                                    print(f"{n}, {m}, {ms}, {mf}, {sm}, {sx}, {end}, {bba}")

            if best_parameters == 0:
                raise RuntimeError("Tuning of random forest classifier for segmentation failed.")

            n, m, ms, mf, sm, sx = best_parameters

            print("Best Binary Balanced Accuracy: {} %".format(best_bba * 100))
            print(f"Best Parameters: {n}, {m}, {ms}, {mf}, {sm}, {sx}")
            print(f"Training Time: {end}")

            # Train using best parameters
            clf = RandomForestClassifier(n_estimators=n, max_depth=m, max_samples=ms, max_features=mf,
                                         n_jobs=-1, oob_score=balanced_accuracy_score)

            clf = future.fit_segmenter(training_masks, training_maps, clf)

            # Save trained model
            with open(save_file, "wb") as f:
                dump(clf, f)

    else:

        # Use pre-trained RF segmentor
        with open(save_file, "rb") as f:
            clf = load(f)

    # FORMAT DATA

    # GET PREDICTIONS

    if real:

        testing_features, testing_labels = extract_features(testing_maps, np.array([]))

        test_data_predictions = future.predict_segmenter(testing_features, clf) - 1
        test_pred = np.reshape(test_data_predictions, shape=(testing_maps.shape[0], 1, 64, 64))

        return np.array([]), np.array([]), test_pred

    else:

        train_data_predictions = np.zeros((training_maps.shape[0], 1, 64, 64))
        validation_data_predictions = np.zeros((validation_maps.shape[0], 1, 64, 64))
        test_data_predictions = np.zeros((testing_maps.shape[0], 1, 64, 64))

        # Similar to batch loading - prevents CPU from running out of storage
        for s in range(0, training_maps.shape[0], 1000):

            lower = s
            upper = min(s + 1000, training_maps.shape[0])

            training_features, _ = extract_features(training_maps[lower:upper], np.array([]))

            train_pred = future.predict_segmenter(training_features, clf) - 1

            train_pred = np.reshape(train_pred, shape=(upper - lower, 1, 64, 64))

            train_data_predictions[lower:upper] = train_pred

        for s in range(0, validation_maps.shape[0], 1000):

            lower = s
            upper = min(s + 1000, validation_maps.shape[0])

            validation_features, _ = extract_features(validation_maps[lower:upper], np.array([]))

            valid_pred = future.predict_segmenter(validation_features, clf) - 1

            valid_pred = np.reshape(valid_pred, shape=(upper - lower, 1, 64, 64))

            validation_data_predictions[lower:upper] = valid_pred


        for s in range(0, testing_maps.shape[0], 1000):
            lower = s
            upper = min(s + 1000, testing_maps.shape[0])

            testing_features, _ = extract_features(testing_maps[lower:upper], np.array([]))

            test_pred = future.predict_segmenter(testing_features, clf) - 1

            test_pred = np.reshape(test_pred, shape=(upper - lower, 1, 64, 64))

            test_data_predictions[lower:upper] = test_pred

        return train_data_predictions, validation_data_predictions, test_data_predictions

# REFERENCES

# Debugging - https://stackoverflow.com/questions/58925808/python-indexerror-boolean-index-did-not-match-indexed-array-
# along-dimension-0
# Estimators and Features - https://stackoverflow.com/questions/46234806/what-n-estimators-and-max-features-means-in-
# randomforestregressor
# Grid Search Multiclass - https://stackoverflow.com/questions/26018543/gridsearch-for-multi-label-classification-in-
# scikit-learn
# Improving RF Classifier - https://www.reddit.com/r/learnmachinelearning/comments/10xhff9/significant_ways_to_improve_
# model_accuracy/
# Improving RF Classifier - https://community.databricks.com/t5/machine-learning/how-do-i-improve-the-performance-of-my-
# random-forest-model-on/td-p/142507
# Improving RF Classifier - https://stackoverflow.com/questions/53634808/how-to-improve-performance-of-random-forest-
# multiclass-classification-model
# Increasing Data Size - https://forum.image.sc/t/adding-samples-to-scikit-image-random-forest-segmentation/55802
# Random Forest Best Practice - https://stackoverflow.com/questions/72081793/best-practices-to-run-a-random-forest-model
# -as-fast-as-possible
# Random Forest Classifier - https://stackoverflow.com/questions/43640546/how-to-make-randomforestclassifier-faster
# Random Forest Classifier - https://www.quora.com/What-makes-Random-Forest-outperform-the-support-vector-machine-SVM-
# and-the-euclidean-distance
# Random Forest Classifier Hyperparameters - https://medium.com/analytics-vidhya/random-forest-classifier-and-its-
# hyperparameters-8467bec755f6
# Random Forest Discussion - https://stats.stackexchange.com/questions/112148/when-to-avoid-random-forest/112151#112151
# Random Forest Large Dataset - https://stats.stackexchange.com/questions/487173/fitting-a-random-forest-classifier-on-a
# -large-dataset
# Random Forest Memory - https://stackoverflow.com/questions/30766253/scikit-learn-random-forest-taking-up-too-much-
# memory
# Reshape Array - https://stackoverflow.com/questions/7372316/how-can-i-make-a-two-dimensional-numpy-array-a-three-
# dimensional-array
# Reshape Array - https://stackoverflow.com/questions/52670278/how-to-reshape-a-numpy-array-from-x-y-z-to-y-z-x
# Saving sklearn Models - https://stackoverflow.com/questions/56107259/how-to-save-a-trained-model-by-scikit-learn
# Segmentation Theory - https://stackoverflow.com/questions/51439053/conditional-random-field-in-semantic-segmentation
# Tuning Parameters - https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/
# Tutorial Code - https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_trainable_segmentation.html#id4
