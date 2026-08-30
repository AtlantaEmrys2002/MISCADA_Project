"""
Conducts map segmentation using traditional ML (as opposed to DL) approaches - the Random Forest Segmentor was adapted
from a tutorial included in scikit-image documentation (the tutorial can be found here - https://scikit-image.org/docs/
stable/auto_examples/segmentation/plot_trainable_segmentation.html#id4)
"""

from functools import partial
import numpy as np
from pickle import dump, load
from skimage import feature, future
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
import time


def random_forest_segmentation(training_maps, training_masks, validation_maps, testing_maps, validation_masks=None, sigma_min=1, sigma_max=16,
                               pretrained=False, real=False,
                               save_file="./algorithms/pre_trained_models/random_forest_segmentation.pt", tune=False):
    # PREPARE DATA

    # N.B. That we add 1 here to indicate classes
    training_masks = np.array(training_masks) + 1

    if isinstance(validation_masks, np.ndarray):

        validation_masks = np.array(validation_masks) + 1

    # features_func = partial(
    #     feature.multiscale_basic_features,
    #     intensity=True,
    #     workers=4,
    #     edges=False,
    #     texture=False,
    #     sigma_min=sigma_min,
    #     sigma_max=sigma_max,
    #     channel_axis=0,
    # )
    #
    # # Extract local features from training data
    # training_maps = np.array([features_func(tm) for tm in training_maps])
    #
    # # Extract local features from validation data
    # validation_maps = np.array([features_func(vm) for vm in validation_maps])
    #
    # # Extract local features from test data
    # testing_maps = np.array([features_func(tm) for tm in testing_maps])

    if not pretrained:
        # Train a new RF segmentor

        # TRAIN RANDOM FOREST CLASSIFIER ON DATA

        if not tune:

            features_func = partial(
                feature.multiscale_basic_features,
                intensity=True,
                workers=4,
                edges=False,
                texture=False,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                channel_axis=0,
            )

            # Extract local features from training data
            training_maps = np.array([features_func(tm) for tm in training_maps])

            # Extract local features from validation data
            validation_maps = np.array([features_func(vm) for vm in validation_maps])

            # Extract local features from test data
            testing_maps = np.array([features_func(tm) for tm in testing_maps])

            # CHANGE THESE TO THE BEST FOUND DURING TUNING

            clf = RandomForestClassifier(n_estimators=60, n_jobs=4, max_depth=7, max_samples=0.05)

            clf = future.fit_segmenter(training_masks, training_maps, clf)

            # GET PREDICTIONS

            # SAVE TRAINED SEGMENTATION ALGORITHM
            with open(save_file, "wb") as f:
                dump(clf, f)

        else:

            # ONES I USED WHEN TUNING:

            # n_estimators = [30, 50, 60, 70, 90, 100, 150]
            #
            # max_depth = [5, 7, 9, 10, 15, 20]
            #
            # max_samples = [0.01, 0.025, 0.05]
            #
            # max_features = [None, "sqrt", "log2"]
            #
            # sms = [1]
            #
            # sxs = [4, 8, 16]



            n_estimators = [150]

            max_depth = [20]

            max_samples = [0.05]

            max_features = [None]

            sms = [1]

            sxs = [4]

            for n in n_estimators:

                for m in max_depth:

                    for ms in max_samples:

                        for mf in max_features:

                            for sm in sms:

                                for sx in sxs:

                                    start = time.time()

                                    features_func = partial(
                                        feature.multiscale_basic_features,
                                        intensity=True,
                                        workers=5,
                                        texture=True,
                                        edges = True,
                                        sigma_min=sm,
                                        sigma_max=sx,
                                        channel_axis=0,
                                    )

                                    # Extract local features from training data
                                    training_features = np.array([features_func(tm) for tm in training_maps])

                                    tmp = training_features[0]
                                    tmp_labels = training_masks[0]

                                    for k in range(1, training_maps.shape[0]):
                                        tmp = np.concatenate((tmp, training_features[k]), 0)
                                        tmp_labels = np.concatenate((tmp_labels, training_masks[k]), 0)

                                    training_features = tmp
                                    training_labels = tmp_labels

                                    # Extract local features from validation data
                                    validation_features = np.array([features_func(vm) for vm in validation_maps])

                                    tmp = validation_features[0]
                                    tmp_labels = validation_masks[0]

                                    for k in range(1, validation_maps.shape[0]):
                                        tmp = np.concatenate((tmp, validation_features[k]), 0)
                                        tmp_labels = np.concatenate((tmp_labels, validation_masks[k]), 0)

                                    validation_features = tmp

                                    # clf = RandomForestClassifier(n_estimators=n, max_depth=m, max_samples=ms, max_features=mf,
                                    #                              n_jobs=-1)



                                    # CHANGED OOB SCORE - better

                                    clf = RandomForestClassifier(n_estimators=n, max_depth=m, max_samples=ms, max_features=mf,
                                                                 n_jobs=-1, oob_score=balanced_accuracy_score)





                                    clf = future.fit_segmenter(training_labels, training_features, clf)

                                    # FIT TO VALIDATION MAPS

                                    predicted_tmp = future.predict_segmenter(validation_features, clf)

                                    predicted_tmp = predicted_tmp.reshape(validation_maps.shape[0], 64, 64)

                                    true_positives = np.sum(
                                        np.logical_and(validation_masks - 1 == 1, predicted_tmp - 1== 1))

                                    true_negatives = np.sum(
                                        np.logical_and(validation_masks - 1 == 0, predicted_tmp - 1== 0))

                                    false_positives = np.sum(
                                        np.logical_and(validation_masks -  1 == 0, predicted_tmp - 1== 1))

                                    false_negatives = np.sum(
                                        np.logical_and(validation_masks - 1 == 1, predicted_tmp - 1== 0))

                                    if (true_positives == 0 and false_negatives == 0) or (
                                            true_negatives == 0 and false_positives == 0):

                                        bba = 0

                                    else:

                                        first_term = true_positives / (true_positives + false_negatives)
                                        second_term = true_negatives / (true_negatives + false_positives)

                                        bba = (first_term + second_term) / 2

                                    print(f"{n}, {m}, {ms}, {mf}, {sm}, {sx}, {time.time() - start}, {bba}")







            # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            # means = grid_result.cv_results_['mean_test_score']
            # stds = grid_result.cv_results_['std_test_score']
            # params = grid_result.cv_results_['params']
            # for mean, stdev, param in zip(means, stds, params):
            #     print("%f (%f) with: %r" % (mean, stdev, param))

            # clf = RandomForestClassifier(**grid_search.best_params)
            #
            # clf = future.fit_segmenter(training_masks, training_maps, clf)
            #
            # # GET PREDICTIONS
            #
            # # SAVE TRAINED SEGMENTATION ALGORITHM
            # with open(save_file, "wb") as f:
            #     dump(clf, f)

    else:

        features_func = partial(
            feature.multiscale_basic_features,
            intensity=True,
            workers=4,
            edges=False,
            texture=False,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            channel_axis=0,
        )

        # Extract local features from training data
        training_maps = np.array([features_func(tm) for tm in training_maps])

        # Extract local features from validation data
        validation_maps = np.array([features_func(vm) for vm in validation_maps])

        # Extract local features from test data
        testing_maps = np.array([features_func(tm) for tm in testing_maps])

        # RETRIEVE RANDOM FOREST CLASSIFIER

        # Use pre-trained RF segmentor
        with open(save_file, "rb") as f:
            clf = load(f)

    # GET PREDICTIONS

    if real:
        test_data_predictions = future.predict_segmenter(testing_maps, clf)
        test_data_predictions = np.array(
            [[test_data_predictions[k]] for k in range(test_data_predictions.shape[0])]) - 1

        return np.array([]), np.array([]), test_data_predictions

    else:

        train_data_predictions = future.predict_segmenter(training_maps, clf)
        validation_data_predictions = future.predict_segmenter(validation_maps, clf)
        test_data_predictions = future.predict_segmenter(testing_maps, clf)

        # Minus 1 such that pixel = 0 indicates background and pixel = 1 indicates foreground
        train_data_predictions = np.array([[train_data_predictions[k]] for k in
                                           range(train_data_predictions.shape[0])]) - 1
        validation_data_predictions = np.array([[validation_data_predictions[k]] for k in
                                                range(validation_data_predictions.shape[0])]) - 1
        test_data_predictions = np.array([[test_data_predictions[k]] for k in
                                          range(test_data_predictions.shape[0])]) - 1

        return train_data_predictions, validation_data_predictions, test_data_predictions




def svm_segmentation(training_maps, training_masks, validation_maps, testing_maps, validation_masks=None, sigma_min=1, sigma_max=16,
                               pretrained=False, real=False,
                               save_file="./algorithms/pre_trained_models/random_forest_segmentation.pt", tune=False):
    # PREPARE DATA

    # N.B. That we add 1 here to indicate classes
    training_masks = np.array(training_masks) + 1

    if isinstance(validation_masks, np.ndarray):

        validation_masks = np.array(validation_masks) + 1

    # features_func = partial(
    #     feature.multiscale_basic_features,
    #     intensity=True,
    #     workers=4,
    #     edges=False,
    #     texture=False,
    #     sigma_min=sigma_min,
    #     sigma_max=sigma_max,
    #     channel_axis=0,
    # )
    #
    # # Extract local features from training data
    # training_maps = np.array([features_func(tm) for tm in training_maps])
    #
    # # Extract local features from validation data
    # validation_maps = np.array([features_func(vm) for vm in validation_maps])
    #
    # # Extract local features from test data
    # testing_maps = np.array([features_func(tm) for tm in testing_maps])

    if not pretrained:
        # Train a new RF segmentor

        # TRAIN RANDOM FOREST CLASSIFIER ON DATA

        if not tune:

            features_func = partial(
                feature.multiscale_basic_features,
                intensity=True,
                workers=4,
                edges=False,
                texture=False,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                channel_axis=0,
            )

            # Extract local features from training data
            training_maps = np.array([features_func(tm) for tm in training_maps])

            # Extract local features from validation data
            validation_maps = np.array([features_func(vm) for vm in validation_maps])

            # Extract local features from test data
            testing_maps = np.array([features_func(tm) for tm in testing_maps])

            # CHANGE THESE TO THE BEST FOUND DURING TUNING
            #
            # clf = RandomForestClassifier(n_estimators=60, n_jobs=4, max_depth=7, max_samples=0.05)

            clf = future.fit_segmenter(training_masks, training_maps, clf)

            # GET PREDICTIONS

            # SAVE TRAINED SEGMENTATION ALGORITHM
            with open(save_file, "wb") as f:
                dump(clf, f)

        else:

            # ONES I USED WHEN TUNING:

            # n_estimators = [30, 50, 60, 70, 90, 100, 150]
            #
            # max_depth = [5, 7, 9, 10, 15, 20]
            #
            # max_samples = [0.01, 0.025, 0.05]
            #
            # max_features = [None, "sqrt", "log2"]
            #
            # sms = [1]
            #
            # sxs = [4, 8, 16]



            n_estimators = [150]

            max_depth = [20]

            max_samples = [0.05]

            max_features = [None]

            sms = [1]

            sxs = [4]

            for n in n_estimators:

                for m in max_depth:

                    for ms in max_samples:

                        for mf in max_features:

                            for sm in sms:

                                for sx in sxs:

                                    start = time.time()

                                    features_func = partial(
                                        feature.multiscale_basic_features,
                                        intensity=True,
                                        workers=5,
                                        texture=True,
                                        edges = True,
                                        sigma_min=sm,
                                        sigma_max=sx,
                                        channel_axis=0,
                                    )

                                    # Extract local features from training data
                                    training_features = np.array([features_func(tm) for tm in training_maps])

                                    tmp = training_features[0]
                                    tmp_labels = training_masks[0]

                                    for k in range(1, training_maps.shape[0]):
                                        tmp = np.concatenate((tmp, training_features[k]), 0)
                                        tmp_labels = np.concatenate((tmp_labels, training_masks[k]), 0)

                                    training_features = tmp
                                    training_labels = tmp_labels

                                    # Extract local features from validation data
                                    validation_features = np.array([features_func(vm) for vm in validation_maps])

                                    tmp = validation_features[0]
                                    tmp_labels = validation_masks[0]

                                    for k in range(1, validation_maps.shape[0]):
                                        tmp = np.concatenate((tmp, validation_features[k]), 0)
                                        tmp_labels = np.concatenate((tmp_labels, validation_masks[k]), 0)

                                    validation_features = tmp

                                    # clf = RandomForestClassifier(n_estimators=n, max_depth=m, max_samples=ms, max_features=mf,
                                    #                              n_jobs=-1)



                                    # CHANGED OOB SCORE - better

                                    # clf = RandomForestClassifier(n_estimators=n, max_depth=m, max_samples=ms, max_features=mf,
                                    #                              n_jobs=-1, oob_score=balanced_accuracy_score, criterion='entropy')
                                    #

                                    clf = SVC()




                                    clf = future.fit_segmenter(training_labels, training_features, clf)

                                    # FIT TO VALIDATION MAPS

                                    predicted_tmp = future.predict_segmenter(validation_features, clf)

                                    predicted_tmp = predicted_tmp.reshape(validation_maps.shape[0], 64, 64)

                                    true_positives = np.sum(
                                        np.logical_and(validation_masks - 1 == 1, predicted_tmp - 1== 1))

                                    true_negatives = np.sum(
                                        np.logical_and(validation_masks - 1 == 0, predicted_tmp - 1== 0))

                                    false_positives = np.sum(
                                        np.logical_and(validation_masks -  1 == 0, predicted_tmp - 1== 1))

                                    false_negatives = np.sum(
                                        np.logical_and(validation_masks - 1 == 1, predicted_tmp - 1== 0))

                                    if (true_positives == 0 and false_negatives == 0) or (
                                            true_negatives == 0 and false_positives == 0):

                                        bba = 0

                                    else:

                                        first_term = true_positives / (true_positives + false_negatives)
                                        second_term = true_negatives / (true_negatives + false_positives)

                                        bba = (first_term + second_term) / 2

                                    print(f"{n}, {m}, {ms}, {mf}, {sm}, {sx}, {time.time() - start}, {bba}")







            # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            # means = grid_result.cv_results_['mean_test_score']
            # stds = grid_result.cv_results_['std_test_score']
            # params = grid_result.cv_results_['params']
            # for mean, stdev, param in zip(means, stds, params):
            #     print("%f (%f) with: %r" % (mean, stdev, param))

            # clf = RandomForestClassifier(**grid_search.best_params)
            #
            # clf = future.fit_segmenter(training_masks, training_maps, clf)
            #
            # # GET PREDICTIONS
            #
            # # SAVE TRAINED SEGMENTATION ALGORITHM
            # with open(save_file, "wb") as f:
            #     dump(clf, f)

    else:

        features_func = partial(
            feature.multiscale_basic_features,
            intensity=True,
            workers=4,
            edges=False,
            texture=False,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            channel_axis=0,
        )

        # Extract local features from training data
        training_maps = np.array([features_func(tm) for tm in training_maps])

        # Extract local features from validation data
        validation_maps = np.array([features_func(vm) for vm in validation_maps])

        # Extract local features from test data
        testing_maps = np.array([features_func(tm) for tm in testing_maps])

        # RETRIEVE RANDOM FOREST CLASSIFIER

        # Use pre-trained RF segmentor
        with open(save_file, "rb") as f:
            clf = load(f)

    # GET PREDICTIONS

    if real:
        test_data_predictions = future.predict_segmenter(testing_maps, clf)
        test_data_predictions = np.array(
            [[test_data_predictions[k]] for k in range(test_data_predictions.shape[0])]) - 1

        return np.array([]), np.array([]), test_data_predictions

    else:

        train_data_predictions = future.predict_segmenter(training_maps, clf)
        validation_data_predictions = future.predict_segmenter(validation_maps, clf)
        test_data_predictions = future.predict_segmenter(testing_maps, clf)

        # Minus 1 such that pixel = 0 indicates background and pixel = 1 indicates foreground
        train_data_predictions = np.array([[train_data_predictions[k]] for k in
                                           range(train_data_predictions.shape[0])]) - 1
        validation_data_predictions = np.array([[validation_data_predictions[k]] for k in
                                                range(validation_data_predictions.shape[0])]) - 1
        test_data_predictions = np.array([[test_data_predictions[k]] for k in
                                          range(test_data_predictions.shape[0])]) - 1

        return train_data_predictions, validation_data_predictions, test_data_predictions



# REFERENCES

# Debugging - https://stackoverflow.com/questions/58925808/python-indexerror-boolean-index-did-not-match-indexed-array-
# along-dimension-0
# Grid Search Multiclass - https://stackoverflow.com/questions/26018543/gridsearch-for-multi-label-classification-in-
# scikit-learn
# Improving RF Classifier - https://www.reddit.com/r/learnmachinelearning/comments/10xhff9/significant_ways_to_improve_
# model_accuracy/
# Improving RF Classifier - https://stackoverflow.com/questions/53634808/how-to-improve-performance-of-random-forest-
# multiclass-classification-model
# Random Forest Classifier - https://stackoverflow.com/questions/43640546/how-to-make-randomforestclassifier-faster
# Random Forest Memory - https://stackoverflow.com/questions/30766253/scikit-learn-random-forest-taking-up-too-much-
# memory
# Reshape Array - https://stackoverflow.com/questions/7372316/how-can-i-make-a-two-dimensional-numpy-array-a-three-
# dimensional-array
# Reshape Array - https://stackoverflow.com/questions/52670278/how-to-reshape-a-numpy-array-from-x-y-z-to-y-z-x
# Tutorial Code - https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_trainable_segmentation.html#id4
