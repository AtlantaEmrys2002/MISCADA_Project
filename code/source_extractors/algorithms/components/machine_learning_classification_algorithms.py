"""
The code for extract_hog_features() was adapted from this tutorial: https://www.geeksforgeeks.org/machine-learning
/random-forest-for-image-classification-using-opencv/. Followed this tutorial for first version of SVM classifier -
https://www.geeksforgeeks.org/machine-learning/image-classification-using-support-vector-machine-svm-in-python/#google_vignette
"""

import numpy as np
from pickle import dump, load
from skimage.feature import hog
from skimage.transform import rescale
from sklearn import svm
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier


# BELOW IS FROM TUTORIAL
def extract_hog_features(image):
    """
    This function was taken and adapted from here - https://www.geeksforgeeks.org/machine-learning/random-forest-for-
    image-classification-using-opencv/
    Parameters
    ----------
    image

    Returns
    -------

    """

    hog_features = hog(
        image,
        orientations=5,
        pixels_per_cell=(3, 3),
        cells_per_block=(2, 2),
        visualize=False,
        channel_axis=0
    )
    return hog_features


def random_forest_classifier(train_data, test_data, pretrained=False, tune=False, is_pspnet=False,
                             save_file="./algorithms/pre_trained_models/random_forest_classifier.pt"):
    """Consulted this heavily when tuning
     https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/

    Parameters
    ----------
    train_data
    test_data
    pretrained
    tune
    save_file

    Returns
    -------

    """

    if not pretrained:
        # EXTRACT TRAINING DATA

        train_map_patches = np.array([extract_hog_features(rescale(k[0], scale=10, channel_axis=0)) for k in train_data])

        train_labels = []

        for k in train_data:

            if np.all(np.equal(k[1], np.array([1., 0., 0.]))):
                train_labels.append("AGN")
            elif np.all(np.equal(k[1], np.array([0., 1., 0.]))):
                train_labels.append("PSR")
            else:
                train_labels.append("FAKE")

        train_labels = np.array(train_labels)

        # just use specified parameters
        if not tune:

            # For all three segmentation approaches, number of estimators was 200, class weight was None,
            # and max features was sqrt

            if is_pspnet:
                clf = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight=None, max_features='sqrt')
            else:
                clf = RandomForestClassifier(n_estimators=200, class_weight=None, max_features='sqrt')

            clf.fit(train_map_patches, train_labels)

            # SAVE TRAINED CLASSIFIER
            with open(save_file, "wb") as f:
                dump(clf, f, protocol=5)

        # find optimal parameters in terms of BALANCED accuracy
        else:

            clf = RandomForestClassifier()

            n_estimators = [30, 50, 70, 90, 200]

            max_depth = [5, 10, 15, 20]

            class_weight = [None, {"AGN" : 0.1, "PSR": 1, "FAKE":0.5}]

            max_features = [None, "sqrt", "log2"]

            grid = dict(n_estimators=n_estimators, max_depth=max_depth,
                        class_weight=class_weight, max_features=max_features)

            cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=3, random_state=1)

            grid_search = GridSearchCV(estimator=clf, param_grid=grid, n_jobs=-1, cv=cv, scoring="balanced_accuracy")

            grid_result = grid_search.fit(train_map_patches, train_labels)

            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))

            # Use classifier with best params found
            clf = RandomForestClassifier(**grid_search.best_params_)
            clf.fit(train_map_patches, train_labels)

            # SAVE TRAINED SEGMENTATION ALGORITHM
            with open(save_file, "wb") as f:
                dump(clf, f, protocol=5)

    else:

        # RETRIEVE RANDOM FOREST CLASSIFIER

        # Use pre-trained RF classifier
        with open(save_file, "rb") as f:
            clf = load(f)

    # EXTRACT TEST DATA
    test_map_patches = np.array([extract_hog_features(rescale(k[0], scale=10, channel_axis=0)) for k in test_data])

    actual_labels = np.array([k[1] for k in test_data])

    # CLASSIFY TEST DATA
    test_predictions = clf.predict(test_map_patches)

    vector_labels = {"AGN": np.array([1., 0., 0.]), "PSR": np.array([0., 1., 0.]), "FAKE": np.array([0., 0., 1.])}

    test_predictions = np.array([vector_labels[k] for k in test_predictions])

    return actual_labels, test_predictions


def svm_classifier(train_data, test_data, pretrained=False,
                   save_file="./algorithms/pre_trained_models/svm_classifier.pt"):

    if not pretrained:
        # EXTRACT TRAINING DATA

        # TRYING WITHOUT HOG - HOG FAILS

        train_map_patches = np.array([k[0].flatten() for k in train_data])

        train_labels = []

        for k in train_data:

            if np.all(np.equal(k[1], np.array([1., 0., 0.]))):
                train_labels.append("AGN")
            elif np.all(np.equal(k[1], np.array([0., 1., 0.]))):
                train_labels.append("PSR")
            else:
                train_labels.append("FAKE")

        train_labels = np.array(train_labels)

        support_vector_machine = svm.SVC(gamma=0.001)

        support_vector_machine.fit(train_map_patches, train_labels)

        # SAVE TRAINED SEGMENTATION ALGORITHM
        with open(save_file, "wb") as f:
            dump(support_vector_machine, f, protocol=5)

    else:

        # RETRIEVE RANDOM FOREST CLASSIFIER

        # Use pre-trained RF classifier
        with open(save_file, "rb") as f:
            support_vector_machine = load(f)

    # EXTRACT TEST DATA

    test_map_patches = np.array([k[0].flatten() for k in test_data])
    actual_labels = np.array([k[1] for k in test_data])

    # CLASSIFY TEST DATA
    test_predictions = support_vector_machine.predict(test_map_patches)

    vector_labels = {"AGN": np.array([1., 0., 0.]), "PSR": np.array([0., 1., 0.]), "FAKE": np.array([0., 0., 1.])}

    test_predictions = np.array([vector_labels[k] for k in test_predictions])

    return actual_labels, test_predictions

# REFERENCES

# Adaboost - https://medium.com/@chaudhurysrijani/tuning-of-adaboost-with-computational-complexity-8727d01a9d20
# Histogram of Gradients (HoG) - https://scikit-image.org/docs/0.25.x/auto_examples/features_detection/plot_hog.html
# RF Image Classification - https://machinelearningmastery.com/random-forest-for-image-classification-using-opencv/
# RF Image Classification - https://www.geeksforgeeks.org/machine-learning/random-forest-for-image-classification-using-
# opencv/
# SVM - https://apmonitor.com/pds/index.php/Main/SupportVectorClassifier
# SVM - https://www.geeksforgeeks.org/machine-learning/image-classification-using-support-vector-machine-svm-in-python/
# Working with HoG - https://forum.image.sc/t/histogram-equalization-before-extracting-hog-features/30409/3
