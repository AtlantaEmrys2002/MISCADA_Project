"""
The code for extract_hog_features() was adapted from this tutorial: https://www.geeksforgeeks.org/machine-learning
/random-forest-for-image-classification-using-opencv/. Followed this tutorial for first version of SVM classifier -
https://www.geeksforgeeks.org/machine-learning/image-classification-using-support-vector-machine-svm-in-python/#google_vignette
"""

import numpy as np
from pickle import dump, load
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


def random_forest_classifier(train_data, test_data, pretrained=False,
                             save_file="./algorithms/pre_trained_models/random_forest_classifier.pt"):
    if not pretrained:
        # EXTRACT TRAINING DATA

        train_map_patches = np.array([k[0].flatten() for k in train_data])
        # train_labels = np.array([k[1] for k in train_data])

        train_labels = []

        for k in train_data:

            if np.all(np.equal(k[1], np.array([1., 0., 0.]))):
                train_labels.append("AGN")
            elif np.all(np.equal(k[1], np.array([0., 1., 0.]))):
                train_labels.append("PSR")
            else:
                train_labels.append("FAKE")

        train_labels = np.array(train_labels)

        clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=10, max_samples=0.05)

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
    # test_map_patches = [extract_hog_features(k[0]) for k in test_data]
    test_map_patches = np.array([k[0].flatten() for k in test_data])
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

# SVM - https://apmonitor.com/pds/index.php/Main/SupportVectorClassifier
# SVM - https://www.geeksforgeeks.org/machine-learning/image-classification-using-support-vector-machine-svm-in-python/
