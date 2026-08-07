"""
The code for extract_hog_features() was adapted from this tutorial: https://www.geeksforgeeks.org/machine-learning
/random-forest-for-image-classification-using-opencv/
"""

import numpy as np
from pickle import dump, load
from sklearn.ensemble import RandomForestClassifier
# from skimage.feature import hog


# def extract_hog_features(image):
#     hog_features = hog(
#         image,
#         orientations=9,
#         pixels_per_cell=(8, 8),
#         cells_per_block=(2, 2),
#         channel_axis=0,
#     )
#     return hog_features


def random_forest_classifier(train_data, test_data, pretrained=False,
                             save_file="./algorithms/pre_trained_models/random_forest_classifier.pt"):
    if not pretrained:
        # EXTRACT TRAINING DATA

        # train_map_patches = [extract_hog_features(k[0]) for k in train_data]

        # TRYING WITHOUT HOG - HOG FAILS

        train_map_patches = np.array([k[0].flatten() for k in train_data])
        train_labels = np.array([k[1] for k in train_data])

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

    return actual_labels, test_predictions
