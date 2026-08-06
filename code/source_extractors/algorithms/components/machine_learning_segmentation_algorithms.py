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


def random_forest_segmentation(training_maps, training_masks, validation_maps, testing_maps, sigma_min=1, sigma_max=16,
                               pretrained=False, real=False,
                               save_file="./algorithms/pre_trained_models/random_forest_segmentation.pt"):
    # PREPARE DATA

    # N.B. That we add 1 here to indicate classes
    training_masks = np.array(training_masks) + 1

    features_func = partial(
        feature.multiscale_basic_features,
        intensity=True,
        edges=False,
        texture=True,
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

    if not pretrained:
        # Train a new RF segmentor

        # TRAIN RANDOM FOREST CLASSIFIER ON DATA

        clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=10, max_samples=0.05)

        clf = future.fit_segmenter(training_masks, training_maps, clf)

        # GET PREDICTIONS

        # train_data_predictions = future.predict_segmenter(training_maps, clf)
        # test_data_predictions = future.predict_segmenter(testing_maps, clf)

        # SAVE TRAINED SEGMENTATION ALGORITHM
        with open(save_file, "wb") as f:
            dump(clf, f, protocol=5)

    else:

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
# Tutorial Code - https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_trainable_segmentation.html#id4
