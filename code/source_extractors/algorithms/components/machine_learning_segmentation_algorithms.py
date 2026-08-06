from functools import partial
import math
import numpy as np
import pandas as pd
from skimage import feature, future
from sklearn.ensemble import RandomForestClassifier
import torch
from torch.utils.data import DataLoader, Subset

# def read_patches(num_patches=int, directory=str):
#     # Read in patches
#     patches = (np.array([np.load("{}/patch_{}/patch.npy".format(directory, n)) for n in range(num_patches)])
#                .astype(np.float32))
#
#     # Read in masks
#     masks = (np.array([[np.load("{}/patch_{}/mask.npy".format(directory, n))] for n in range(num_patches)])
#              .astype(np.float32))
#
#     # Read in metadata for each patch
#
#     source_ids = []
#     actual_cartesian_locations = []
#     types = []
#
#     for n in range(num_patches):
#         file_name = "{}/patch_{}/metadata.csv".format(directory, n)
#
#         df = pd.read_csv(file_name)
#
#         source_ids.append(df["source_id"].to_numpy())
#
#         ys = df["cartesian_y"].to_numpy()
#         xs = df["cartesian_x"].to_numpy()
#
#         cartesian_coords = np.array(list(zip(ys, xs)))
#
#         actual_cartesian_locations.append(cartesian_coords)
#
#         types.append(df["source_type"].to_numpy())
#
#     # Combine to create test data
#     test_data = [(k, torch.from_numpy(patches[k]), torch.from_numpy(masks[k])) for k in range(num_patches)]
#
#     # Select random samples
#     train_set_size = math.floor(float(num_patches) * 0.7)
#     validation_set_size = math.floor(float(num_patches) * 0.2)
#     test_set_size = int(num_patches) - train_set_size - validation_set_size
#
#     # RANDOM SPLIT OF INDICES - 70% vs 20% v 10%
#
#     possible_indices = np.arange(num_patches)
#
#     train_indices = np.random.choice(possible_indices, size=train_set_size, replace=False)
#
#     validation_indices = np.random.choice(np.setdiff1d(possible_indices, train_indices), size=validation_set_size,
#                                           replace=False)
#
#     test_indices = np.random.choice(np.setdiff1d(possible_indices, np.concatenate((train_indices, validation_indices))),
#                                     size=test_set_size, replace=False)
#
#     # Split data into train, validation, and test sets
#     train_split = Subset(test_data, train_indices)
#     validation_split = Subset(test_data, validation_indices)
#
#     # Create batches
#     train_batches = DataLoader(train_split, batch_size=128, shuffle=True)
#     validation_batches = DataLoader(validation_split, batch_size=128, shuffle=False)
#
#     # N.B. Only need the IDs of each test patch and make sure not to shuffle the test patches
#
#     data_for_testing = [test_data[k] for k in test_indices]
#
#     return train_batches, validation_batches, data_for_testing



def random_forest_segmentation(training_data, testing_data, sigma_min=1, sigma_max=16):
    # EXTRACT TRAINING DATA
    train_patch_ids = []
    training_maps = []
    training_masks = []

    for i, x in enumerate(training_data):

        train_patch_ids.append(x[0])

        for k in range(x[1].shape[0]):
            training_maps.append(x[1][k])

        for j in range(x[2].shape[0]):
            training_masks.append(x[2][j][0])

    training_maps = np.array(training_maps)

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

    # Extract local features from data
    training_maps = np.array([features_func(tm) for tm in training_maps])

    # TRAIN RANDOM FOREST CLASSIFIER ON DATA

    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=10, max_samples=0.05)

    clf = future.fit_segmenter(training_masks, training_maps, clf)

    # EXTRACT TEST DATA

    # test_patch_ids = np.array([k[0] for k in testing_data])
    testing_maps = np.array([k[1] for k in testing_data])
    # testing_masks = np.array([k[2][0] for k in testing_data])

    # Extract local features from test data
    testing_maps = np.array([features_func(tm) for tm in testing_maps])

    test_data_predictions = future.predict_segmenter(testing_maps, clf)

    test_data_predictions = np.array([[test_data_predictions[k]] for k in range(test_data_predictions.shape[0])])

    return test_data_predictions

# REFERENCES

# Debugging - https://stackoverflow.com/questions/58925808/python-indexerror-boolean-index-did-not-match-indexed-array-
# along-dimension-0

