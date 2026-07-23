import copy
import math
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset


def read_patches(num_patches=int, directory=str):

    # Read in patches
    patches = (np.array([np.load("{}/patch_{}/patch.npy".format(directory, n)) for n in range(num_patches)])
               .astype(np.float32))

    # Read in masks
    masks = (np.array([[np.load("{}/patch_{}/mask.npy".format(directory, n))] for n in range(num_patches)])
             .astype(np.float32))

    # Read in metadata for each patch

    source_ids = []
    actual_cartesian_locations = []
    types = []
    patch_ids = list(range(1, int(num_patches) + 1))

    for n in range(num_patches):

        file_name = "{}/patch_{}/metadata.csv".format(directory, n)

        df = pd.read_csv(file_name)

        source_ids.append(df["source_id"].to_numpy())

        ys = df["cartesian_y"].to_numpy()
        xs = df["cartesian_x"].to_numpy()

        cartesian_coords = np.array(list(zip(ys, xs)))

        actual_cartesian_locations.append(cartesian_coords)

        types.append(df["source_type"].to_numpy())

    # metadata = [patch_ids, source_ids, actual_cartesian_locations, types]

    # metadata = [[patch_ids[k], source_ids[k], actual_cartesian_locations[k], types[k]] for k in range(num_patches)]

    # Combine to create test data
    test_data = [(torch.from_numpy(patches[k]), torch.from_numpy(masks[k])) for k in range(num_patches)]

    # Select random samples
    train_set_size = math.floor(float(num_patches) * 0.7)
    validation_set_size = math.floor(float(num_patches) * 0.2)
    test_set_size = int(num_patches) - train_set_size - validation_set_size

    # RANDOM SPLIT OF INDICES - 70% vs 20% v 10%

    possible_indices = np.arange(num_patches)

    train_indices = np.random.choice(possible_indices, size=train_set_size, replace=False)

    validation_indices = np.random.choice(np.setdiff1d(possible_indices, train_indices), size=validation_set_size,
                                          replace=False)

    test_indices = np.random.choice(np.setdiff1d(possible_indices, np.concatenate((train_indices, validation_indices))),
                                    size=test_set_size, replace=False)

    # Split data into train, validation, and test sets
    train_split = Subset(test_data, train_indices)
    validation_split = Subset(test_data, validation_indices)
    test_split = Subset(test_data, test_indices)

    # Create batches
    train_batches = DataLoader(train_split, batch_size=128, shuffle=True)
    validation_batches = DataLoader(validation_split, batch_size=128, shuffle=True)
    test_batches = DataLoader(test_split, batch_size=128, shuffle=False)

    # N.B. only need metadata for testing data
    # test_metadata = [metadata[k] for k in test_indices]

    # N.B. Only need the IDs of each test patch and make sure not to shuffle the test patches

    test_patch_ids = copy.deepcopy(test_indices)  # + 1

    return train_batches, validation_batches, test_batches, test_patch_ids  # test_metadata


def save_predictions(patch_ids, predicted_segmentations, predicted_locations, directory: str, method: str):

    # Create directory to store results

    save_location = directory + "/{}/".format(method)

    Path(save_location).mkdir(parents=True, exist_ok=True)

    patches_location = "./../data_simulation/simulated_data/patches/"

    # SAVE ACTUAL AND PREDICTED SEGMENTATIONS OF EACH PATCH

    actual_segmentations = [np.load(patches_location + "patch_{}/mask.npy".format(p)) for p in patch_ids]

    predicted_segmentations = [predicted_segmentations[p][0].astype(np.float64) for p in range(len(patch_ids))]

    segmentations = np.array(list(zip(actual_segmentations, predicted_segmentations)))

    np.save(save_location + "segmentations.npy", segmentations)

    # SAVE ACTUAL AND PREDICTED LOCATIONS OF EACH PATCH

    np.save(save_location + "predicted_locations.npy", predicted_locations)

    dfs = [pd.read_csv(patches_location + "patch_{}/metadata.csv".format(p)) for p in patch_ids]

    print(dfs[0])
    print(predicted_locations)




# REFERENCES

# Order with Test Data - https://discuss.pytorch.org/t/maintaining-order-of-data-while-using-dataloader/32297/2

