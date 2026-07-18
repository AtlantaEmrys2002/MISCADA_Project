import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def read_patches(num_patches=int, directory=str):

    # Read in patches
    patches = (np.array([np.load("{}/patch_{}/patch.npy".format(directory, n)) for n in range(num_patches)])
               .astype(np.float32))

    # Read in masks
    masks = (np.array([[np.load("{}/patch_{}/mask.npy".format(directory, n))] for n in range(num_patches)])
             .astype(np.float32))

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
    test_batches = DataLoader(test_split, batch_size=128)

    return train_batches, validation_batches, test_batches


