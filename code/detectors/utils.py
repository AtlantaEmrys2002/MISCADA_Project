import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def random_data(n=256):

    # Creates random test data to feed to neural networks that is the same shape and type as telescope data to
    # be passed. n is the number of random samples to generate.

    # Binned count maps to be passed to segmentation algorithms - note that there are 5 energy bins
    test_images = torch.rand((n, 5, 64, 64))

    # Segmented images (with radii around random points)
    test_segments = []

    for x in range(n):

        # Create tmp mask
        tmp_msk = np.zeros(shape=(64, 64))

        # Choose to simulate between 0 and 5 point source (circles)
        for x in range(np.random.choice(5)):

            # Assume sources with centres too close to the edge would be discounted
            centre_coord = np.random.choice(a=np.arange(5, 59), size=2)

            for k in range(64):
                for j in range(64):
                    dist = np.sqrt(((centre_coord[0] - k) ** 2) + ((centre_coord[1] - j) ** 2))
                    if dist <= 2.5:
                        # 2.5 from paper
                        tmp_msk[k, j] = 1

        # Add some random noise - reflects what U-Net will likely predict
        noise = np.random.choice(a=[0, 1], size=tmp_msk.shape, p=[0.99, 0.01])
        tmp_msk = np.maximum(tmp_msk, noise)

        tmp_msk = [tmp_msk]

        test_segments.append(tmp_msk)

    test_segments = np.array(test_segments)

    test_segments = torch.from_numpy(test_segments)

    # Combine to create test data
    test_data = [(test_images[k], test_segments[k]) for k in range(n)]

    # RANDOM SPLIT OF INDICES AT THE MOMENT - 70% vs 20% v 10% split
    train_indices = np.random.choice(256, size=179, replace=False)
    validation_indices = np.random.choice(np.array([k for k in range(256) if k not in train_indices]), size=51,
                                          replace=False)
    test_indices = np.array([k for k in range(256) if (k not in train_indices) and (k not in validation_indices)])

    # Split data into test and train (EVENTUALLY USE SCIKIT LEARN TO DO SO)
    train_split = Subset(test_data, train_indices)
    validation_split = Subset(test_data, validation_indices)
    test_split = Subset(test_data, test_indices)

    # Create batches
    train_batches = DataLoader(train_split, batch_size=128, shuffle=True)
    validation_batches = DataLoader(validation_split, batch_size=128, shuffle=True)
    test_batches = DataLoader(test_split, batch_size=128)

    return train_batches, validation_batches, test_batches


# REFERENCES

# Pytorch Random Choice - https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146/14
# Torch Random Numbers - https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146/6
# Train/Test/Validation Split - https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-
# training-and-test-datasets
