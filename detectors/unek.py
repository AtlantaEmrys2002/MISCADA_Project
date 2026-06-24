import numpy as np
from benchmarks.unek import unek_algorithm
from benchmarks.components.clustering_algorithms import k_means_clustering

#TODO
# Use this answer https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
# to do train-test-validation split of data. Validation used to calculate loss in U-Net training

import torch
from torch.utils.data import DataLoader, Subset

# for k in test_images[0]:
#     tmp = plt.imshow(k)
#     plt.show()
#
# tmp = plt.imshow(test_segments[0], cmap='gray')
# plt.show()

# tmp = plt.imshow(test_masks[0], cmap='gray')
# plt.show()

# N.B. Refamiliarised myself and started with tutorial https://medium.com/@alessandromondin/semantic-segmentation-with-
# pytorch-u-net-from-scratch-502d6565910a (referenced below) and built on top of that.
# Also consulted author's GitHub implementation - https://github.com/AlessandroMondin/U-NET

# Adjusted below implementation such that it was compatible with the dataset I have created (and the one created by ID8)



# CREATE RANDOM TEST DATA

# No. samples to generate
n = 256

# 64 x 64 image for each of the energy bins which will be 6 here

# GENERATING 100 RANDOM IMAGES EACH WITH 5 ENERGY BINS

test_images = torch.rand((n, 5, 64, 64))

# RANDOM BINARY SEGMENTATIONS OF THE OUTPUT (WHICH IS THE IMAGE SEGMENTED AS BACKGROUND OR FOREGROUND - SOURCED
# REMEMBER - WILL NOT BE BINNED - SO JUST NEED 100 64 x 64 IMAGES
# test_segments = np.random.choice(a=np.array([0.0, 1.0]), size=(n, 1, 64, 64))

# test_segments = torch.rand((n, 1, 64, 64))
# test_segments = (test_segments > 0.5).float()

test_segments = []
segment_centres = []

for x in range(n):

    # Create tmp mask
    tmp_msk = np.zeros(shape=(64, 64))

    tmp_centres = []

    # Choose to simulate between 0 and 5 point source (circles)
    for x in range(np.random.choice(5)):

        # Assume sources with centres too close to the edge would be discounted
        centre_coord = np.random.choice(a=np.arange(5, 59), size=2)

        tmp_centres.append(centre_coord)

        for k in range(64):
            for j in range(64):
                dist = np.sqrt(((centre_coord[0] - k) ** 2) + ((centre_coord[1] - j) ** 2))
                if dist <= 2.5:
                    # 2.5 from paper
                    tmp_msk[k, j] = 1

    # Add some random noise - reflects what U-Net will likely predict
    noise = np.random.choice(a=[0, 1], size=tmp_msk.shape, p=[0.9, 0.1])
    tmp_msk = np.maximum(tmp_msk, noise)

    tmp_msk = [tmp_msk]
    segment_centres.append(tmp_centres)

    test_segments.append(tmp_msk)

test_segments = np.array(test_segments)

test_segments = torch.from_numpy(test_segments)

#
# tmp = plt.imshow(test_segments[2][0], cmap='gray')
# plt.show()



# Random 0 and 1s torch tensor https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146/14

# a = torch.tensor([0.0, 1.0])
# p = torch.tensor([0.5, 0.5])
# replace = True
#
# for k in range(n):
#
#     idx = p.multinomial(num_samples=n, replacement=replace)
#     test_segments = a[idx]






# RANDOM POSITIONS (CIRCLES) - OUTPUT OF k-MEANS
test_masks = []
centres = []
for x in range(10):

    # Assume sources with centres too close to the edge would be discounted
    centre_coord = np.random.choice(a=np.arange(5, 59), size=2)

    centres.append(centre_coord)

    # Create tmp mask
    tmp_msk = np.zeros(shape=(64, 64))

    for k in range(64):
        for j in range(64):
            dist = np.sqrt(((centre_coord[0] - k) ** 2) + ((centre_coord[1] - j) ** 2))
            if dist <= 2.5:
                # 2.5 from paper
                tmp_msk[k, j] = 1

    test_masks.append(tmp_msk)

test_masks = np.array(test_masks)
centres = np.array(centres)

# RANDOM SPLIT OF INDICES AT THE MOMENT - 70% vs 20% v 10% split
train_indices = np.random.choice(256, size=179, replace=False)
validation_indices = np.random.choice(np.array([k for k in range(256) if k not in train_indices]), size=51, replace=False)
test_indices = np.array([k for k in range(256) if (k not in train_indices) and (k not in validation_indices)])

# test_indices = np.array([k for k in range(256) if k not in train_indices])

# Stack test images and test segmented images

test_data = [(test_images[k], test_segments[k]) for k in range(n)]

# Split data into test and train (EVENTUALLY USE SCIKIT LEARN TO DO SO)
train_split = Subset(test_data, train_indices)
validation_split = Subset(test_data, validation_indices)
test_split = Subset(test_data, test_indices)

# Create batches
train_batches = DataLoader(train_split, batch_size=128, shuffle=True)
validation_batches = DataLoader(validation_split, batch_size=128, shuffle=True)
test_batches = DataLoader(test_split, batch_size=128)

# Binary segmentation of image

# proposed_sources = k_means_clustering(test_segments)

print(unek_algorithm(train_batches, validation_batches, test_batches))


# Train U-Net
# trained_model = unet_train(train_batches, test_batches)

# Test outputs
# unet_outputs = trained_model(test_batches)

# Create sources and their centres
# k_means_clustering(unet_outputs)

# proposed_sources = k_means_clustering(test_segments)




# test = test_images
# unet = UNET(5, 16, 1, padding=1, downhill=4)
# output = unet(test).detach().numpy()
# print("out unet shape is", output.shape)
# plt.imshow(output[0][0])
# plt.show()






# REFERENCES
# ID8 - followed their theory/mathematical definition to implement my own version
# Indexing with array of indices - https://stackoverflow.com/questions/19821425/how-can-i-filter-numpy-array-by-list-of-
# indices
# Pytorch Documentation - https://pytorch.org/get-started/locally/
# Tutorial - https://medium.com/@alessandromondin/semantic-segmentation-with-pytorch-u-net-from-scratch-502d6565910a
# Tutorial GitHub - https://github.com/AlessandroMondin/U-NET/blob/main/dataset.py
