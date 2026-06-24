import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

#TODO
# Use this answer https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
# to do train-test-validation split of data. Validation used to calculate loss in U-Net training

import torch
from torch import nn
from torchvision.transforms.functional import center_crop
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


class CNNBlock(nn.Module):

    # Changed here, as padding must = 1 to get the correct shapes of outputs indicated in ID8
    def __init__(self, in_chan, out_chan, kernel_size=3, stride=1, padding=1):

        super(CNNBlock, self).__init__()

        self.seq_block = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x = self.seq_block(x)

        return x


class CNNBlocks(nn.Module):

    def __init__(self, n_conv, in_chan, out_chan, padding):

        super(CNNBlocks, self).__init__()

        self.layers = nn.ModuleList()

        for i in range(n_conv):

            self.layers.append(CNNBlock(in_chan, out_chan, padding=padding))

            in_chan = out_chan

    def forward(self, x):

        for layer in self.layers:

            x = layer(x)

        return x


class Encoder(nn.Module):

    def __init__(self, in_chan, out_chan, padding, downhill=3):

        super(Encoder, self).__init__()

        self.enc_layers = nn.ModuleList()

        for _ in range(downhill):

            self.enc_layers += [
                CNNBlocks(n_conv=2, in_chan=in_chan, out_chan=out_chan, padding=padding),
                nn.MaxPool2d(2, 2)
            ]

            in_chan = out_chan
            out_chan *= 2

        self.enc_layers.append(CNNBlocks(n_conv=2, in_chan=in_chan, out_chan=out_chan, padding=padding))

    def forward(self, x):

        route_connection = []

        for layer in self.enc_layers:

            if isinstance(layer, CNNBlocks):

                x = layer(x)

                route_connection.append(x)

                # Used for checking the size of the output
                # print(x.shape)

            else:

                x = layer(x)

                # Used for checking the size of the output
                # print(x.shape)

        return x, route_connection


class Decoder(nn.Module):

    def __init__(self, in_chan, out_chan, exit_chan, padding, uphill=4):

        super(Decoder, self).__init__()

        self.exit_chan = exit_chan

        self.layers = nn.ModuleList()

        for i in range(uphill):

            self.layers += [
                nn.ConvTranspose2d(in_chan, out_chan, kernel_size=2, stride=2),
                CNNBlocks(n_conv=2, in_chan=in_chan, out_chan=out_chan, padding=padding),
            ]

            in_chan //= 2
            out_chan //= 2

        # I changed this here to explicitly indicate padding = 0
        self.layers.append(
            nn.Conv2d(in_chan, exit_chan, kernel_size=1, padding=0),
        )

        # This is introduced in ID8 such that binary classification of each pixel may be performed
        self.layers.append(
            nn.Softmax(dim=1)
        )

    def forward(self, x, routes_connection):

        routes_connection.pop(-1)

        for layer in self.layers:

            if isinstance(layer, CNNBlocks):

                routes_connection[-1] = center_crop(routes_connection[-1], x.shape[2])

                x = torch.cat([x, routes_connection.pop(-1)], dim=1)

                x = layer(x)

                # Used for checking the size of the output
                # print(x.shape)

            else:

                x = layer(x)

        return x


class UNET(nn.Module):

    def __init__(self, in_chan, first_out_chan, exit_chan, downhill, padding=0):

        super(UNET, self).__init__()

        self.encoder = Encoder(in_chan, first_out_chan, padding=padding, downhill=downhill)

        self.decoder = Decoder(first_out_chan * (2 ** downhill), first_out_chan * (2 ** (downhill - 1)), exit_chan,
                               padding=padding, uphill=downhill)

    def forward(self, x):

        enc_out, routes = self.encoder(x)

        # Used to indicate when model shifts from downsampling to upsampling
        # print("BOTTLENECK")

        out = self.decoder(enc_out, routes)

        return out


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

# print(test_segments.shape)
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






def unek_segmentation(train_data, test_data, training_epochs=50):

    # HAD TO ADD IN PADDING = 1 FOR THIS TO WORK AND MATCH NO. CHANNELS AND IM SIZE GIVEN IN DIAGRAM IN ID8
    # EVEN THOUGH ORIGINAL PAPER SAID NO PADDING
    # unet = UNET(5, 16, 1, padding=1, downhill=4)
    #
    # predicted_segmentations = unet(binned_skymap_patches)

    # Create U-Net model - padding = 1 ("same convolution") to match no. channels and output sizes given in ID8 diagram
    unet = UNET(5, 16, 1, padding=1, downhill=4)

    # TRAINING

    # Define loss function and optimiser - assume same as original U-Net paper and that of the Centroid-NET in ID8
    loss_fn = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(unet.parameters(), lr=0.01)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, min_lr=10 ** -5)

    # Training epochs
    for epoch in range(training_epochs):

        print("EPOCH {}".format(epoch))

        unet.train(True)

        for i, data in enumerate(train_data):

            inputs, labels = data[0], data[1]

            # Make sure to zero gradients when calculating loss and don't update model
            output = unet(inputs)

            # Compute loss and gradients
            loss = loss_fn(output, labels)
            loss.backward()

            # Adjust weights
            optimiser.step()

        # Set to evaluate mode
        unet.eval()

        running_vloss = 0.0

        with torch.no_grad():
            for i, vdata in enumerate(test_data):

                vinputs, vlabels = vdata[0], vdata[1]

                voutputs = unet(vinputs)

                vloss = loss_fn(voutputs, vlabels)

                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)

        # Calling after validation loss - decrease learning rate if no improvement
        scheduler.step(avg_vloss)

    return unet


def k_means_clustering(binary_segments):

    source_centres_in_each_image = []

    num_segments_done = 0

    for segment in binary_segments:

        print("Segment: {}".format(num_segments_done + 1))

        D = segment[0].detach().numpy()

        l_sth = 0.2
        l_snn = -10
        R = 5
        diameter = R * 2

        # As we have used SoftMax, our image isn't exactly binary - this will make it so
        V_D = np.argwhere(D > l_sth)

        # Best score so far
        s_k_max = 0

        # Best number of sources so far
        k_best = 0

        # Centre of each cluster determined by k_best
        best_centres = []

        # Determine the number of sources/clusters present in the image
        for k in range(1, 50):

            D_tmp = copy.deepcopy(D)

            s_k = 0

            k_centroids = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(V_D)

            for c in k_centroids.cluster_centers_:

                # find pixels of D inside R

                x_centre = round(c[0])
                y_centre = round(c[1])

                x_range = np.clip(np.arange(x_centre - diameter, x_centre + diameter), a_min=0, a_max=63)
                y_range = np.clip(np.arange(y_centre - diameter, y_centre + diameter), a_min=0, a_max=63)

                potential_coords = np.unique(np.array([(x_val, y_val) for x_val in x_range for y_val in y_range]),
                                             axis=0)

                distances_from_centre_coord = np.linalg.norm(potential_coords - c, ord=2, axis=1)

                p_c = potential_coords[(distances_from_centre_coord < R)]

                # Update s_k
                s_k += np.sum(D_tmp[p_c[:, 0], p_c[:, 1]])

                # Redefine scores such that if the points are included in another cluster, the score is penalised
                for p_c_coord in p_c:
                    D_tmp[p_c_coord[0], p_c_coord[1]] = l_snn

            if s_k > s_k_max:

                k_best = k
                s_k_max = s_k
                best_centres = k_centroids.cluster_centers_

        source_centres_in_each_image.append(best_centres)

        num_segments_done += 1

    return source_centres_in_each_image











# RANDOM SPLIT OF INDICES AT THE MOMENT - 80% vs 20% split
train_indices = np.random.choice(256, size=204, replace=False)

test_indices = np.array([k for k in range(256) if k not in train_indices])

# Stack test images and test segmented images

test_data = [(test_images[k], test_segments[k]) for k in range(n)]

# Split data into test and train (EVENTUALLY USE SCIKIT LEARN TO DO SO)
train_split = Subset(test_data, train_indices)
test_split = Subset(test_data, test_indices)

# Create batches
train_batches = DataLoader(train_split, batch_size=128, shuffle=True)
test_batches = DataLoader(test_split, batch_size=128)

# Binary segmentation of image

# Train U-Net
# trained_model = unek_segmentation(train_batches, test_batches)

# Test outputs
# unet_outputs = trained_model(test_batches)

# Create sources and their centres
# k_means_clustering(unet_outputs)

k_means_clustering(test_segments)






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
