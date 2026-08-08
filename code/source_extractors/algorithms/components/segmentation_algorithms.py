"""
Segmentation algorithms used to distinguish between the foreground and background of photon count maps. N.B. includes
the training loops of deep learning-based methods.

This author consulted and adapted code from this tutorial - https://medium.com/@alessandromondin/semantic-segmentation-
with-pytorch-u-net-from-scratch-502d6565910a. This author also consulted the GitHub implementation here -
https://github.com/AlessandroMondin/U-NET. This author adjusted their implementation to be compatible with the dataset
created and processed inputs with layer sizes and pooling determined by the architecture presented in ID8 (Panes et al.
in report). This author also created their own training loop.

"""

import copy
from .data_preparation import ml_segmentation_data_prep
import numpy as np
import torch
from torch import nn
from torchvision.transforms.functional import center_crop


class LinearCombinationLoss(nn.Module):

    def __init__(self):
        super(LinearCombinationLoss, self).__init__()

    @staticmethod
    def forward(predictions, targets):
        N = len(predictions)

        # FROM ID11

        mse = (1 / (N ** 2)) * torch.sum((targets - predictions) ** 2)
        ce = (- 1 / (N ** 2)) * torch.sum(targets * torch.log(predictions))

        return mse + (100 * ce)

        # return -1 / N * torch.sum(torch.sum(targets * torch.log(predictions)))


class CNNBlock(nn.Module):

    # Changed here, as padding must = 1 to get the correct shapes of outputs indicated in ID8
    # N.B. do NOT use batch normalisation (ID11)
    def __init__(self, in_chan, out_chan, kernel_size=3, stride=1, padding=1):
        super(CNNBlock, self).__init__()

        self.seq_block = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False),
            # nn.BatchNorm2d(out_chan),
            # nn.ReLU(inplace=True)
            nn.ReLU()
        )

    def forward(self, x):
        """Returns the output of the block when passed data.

        Parameters
        ----------
        x
            Data to evaluate.
        """
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
        """Returns the output of the blocks when passed data.

        Parameters
        ----------
        x
            Data to evaluate.
        """
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
        """Returns the output of the encoder when passed data.

        Parameters
        ----------
        x
            Data to evaluate.
        """
        route_connection = []

        for layer in self.enc_layers:

            if isinstance(layer, CNNBlocks):
                x = layer(x)
                route_connection.append(x)

            else:
                x = layer(x)

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
            nn.Conv2d(in_chan, exit_chan, kernel_size=1),
        )

        # # This is introduced in ID8 such that binary classification of each pixel may be performed
        # self.layers.append(
        #     nn.Softmax(dim=1)
        # )

    def forward(self, x, routes_connection):
        """Returns the output of the decoder when passed data.

        Parameters
        ----------
        x
            Data to evaluate.
        routes_connection
            Outputs of previous layers to concatenate with outputs of later layers as a skip connection.
        """

        routes_connection.pop(-1)

        for layer in self.layers:

            if isinstance(layer, CNNBlocks):
                routes_connection[-1] = center_crop(routes_connection[-1], x.shape[2])
                x = torch.cat([x, routes_connection.pop(-1)], dim=1)
                x = layer(x)

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
        """Returns the output of the U-Net when passed data.

        Parameters
        ----------
        x
            Data to evaluate.
        """

        enc_out, routes = self.encoder(x)
        out = self.decoder(enc_out, routes)

        return out


def unet_train(train_data, test_data, device, training_epochs: int = 50,
               save_file: str = "./algorithms/pre_trained_models/unet.pt"):
    """Trains a U-Net on provided data then evaluates the loss function on the validation data (here, called test data).

    Parameters
    ----------
    train_data
        Data upon which to train the U-Net.
    test_data
        Data upon which to validate the loss function.
    device
        Indicates whether to use CPU or GPU.
    training_epochs : int
        The number of epochs for which to train the U-Net.
    save_file : str
        The file in which to save the weights of the best model.

    """
    # Adapted this code from https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html

    # HAD TO ADD IN PADDING = 1 FOR THIS TO WORK AND MATCH NO. CHANNELS AND IM SIZE GIVEN IN DIAGRAM IN ID8
    # EVEN THOUGH ORIGINAL PAPER SAID NO PADDING

    # Create U-Net model - padding = 1 ("same convolution") to match no. channels and output sizes given in ID8 diagram
    unet_model = UNET(5, 16, 1, padding=1, downhill=4).to(device)
    # unet_model = UNET(5, 32, 1, padding=1, downhill=3).to(device)

    # TRAINING

    # Define loss function and optimiser - assume same as original U-Net paper and that of the Centroid-NET in ID8
    # loss_fn = torch.nn.CrossEntropyLoss()

    # loss_fn = LinearCombinationLoss()

    loss_fn = torch.nn.BCEWithLogitsLoss().to(device)

    # optimiser = torch.optim.Adam(unet_model.parameters(), lr=0.01)
    optimiser = torch.optim.Adam(unet_model.parameters(), lr=1.e-4)
    # optimiser = torch.optim.SGD(unet_model.parameters(), lr=0.001)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, min_lr=10 ** -5)

    # Start with large value that is easily surpassed
    best_vloss = 100000000000000
    best_epoch = 0

    # Training epochs
    for epoch in range(training_epochs):

        print("EPOCH {}".format(epoch))

        unet_model.train()

        for i, data in enumerate(train_data):

            inputs, labels = data[1].to(device), data[2].to(device)

            # print(labels.shape)

            # # zero gradients for each batch
            # optimiser.zero_grad()

            # Make sure to zero gradients when calculating loss and don't update model
            output = unet_model(inputs)

            # print(output.shape)

            # print(output)

            # print(torch.sum(labels))

            # print(labels)

            # Compute loss and gradients
            # loss = loss_fn(output, labels)

            loss = loss_fn(output, labels)

            # zero gradients for each batch
            optimiser.zero_grad()

            loss.backward()

            # Adjust weights
            optimiser.step()

        running_vloss = 0.0

        # Set to evaluate mode
        unet_model.eval()

        with torch.no_grad():
            for i, vdata in enumerate(test_data):
                vinputs, vlabels = vdata[1].to(device), vdata[2].to(device)

                voutputs = unet_model(vinputs)

                vloss = loss_fn(voutputs, vlabels)

                running_vloss += vloss.item()

        avg_vloss = running_vloss / (i + 1)

        print(avg_vloss)

        # if this is the best model (in terms of loss) found so far, save model
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            best_epoch = epoch

            torch.save(unet_model.state_dict(), save_file)

        # Calling after validation loss - decrease learning rate if no improvement
        # scheduler.step(avg_vloss)

    return unet_model, best_epoch


def unet(training_maps, validation_maps, testing_maps, pretrained=False, real=False,
         save_file="./algorithms/pre_trained_models/unet.pt"):
    device = torch.device("mps")
    print("Using Device: ", device)

    # INITIALISE SEGMENTER
    if not pretrained:

        # Train classifier on data
        model, best_epoch = unet_train(train_data=training_maps, test_data=validation_maps,
                                       save_file=save_file, device=device)

        print("BEST EPOCH: {}".format(best_epoch))

    else:

        # Use pre-trained model
        model = UNET(5, 16, 1, padding=1, downhill=4).to(device)
        model.load_state_dict(torch.load(save_file, weights_only=True, map_location=device))

    # Set to model evaluation model to ensure not accidentally continuing training
    model.eval()

    if real:

        _, _, _, _, _, _, _, real_maps, _ = (
            ml_segmentation_data_prep(train_data=np.array([]), validation_data=np.array([]),
                                      test_data=copy.deepcopy(testing_maps)))

        with torch.no_grad():
            test_data_predictions = model(torch.from_numpy(real_maps).to(device)).detach().cpu().numpy()

        return np.array([]), np.array([]), test_data_predictions

    else:

        # EXTRACT TEST DATA

        (_, training_maps, _, _, validation_maps, _, _, testing_maps, _) = (
            ml_segmentation_data_prep(train_data=training_maps, validation_data=validation_maps,
                                      test_data=testing_maps))

        with torch.no_grad():
            train_data_predictions = model(torch.from_numpy(training_maps).to(device)).detach().cpu().numpy()
            validation_data_predictions = model(torch.from_numpy(validation_maps).to(device)).detach().cpu().numpy()
            test_data_predictions = model(torch.from_numpy(testing_maps).to(device)).detach().cpu().numpy()

        return train_data_predictions, validation_data_predictions, test_data_predictions

# REFERENCES
# Convolutional Layers - https://en.wikipedia.org/wiki/Convolutional_layer
# Pytorch Documentation - https://pytorch.org/get-started/locally/
# Torch Types - https://stackoverflow.com/questions/70267810/pytorch-runtimeerror-expected-floating-point-type-for-
# target-with-class-proba
# Softmax Error - https://discuss.pytorch.org/t/implicit-dimension-choice-for-softmax-warning/12314/2
# Training - https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
# Training Loop - https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
# Tutorial GitHub - https://github.com/AlessandroMondin/U-NET/blob/main/dataset.py
# Tutorial GitHub - https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/image_
# segmentation/semantic_segmentation_unet/model.py
# U-Net Tutorial - https://medium.com/@alessandromondin/semantic-segmentation-with-pytorch-u-net-from-
# scratch-502d6565910a
