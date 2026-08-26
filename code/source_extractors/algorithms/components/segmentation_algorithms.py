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
from .data_preparation import ml_segmentation_data_prep, FermiCountMapDataset
import numpy as np
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.functional import center_crop


class CNNBlock(nn.Module):

    # Changed here, as padding must = 1 to get the correct shapes of outputs indicated in ID8
    # N.B. do NOT use batch normalisation (ID11)
    def __init__(self, in_chan, out_chan, kernel_size=3, stride=1, padding=1):
        super(CNNBlock, self).__init__()

        self.seq_block = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=True),
            nn.ReLU(inplace=True)
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

        # x = self.batch_norm(x)
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

    # TRAINING

    # Define loss function and optimiser - changed from that proposed in ID11 and ID8
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # optimiser = torch.optim.Adam(unet_model.parameters(), lr=1.e-5)



    optimiser = torch.optim.Adam(unet_model.parameters(), lr=1.e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, min_lr=1.e-7, patience=5)



    # Start with large value that is easily surpassed
    best_vloss = 100000000000000
    best_epoch = 0

    # Training epochs
    for epoch in range(training_epochs):

        print("EPOCH {}".format(epoch))

        unet_model.train()

        for i, data in enumerate(train_data):

            inputs, labels = data["patch"].to(device), data["mask"].to(device)

            # zero gradients for each batch
            optimiser.zero_grad()

            # Make sure to zero gradients when calculating loss and don't update model
            output = unet_model(inputs).squeeze(1)

            loss = loss_fn(output.float(), labels.float())

            loss.backward()

            # Adjust weights
            optimiser.step()

        # Set to evaluate mode
        unet_model.eval()

        running_vloss = 0.0

        with torch.no_grad():
            for i, vdata in enumerate(test_data):

                vinputs, vlabels = vdata["patch"].to(device), vdata["mask"].to(device)

                voutputs = unet_model(vinputs).squeeze(1)

                vloss = loss_fn(voutputs.float(), vlabels.float())

                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)

        print(avg_vloss)

        # if this is the best model (in terms of loss) found so far, save model
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            best_epoch = epoch

            torch.save(unet_model.state_dict(), save_file)

        # Calling after validation loss - decrease learning rate if no improvement
        scheduler.step(avg_vloss)

        if epoch == 40:

            import matplotlib.pyplot as plt

            plt.imshow(voutputs[0].to('cpu').detach().numpy())

            plt.show()

    return unet_model, best_epoch


def unet(training_maps, training_masks, validation_maps, validation_masks, testing_maps, pretrained=False,
          real=False, save_file="./algorithms/pre_trained_models/unet.pt"):

    device = torch.device("mps")
    print("Using Device: ", device)

    # INITIALISE SEGMENTER
    if not pretrained:

        # CREATE DATASETS

        training_patch_dataset = DataLoader(
            FermiCountMapDataset(patches=copy.deepcopy(training_maps), masks=copy.deepcopy(training_masks), num_patches=training_maps.shape[0]),
            batch_size=64, shuffle=True)

        validation_patch_dataset = DataLoader(
            FermiCountMapDataset(patches=copy.deepcopy(validation_maps), masks=copy.deepcopy(validation_masks), num_patches=validation_maps.shape[0]),
            batch_size=64, shuffle=True)

        # Train classifier on data
        model, best_epoch = unet_train(train_data=training_patch_dataset, test_data=validation_patch_dataset,
                                       save_file=save_file, device=device, training_epochs=50)

        print("BEST EPOCH: {}".format(best_epoch))

    else:

        # Use pre-trained model
        model = UNET(5, 16, 1, padding=1, downhill=4).to(device)
        model.load_state_dict(torch.load(save_file, weights_only=True, map_location=device))

    # Set to model evaluation model to ensure not accidentally continuing training
    model.eval()

    if real:

        with torch.no_grad():

            test_data_predictions = model(torch.from_numpy(testing_maps).to(device)).detach().cpu().numpy()

        return np.array([]), np.array([]), test_data_predictions

    else:

        with torch.no_grad():

            train_data_predictions = np.zeros((training_maps.shape[0], 1, 64, 64))
            validation_data_predictions = np.zeros((validation_maps.shape[0], 1, 64, 64))
            test_data_predictions = np.zeros((testing_maps.shape[0], 1, 64, 64))

            # Similar to batch loading - prevents GPU from running out of storage
            for s in range(0, training_maps.shape[0], 1000):

                lower = s
                upper = min(s + 1000, training_maps.shape[0])

                train_data_predictions[lower:upper] = model(torch.from_numpy(training_maps[lower:upper]).to(device)).detach().cpu().numpy()

                validation_data_predictions[lower:upper] = model(torch.from_numpy(validation_maps[lower:upper]).to(device)).detach().cpu().numpy()

                test_data_predictions[lower:upper] = model(torch.from_numpy(testing_maps[lower:upper]).to(device)).detach().cpu().numpy()

        return train_data_predictions, validation_data_predictions, test_data_predictions

# REFERENCES
# All 1s or 0s - https://discuss.huggingface.co/t/binary-model-either-predicts-all-0s-or-all-1s/27871
# Class Imbalance - https://stackoverflow.com/questions/56841451/why-is-my-neural-net-only-predicting-one-class-binary-
# classification
# Casting Tensor Types - https://discuss.pytorch.org/t/how-to-cast-a-tensor-to-another-type/2713
# Class Weights - https://discuss.pytorch.org/t/using-class-weights-with-loss-function-for-train-val-test-splits/222696
# Convolutional Layers - https://en.wikipedia.org/wiki/Convolutional_layer
# Cross-Entropy Loss Segmentation - https://discuss.pytorch.org/t/use-crossentropyloss-in-multiclass-semantic-
# segmentation/158141
# Custom Datasets - https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html
# Data Types - https://stackoverflow.com/questions/67456368/pytorch-getting-runtimeerror-found-dtype-double-but-expected
# -float
# Detach - https://stackoverflow.com/questions/49768306/pytorch-tensor-to-numpy-array
# Early Convergence - https://stackoverflow.com/questions/55973335/best-way-to-overcome-early-convergence-for-machine-
# learning-model
# GPU - https://stackoverflow.com/questions/61565293/issue-training-pytorch-model-on-gpu?rq=4
# GPU Error - https://stackoverflow.com/questions/59013109/runtimeerror-input-type-torch-floattensor-and-weight-type-
# torch-cuda-floatte
# GPU Transfer - https://stackoverflow.com/questions/63061779/pytorch-when-do-i-need-to-use-todevice-on-a-model-or-
# tensor
# Looked At (Did Not Use) - https://github.com/milesial/Pytorch-UNet/blob/master/train.py
# Model Not Training - https://discuss.pytorch.org/t/model-not-training/5055
# Multiclass Loss Function - https://discuss.pytorch.org/t/unet-multiclass-loss-function-selection/138106
# NaN Loss - https://discuss.pytorch.org/t/nan-loss-issues-with-precision-16-in-pytorch-lightning-gan-training/204369/4
# Neuron Datatype - https://discuss.pytorch.org/t/can-i-explicitly-set-dtype-for-torch-nn-conv2d/56563
# No Gradient - https://stackoverflow.com/questions/65532022/lack-of-gradient-when-creating-tensor-from-numpy
# Prediction with Model - https://discuss.pytorch.org/t/making-a-prediction-with-a-trained-model/2193
# Pytorch Documentation - https://pytorch.org/get-started/locally/
# Torch Types - https://stackoverflow.com/questions/70267810/pytorch-runtimeerror-expected-floating-point-type-for-
# target-with-class-proba
# Random Forest Classifier - https://stackoverflow.com/questions/49991677/using-randomforestclassifier-decision-path-how
# -do-i-tell-which-samples-the-clas
# Removing Channel Dimension - https://stackoverflow.com/questions/74764062/how-to-remove-the-channel-dimension-within-a
# -pytorch-model
# Requires Grad - https://discuss.pytorch.org/t/when-might-i-want-to-use-requires-grad-true-on-an-input/9624/2
# Segmentation Loss Function - https://discuss.pytorch.org/t/loss-function-for-segmentation/129703
# Segmentation with Cross Entropy - https://discuss.pytorch.org/t/image-segmentation-with-cross-entropy-loss/79138/4
# Softmax Dimension - https://stackoverflow.com/questions/52513802/pytorch-softmax-with-dim
# Softmax Error - https://discuss.pytorch.org/t/implicit-dimension-choice-for-softmax-warning/12314/2
# Slicing Numpy Arrays - https://stackoverflow.com/questions/4257394/slicing-of-a-numpy-2d-array-or-how-do-i-extract-an-
# mxm-submatrix-from-an-nxn-ar
# Spatial Dimensions - https://discuss.pytorch.org/t/runtimeerror-only-batches-of-spatial-targets-supported-3d-tensors-
# but-got-targets-of-dimension-4/82098
# Squeeze - https://stackoverflow.com/questions/60619886/torch-squeeze-and-the-batch-dimension
# Subsets - https://stackoverflow.com/questions/47432168/taking-subsets-of-a-pytorch-dataset
# Training - https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
# Training Loop - https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
# Training Loss Plateau - https://stackoverflow.com/questions/76063234/pytorch-training-loss-is-0-00-and-validation-
# accuracy-is-1-00-from-first-epoch
# Training Loss Plateau - https://discuss.pytorch.org/t/loss-always-equal-to-zero-while-training-the-model/173676
# Training Loss Plateau - https://stackoverflow.com/questions/55311932/loss-not-decreasing-pytorch
# Training Loss Plateau - https://stackoverflow.com/questions/47813715/pytorch-loss-value-not-change
# Training Loss Plateau - https://discuss.pytorch.org/t/sloved-why-my-loss-not-decreasing/15924
# Training Loss Plateau - https://stackoverflow.com/questions/69352375/pytorch-is-running-natively-on-m1-macbook-but-
# something-isnt-working-properly
# Training Loss Plateau - https://www.reddit.com/r/learnmachinelearning/comments/12g78fz/training_loss_literally_does_
# not_change_help/
# Training Loss Plateau - https://discuss.pytorch.org/t/u-net-segmentation-loss-does-not-decrease/68962
# Training Loss Plateau - https://discuss.pytorch.org/t/model-does-not-train-same-loss-in-every-epoch/121428/3
# Training Loss Plateau - https://discuss.pytorch.org/t/loss-always-equal-to-zero-while-training-the-model/173676
# Training Loss Plateau - https://discuss.pytorch.org/t/loss-not-updating-in-pytorch/169048
# Tutorial GitHub - https://github.com/AlessandroMondin/U-NET/blob/main/dataset.py
# Tutorial GitHub - https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/image_
# segmentation/semantic_segmentation_unet/model.py
# U-Net Tutorial - https://medium.com/@alessandromondin/semantic-segmentation-with-pytorch-u-net-from-
# scratch-502d6565910a
# U-Net Not Working - https://discuss.pytorch.org/t/unet-implementation/426
# Weighted Loss Function - https://medium.com/@zergtant/use-weighted-loss-function-to-solve-imbalanced-data-
# classification-problems-749237f38b75
