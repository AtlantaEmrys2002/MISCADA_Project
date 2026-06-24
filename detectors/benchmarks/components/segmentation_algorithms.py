import torch
from torch import nn
from torchvision.transforms.functional import center_crop

# Consulted this tutorial when building the U-Net class -  https://medium.com/@alessandromondin/semantic-segmentation-
# with-pytorch-u-net-from-scratch-502d6565910a. Adjusted the implementation such that it was compatible with the
# dataset I have created (and the one created by ID8) and processed inputs with layer sizes and pooling identical to
# that outlined in ID8.


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
        out = self.decoder(enc_out, routes)

        return out


def unet_train(train_data, test_data, training_epochs=50):

    # Adapted this code from https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html

    # HAD TO ADD IN PADDING = 1 FOR THIS TO WORK AND MATCH NO. CHANNELS AND IM SIZE GIVEN IN DIAGRAM IN ID8
    # EVEN THOUGH ORIGINAL PAPER SAID NO PADDING

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


# REFERENCES
# Training Loop - https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
# U-Net Tutorial - https://medium.com/@alessandromondin/semantic-segmentation-with-pytorch-u-net-from-
# scratch-502d6565910a
