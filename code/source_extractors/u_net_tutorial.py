# N.B. Refamiliarised myself and started with tutorial https://medium.com/@alessandromondin/semantic-segmentation-with-
# pytorch-u-net-from-scratch-502d6565910a (referenced below) and built on top of that.
# Also consulted author's GitHub implementation - https://github.com/AlessandroMondin/U-NET

import torch
from torch import nn
from torchvision.transforms.functional import center_crop


class CNNBlock(nn.Module):

    def __init__(self, in_chan, out_chan, kernel_size=3, stride=1, padding=0):

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

    def __init__(self, in_chan, out_chan, padding, downhill=4):

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

        self.layers.append(
            nn.Conv2d(in_chan, exit_chan, kernel_size=1, padding=padding),
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


# unet = UNET(1, 64, 1, padding=0, downhill=4)
#
# test = torch.rand((3, 1, 572, 572))
#
# print("out unet shape is", unet(test).shape)


# REFERENCES

# Pytorch Documentation - https://pytorch.org/get-started/locally/
# Tutorial - https://medium.com/@alessandromondin/semantic-segmentation-with-pytorch-u-net-from-scratch-502d6565910a
# Tutorial GitHub - https://github.com/AlessandroMondin/U-NET/blob/main/dataset.py
