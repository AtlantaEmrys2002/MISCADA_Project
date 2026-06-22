import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.transforms.functional import center_crop



# Generate random test data with the correct shape

# 64 x 64 image for each of the energy bins which will be 6 here - IMAGES MUST BE NORMALISED
# GENERATING 100 RANDOM IMAGES

test_images = torch.rand((10, 5, 64, 64))

# RANDOM BINARY SEGMENTATIONS OF THE OUTPUT (WHICH IS THE IMAGE SEGMENTED AS BACKGROUND OR FOREGROUN - SOURCED
# REMEMBER - WILL NOT BE BINNED - SO JUST NEED 100 64 x 64 IMAGES
# test_segments = np.random.choice(a=np.array([0, 1]), size=(10, 64, 64))

test_segments = np.random.choice(a=np.array([0, 1]), size=(10, 64, 64))

# Random 0 and 1s torch tensor https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146/14

a = np.array([0, 1])
p = np.array([0.5, 0.5])
n = 10
replace = True

b = np.random.choice(a, p=p, size=n, replace=replace)

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


class CNNBlock(nn.Module):

    def __init__(self, in_chan, out_chan, kernel_size=3, stride=1, padding=0):

    # CHANGED AS ORIGINAL RECOMMENDED STRIDE OF 2 AND NO PADDING
    # def __init__(self, in_chan, out_chan, kernel_size=3, stride=2, padding=0):

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

    # FOLLOWING ID8 - RECOMMEND 2 CONV BEFORE POOLING PER CNN BLOCK
    # DOWNHILL SHOULD ALSO BE 4

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

                print(x.shape)

            else:

                x = layer(x)

                print(x.shape)

        return x, route_connection


class Decoder(nn.Module):

    def __init__(self, in_chan, out_chan, exit_chan, padding, uphill=4):

        super(Decoder, self).__init__()

        self.exit_chan = exit_chan

        self.layers = nn.ModuleList()

        for i in range(uphill):

            # KEEPING STRIDE = 2 AT THE MOMENT

            self.layers += [
                nn.ConvTranspose2d(in_chan, out_chan, kernel_size=2, stride=2),
                # nn.ConvTranspose2d(in_chan, out_chan, kernel_size=2, stride=1),
                CNNBlocks(n_conv=2, in_chan=in_chan, out_chan=out_chan, padding=padding),
            ]

            in_chan //= 2
            out_chan //= 2

        # self.layers.append(
        #     nn.Conv2d(in_chan, exit_chan, kernel_size=1, padding=padding),
        # )

        self.layers.append(
            nn.Conv2d(in_chan, exit_chan, kernel_size=1, padding=0),
        )

    def forward(self, x, routes_connection):

        routes_connection.pop(-1)

        for layer in self.layers:

            if isinstance(layer, CNNBlocks):

                routes_connection[-1] = center_crop(routes_connection[-1], x.shape[2])

                x = torch.cat([x, routes_connection.pop(-1)], dim=1)

                x = layer(x)

                print(x.shape)

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

        print("BOTTLENECK")

        out = self.decoder(enc_out, routes)

        return out


# HAD TO ADD IN PADDING = 1 FOR THIS TO WORK AND MATCH NO. CHANNELS AND IM SIZE GIVEN IN DIAGRAM IN ID8
# EVEN THOUGH ORIGINAL PAPER SAID NO PADDING
unet = UNET(5, 16, 1, padding=1, downhill=4)

print(unet)

test = test_images

print("out unet shape is", unet(test).shape)







# REFERENCES
# ID8 - followed their theory/mathematical definition to implement my own version
# Pytorch Documentation - https://pytorch.org/get-started/locally/
# Tutorial - https://medium.com/@alessandromondin/semantic-segmentation-with-pytorch-u-net-from-scratch-502d6565910a
# Tutorial GitHub - https://github.com/AlessandroMondin/U-NET/blob/main/dataset.py
