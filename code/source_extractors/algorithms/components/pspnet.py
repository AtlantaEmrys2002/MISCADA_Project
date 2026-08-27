import torch
from torch import nn

class CNN(nn.Module):

    def __init__(self, in_chan, out_chan):

        super(CNN, self).__init__()

        self.seq_block = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=True),
            nn.LeakyReLU(inplace=True)
        )




    def forward(self, x):

        pass



