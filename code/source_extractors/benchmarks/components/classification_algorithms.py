import torch
from torch import nn

class SourceClassifier(nn.Module):

    # Architecture is that specified in ID8. This implementation is completely unique and created by this author.

    def __init__(self):

        # INPUT is 128, 5, 7, 7

        super().__init__()

        self.batch_norm = nn.BatchNorm2d(num_features=5)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # SHOULD BE SAME OUTPUT SIZE
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.flatten = nn.Flatten()

        # Dense layers - here the activation functions are applied separately
        self.dense_layers = nn.Sequential(

            nn.Linear(in_features=288, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=16, out_features=3),
            nn.Softmax()

        )

    def forward(self, x):

        x = self.batch_norm(x)
        x = self.conv_layers(x)
        x = self.max_pooling(x)

        print(x.shape)

        x = self.flatten(x)
        x = self.dense_layers(x)

        return x







# classifier = SourceClassifier()
#
# print(classifier)
#
# test = torch.rand(128, 5, 7, 7)
#
# print(classifier(test))

