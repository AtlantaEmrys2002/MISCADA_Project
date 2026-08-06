import torch
from torch import nn


class CategoricalCrossEntropy(nn.Module):

    def __init__(self):
        super(CategoricalCrossEntropy, self).__init__()


    def forward(self, predictions, targets):

        N = len(predictions)

        # return nn.NLLLoss()(torch.log(predictions), targets)

        # see references for this below line - borrowed for medium article on categorical cross entropy loss
        return -1 / N * torch.sum(torch.sum(targets * torch.log(predictions)))


class SourceClassifier(nn.Module):

    # Architecture is that specified in ID8. This implementation is completely unique and created by this author.

    def __init__(self):
        # Input has dimensions of 128, 5, 7, 7

        super(SourceClassifier, self).__init__()

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
            # nn.Softmax(dim=1)
            nn.Softmax(dim=1)

        )

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.conv_layers(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.dense_layers(x)

        return x


def classifier_train(train_data, test_data, device, training_epochs=50, save_file="./benchmarks/pre_trained_models/"
                                                                          "classifier.pt"):

    classifier = SourceClassifier().to(device)

    # Define loss function

    loss_fn = CategoricalCrossEntropy()
    optimiser = torch.optim.Adam(classifier.parameters(), lr=1.e-6)

    # Start with large value that is easily surpassed
    best_vloss = 100000000000000
    best_epoch = 0

    epochs_since_improvement = 0

    # Training epochs
    for epoch in range(training_epochs):

        print("EPOCH {}".format(epoch))

        classifier.train()

        for i, data in enumerate(train_data):
            inputs, labels = data[0].to(device), data[1].to(device)

            # Make sure to zero gradients when calculating loss and don't update model
            output = classifier(inputs)

            # Compute loss and gradients
            loss = loss_fn(output, labels)
            loss.backward()

            # Adjust weights
            optimiser.step()

        # Set to evaluate mode
        classifier.eval()

        running_vloss = 0.0

        with torch.no_grad():
            for i, vdata in enumerate(test_data):

                vinputs, vlabels = vdata[0].to(device), vdata[1].to(device)

                voutputs = classifier(vinputs)

                vloss = loss_fn(voutputs, vlabels)

                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)

        # if this is the best model (in terms of loss) found so far, save model
        if avg_vloss < best_vloss:

            best_vloss = avg_vloss
            best_epoch = epoch

            epochs_since_improvement = 0

            torch.save(classifier.state_dict(), save_file)

        else:

            epochs_since_improvement += 1

        # if no improvement in loss for 50 epochs, stop training
        if epochs_since_improvement == 50:

            break

        elif epochs_since_improvement == 5:

            # Half the learning rate

            for g in optimiser.param_groups:
                g['lr'] /= 2

    return classifier, best_epoch


# REFERENCES

# Categorical Cross Entropy - https://discuss.pytorch.org/t/categorical-cross-entropy-loss-function-equivalent-in-
# pytorch/85165/3
# Categorical Cross Entropy Loss - https://arjun-sarkar786.medium.com/implementation-of-all-loss-functions-deep-learning
# -in-numpy-tensorflow-and-pytorch-e20e72626ebd
# Categorical Cross Entropy Loss - https://www.geeksforgeeks.org/deep-learning/categorical-cross-entropy-in-multi-class
# -classification/
# Custom Loss Functions - https://machinelearningmastery.com/creating-custom-layers-loss-functions-pytorch/
# Dense Layers - https://apxml.com/courses/pytorch-for-tensorflow-developers/chapter-2-pytorch-nn-module-for-keras-
# users/common-layer-types-pytorch-tf
# Dense Layers 2 - https://discuss.pytorch.org/t/pytorch-torch-nn-equivalent-of-tensorflow-keras-dense-layers/133518
# Fixing Nan Predictions - https://discuss.pytorch.org/t/outputing-nan-as-predictions-in-my-neural-network-training-loop
# /183151/2
# Flatten with Linear - https://discuss.pytorch.org/t/should-i-flatten-before-the-linear-layer/43570
# Nan Prediction Error - https://discuss.pytorch.org/t/outputing-nan-as-predictions-in-my-neural-network-training-loop/
# 183151/2
# Updating Learning Rate - https://stackoverflow.com/questions/48324152/how-to-change-the-learning-rate-of-an-optimizer-
# at-any-given-moment-no-lr-sched
