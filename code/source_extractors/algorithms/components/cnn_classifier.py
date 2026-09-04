from .custom_datasets import ClassifierSubPatchesDataset
import numpy as np
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader


class SourceClassifier(nn.Module):

    # Architecture is that specified in ID8. This implementation is completely unique and created by this author.

    def __init__(self):
        # Input has dimensions of 128, 5, 7, 7

        super(SourceClassifier, self).__init__()

        # self.batch_norm = nn.BatchNorm2d(num_features=5)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True)
        )

        # SHOULD BE SAME OUTPUT SIZE
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.flatten = nn.Flatten()

        # Dense layers - here the activation functions are applied separately
        self.dense_layers = nn.Sequential(

            # nn.Linear(in_features=288, out_features=128),
            # nn.ReLU(inplace=True),
            # nn.Linear(in_features=1568, out_features=128),
            # nn.ReLU(inplace=True),
            nn.Linear(in_features=144, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=16, out_features=3),
            # nn.Softmax(dim=1)

        )

    def forward(self, x):
        # x = self.batch_norm(x)
        x = self.conv_layers(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.dense_layers(x)

        return x


def classifier_train(train_data, test_data, device, training_epochs=50, save_file="./algorithms/pre_trained_models/"
                                                                                  "classifier.pt"):

    classifier = SourceClassifier().to(device)

    # Define loss function
    loss_fn = nn.CrossEntropyLoss()

    # ADDED IN SOME WEIGHT DECAY
    optimiser = torch.optim.Adam(classifier.parameters(), lr=1.e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=2, min_lr=1.e-8)

    # Start with large value that is easily surpassed
    best_vloss = 100000000000000
    best_epoch = 0

    # Training epochs
    for epoch in range(training_epochs):

        print("EPOCH {}".format(epoch))

        classifier.train()

        training_loss = 0.0

        for i, data in enumerate(train_data):

            inputs, labels = data["subpatch"].to(device, dtype=torch.float32), data["label"].to(device, dtype=torch.float32)

            optimiser.zero_grad()

            # Make sure to zero gradients when calculating loss and don't update model
            output = classifier(inputs)

            # Compute loss and gradients
            loss = loss_fn(output, labels)
            loss.backward()

            # Adjust weights
            optimiser.step()

            training_loss += loss.item()

        # Set to evaluate mode
        classifier.eval()

        avg_training_loss = training_loss / (i + 1)

        print("Training loss: {}".format(avg_training_loss))

        running_vloss = 0.0

        with torch.no_grad():
            for i, vdata in enumerate(test_data):

                vinputs, vlabels = vdata["subpatch"].to(device, dtype=torch.float32), vdata["label"].to(device, dtype=torch.float32)

                voutputs = classifier(vinputs)

                vloss = loss_fn(voutputs, vlabels)

                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)

        print("Validation Loss: {}".format(avg_vloss.item()))

        # if this is the best model (in terms of loss) found so far, save model
        if avg_vloss < best_vloss:

            best_vloss = avg_vloss
            best_epoch = epoch

            torch.save(classifier.state_dict(), save_file)

        scheduler.step(avg_vloss)

    return classifier, best_epoch


def classification_neural_network(train_data, validation_data, test_data, pretrained=False,
                                  save_file="./algorithms/pre_trained_models/classifier.pt"):
    device = torch.device("mps")
    print("Using Device: ", device)

    # INITIALISE CLASSIFIER
    if not pretrained:

        # Get data into efficient dataloader

        # train_data_loader = DataLoader(ClassifierSubPatchesDataset(data=train_data, num_sub_patches=len(train_data)), batch_size=128, shuffle=True)
        #
        # validation_data_loader = DataLoader(ClassifierSubPatchesDataset(data=validation_data, num_sub_patches=len(validation_data)),
        #                         batch_size=128, shuffle=True)

        train_data_loader = DataLoader(ClassifierSubPatchesDataset(data=train_data, num_sub_patches=len(train_data)), batch_size=64, shuffle=True)

        validation_data_loader = DataLoader(ClassifierSubPatchesDataset(data=validation_data, num_sub_patches=len(validation_data)),
                                batch_size=64, shuffle=True)

        # Train classifier on data
        classifier_model, best_epoch_classifier = classifier_train(train_data=train_data_loader, test_data=validation_data_loader,
                                                                   device=device, save_file=save_file)

        print("BEST EPOCH: {}".format(best_epoch_classifier))

    else:

        # Use pre-trained model
        classifier_model = SourceClassifier().to(device)
        classifier_model.load_state_dict(torch.load(save_file, weights_only=True, map_location=device))

    # Set to model evaluation model to ensure not accidentally continuing training
    classifier_model.eval()

    # EXTRACT TEST DATA

    testing_patches = np.array([i[0] for i in test_data])
    actual_labels = np.array([i[1] for i in test_data])

    with torch.no_grad():
        classifier_predictions = classifier_model(torch.tensor(testing_patches, dtype=torch.float32).to(device))

    classifier_predictions = classifier_predictions.detach().cpu().numpy()

    return actual_labels, classifier_predictions

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
# Drop out - https://machinelearningmastery.com/using-dropout-regularization-in-pytorch-models/
# Fixing Nan Predictions - https://discuss.pytorch.org/t/outputing-nan-as-predictions-in-my-neural-network-training-loop
# /183151/2
# Flatten with Linear - https://discuss.pytorch.org/t/should-i-flatten-before-the-linear-layer/43570
# ML Image Classification - https://blog.hyperiondev.com/post/machine-learning/
# Nan Prediction Error - https://discuss.pytorch.org/t/outputing-nan-as-predictions-in-my-neural-network-training-loop/
# 183151/2
# Overfitting Mitigation - https://datascience.stackexchange.com/questions/65471/validation-loss-much-higher-than-
# training-loss
# RF Classifier - https://www.geeksforgeeks.org/machine-learning/random-forest-for-image-classification-using-opencv/
# #google_vignette
# RF Classifier - https://machinelearningmastery.com/random-forest-for-image-classification-using-opencv/
# SVM Classifier - https://machinelearningmastery.com/support-vector-machines-for-image-classification-and-detection-
# using-opencv/
# Updating Learning Rate - https://stackoverflow.com/questions/48324152/how-to-change-the-learning-rate-of-an-optimizer-
# at-any-given-moment-no-lr-sched
