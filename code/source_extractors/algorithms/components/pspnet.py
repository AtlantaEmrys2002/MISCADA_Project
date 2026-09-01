import copy
from .custom_datasets import FermiCountMapDataset
import numpy as np
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from ..utils import loss_values
import time
from torch.nn.functional import adaptive_avg_pool2d


class Block(nn.Module):
    """
    Consulted https://www.digitalocean.com/community/tutorials/writing-resnet-from-scratch-in-pytorch when creating
    ResNet blocks
    """

    def __init__(self, in_chan, out_chan, stride=1):

        super(Block, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=3, stride=stride, padding=1),
            nn.ReLU()
        )

        self.conv2 = nn.Conv2d(in_channels=out_chan, out_channels=out_chan, kernel_size=3, stride=1, padding=1)

        self.second_relu = nn.ReLU()

        self.downsample = None if stride == 1 else nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride)

        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample:
            out += self.downsample(x)

        return self.second_relu(out)


class FeatureMap(nn.Module):

    def __init__(self):

        super(FeatureMap, self).__init__()

        self.blocks = nn.Sequential(

            Block(5, 16, stride=1),
            Block(16, 32, stride=2),
            Block(32, 64, stride=2),

        )

    def forward(self, x):

        x = self.blocks(x)

        return x


class PyramidPoolingModule(nn.Module):

    def __init__(self, in_chan):

        super(PyramidPoolingModule, self).__init__()

        self.conv_1_x_1 = nn.Conv2d(in_channels=in_chan, out_channels=1, kernel_size=1, padding=0, stride=1)
        self.conv_2_x_2 = nn.Conv2d(in_channels=in_chan, out_channels=2, kernel_size=1, padding=0, stride=1)
        self.conv_3_x_3 = nn.Conv2d(in_channels=in_chan, out_channels=3, kernel_size=1, padding=0, stride=1)
        self.conv_6_x_6 = nn.Conv2d(in_channels=in_chan, out_channels=6, kernel_size=1, padding=0, stride=1)

        self.upsample_conv_1_x_1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=8)
        self.upsample_conv_2_x_2 = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=6, stride=2)
        self.upsample_conv_3_x_3 = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=2)
        self.upsample_conv_6_x_6 = nn.ConvTranspose2d(in_channels=6, out_channels=6, kernel_size=3, stride=1)

        self.upsample_1_x_1 = nn.Upsample(size=(16, 16), mode='bilinear')
        self.upsample_2_x_2 = nn.Upsample(size=(16, 16), mode='bilinear')
        self.upsample_3_x_3 = nn.Upsample(size=(16, 16), mode='bilinear')
        self.upsample_6_x_6 = nn.Upsample(size=(16, 16), mode='bilinear')

        self.final_conv = nn.Conv2d(in_channels=76, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.bilinear_upsampling = nn.Upsample(size=(64, 64), mode='bilinear')

    def forward(self, x):

        # Global average pooling - not implemented for GPU, so have to transport back to CPU.

        x_cpu = x.to('cpu')

        pool_6_x_6 = adaptive_avg_pool2d(x_cpu, output_size=(6, 6)).to('mps')
        pool_3_x_3 = adaptive_avg_pool2d(x_cpu, output_size=(3, 3)).to('mps')
        pool_2_x_2 = adaptive_avg_pool2d(x_cpu, output_size=(2, 2)).to('mps')
        pool_1_x_1 = adaptive_avg_pool2d(x_cpu, output_size=(1, 1)).to('mps')

        pooled_6 = self.conv_6_x_6(pool_6_x_6)
        pooled_3 = self.conv_3_x_3(pool_3_x_3)
        pooled_2 = self.conv_2_x_2(pool_2_x_2)
        pooled_1 = self.conv_1_x_1(pool_1_x_1)

        upsampled_1 = self.upsample_1_x_1(pooled_1)
        upsampled_2 = self.upsample_2_x_2(pooled_2)
        upsampled_3 = self.upsample_3_x_3(pooled_3)
        upsampled_6 = self.upsample_6_x_6(pooled_6)

        x = torch.cat((x, upsampled_6, upsampled_3, upsampled_2, upsampled_1), axis=1)

        x = self.final_conv(self.bilinear_upsampling(x))

        return x


class PSPNet(nn.Module):

    def __init__(self):
        super(PSPNet, self).__init__()

        self.feature_map = FeatureMap()

        self.pooling_module = PyramidPoolingModule(in_chan=64)

    def forward(self, x):

        x = self.feature_map(x)

        x = self.pooling_module(x)

        return x

def pspnet_train(train_data, test_data, device, training_epochs: int = 50,
               save_file: str = "./algorithms/pre_trained_models/pspnet.pt"):
    """Trains a PSPNet on provided data then evaluates the loss function on the validation data (here, called test data).

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

    start = time.time()

    # Create model
    pspnet_model = PSPNet().to(device)

    # TRAINING

    # Define loss function and optimiser - changed from that proposed in ID11 and ID8
    loss_fn = torch.nn.BCEWithLogitsLoss()

    optimiser = torch.optim.Adam(pspnet_model.parameters(), lr=1.e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, min_lr=1.e-8, patience=2)

    # Start with large value that is easily surpassed
    best_vloss = 100000000000000
    best_epoch = 0

    loss_record = []

    # Training epochs
    for epoch in range(training_epochs):

        print("EPOCH {}".format(epoch))

        pspnet_model.train()

        for i, data in enumerate(train_data):

            inputs, labels = data["patch"].to(device), data["mask"].to(device)

            # zero gradients for each batch
            optimiser.zero_grad()

            # Make sure to zero gradients when calculating loss and don't update model
            output = pspnet_model(inputs).squeeze(1)

            loss = loss_fn(output.float(), labels.float())

            loss.backward()

            # Adjust weights
            optimiser.step()

        # Set to evaluate mode
        pspnet_model.eval()

        running_vloss = 0.0

        with torch.no_grad():
            for i, vdata in enumerate(test_data):

                vinputs, vlabels = vdata["patch"].to(device), vdata["mask"].to(device)

                voutputs = pspnet_model(vinputs).squeeze(1)

                vloss = loss_fn(voutputs.float(), vlabels.float())

                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)

        loss_record.append(avg_vloss.item())

        # if this is the best model (in terms of loss) found so far, save model
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            best_epoch = epoch

            torch.save(pspnet_model.state_dict(), save_file)

        # Calling after validation loss - decrease learning rate if no improvement
        scheduler.step(avg_vloss)

    # Save loss values
    loss_values(epochs=list(range(0, training_epochs)), losses=loss_record, method="pspnet",
                directory="./../results/analysis_results/")

    print("Training time: {}".format(time.time() - start))

    return pspnet_model, best_epoch


def pspnet(training_maps, training_masks, validation_maps, validation_masks, testing_maps, pretrained=False,
          real=False, save_file="./algorithms/pre_trained_models/pspnet.pt"):

    device = torch.device("mps")
    print("Using Device: ", device)

    # INITIALISE SEGMENTER
    if not pretrained:

        # CREATE DATASETS

        training_patch_dataset = DataLoader(
            FermiCountMapDataset(patches=copy.deepcopy(training_maps), masks=copy.deepcopy(training_masks), num_patches=training_maps.shape[0]),
            batch_size=32, shuffle=True)

        validation_patch_dataset = DataLoader(
            FermiCountMapDataset(patches=copy.deepcopy(validation_maps), masks=copy.deepcopy(validation_masks), num_patches=validation_maps.shape[0]),
            batch_size=32, shuffle=False)

        # Train classifier
        model, best_epoch = pspnet_train(train_data=training_patch_dataset, test_data=validation_patch_dataset,
                                       save_file=save_file, device=device, training_epochs=70)

        print("BEST EPOCH: {}".format(best_epoch))

    else:

        # Use pre-trained model
        model = PSPNet().to(device)

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

            # Similar to batch loading - prevents GPU from running out of memory
            for s in range(0, training_maps.shape[0], 1000):

                lower = s
                upper = min(s + 1000, training_maps.shape[0])

                train_data_predictions[lower:upper] = model(
                    torch.from_numpy(training_maps[lower:upper]).to(device)).detach().cpu().numpy()

            for s in range(0, validation_maps.shape[0], 1000):

                lower = s
                upper = min(s + 1000, validation_maps.shape[0])

                validation_data_predictions[lower:upper] = model(
                    torch.from_numpy(validation_maps[lower:upper]).to(device)).detach().cpu().numpy()

            for s in range(0, testing_maps.shape[0], 1000):

                lower = s
                upper = min(s + 1000, testing_maps.shape[0])

                test_data_predictions[lower:upper] = model(
                    torch.from_numpy(testing_maps[lower:upper]).to(device)).detach().cpu().numpy()










            # for s in range(0, training_maps.shape[0], 1000):
            #
            #     lower = s
            #     upper = min(s + 1000, training_maps.shape[0])
            #
            #     train_data_predictions[lower:upper] = model(torch.from_numpy(training_maps[lower:upper]).to(device)).detach().cpu().numpy()
            #
            #     validation_data_predictions[lower:upper] = model(torch.from_numpy(validation_maps[lower:upper]).to(device)).detach().cpu().numpy()
            #
            #     test_data_predictions[lower:upper] = model(torch.from_numpy(testing_maps[lower:upper]).to(device)).detach().cpu().numpy()

        return train_data_predictions, validation_data_predictions, test_data_predictions

# REFERENCES

# Batch Size - https://www.coursera.org/articles/what-does-batch-size-mean-in-deep-learning
# Batch Size and Overfitting - https://stats.stackexchange.com/questions/266368/deep-learning-why-does-increase-batch-
# size-cause-overfitting-and-how-does-one-r
# Choosing Models - https://stackoverflow.com/questions/38724623/too-much-data-for-svm
# Concatenation with Torch - https://stackoverflow.com/questions/69115837/how-to-concatenate-a-list-of-tensors-on-a-
# specific-axis
# Learning Rate - https://stackoverflow.com/questions/63108131/pytorch-schedule-learning-rate
# Loss Convergence - https://www.reddit.com/r/deeplearning/comments/1npv5r6/why_the_loss_is_not_converging_in_my_neural/
# ResNet - https://medium.com/@raunakgola123/resnet-in-pytorch-unlock-the-magic-of-cnn-architectures-part-4-9a8b6e471b1e
# ResNet and Skip Connections - https://medium.com/@raunakgola123/resnet-in-pytorch-unlock-the-magic-of-cnn-architecture
# s-part-4-9a8b6e471b1e
# Tuning CNNs - https://www.reddit.com/r/MachineLearning/comments/3l5qu7/rules_of_thumb_for_cnn_architectures/
# Varying LR - https://discuss.pytorch.org/t/change-learning-rate/166336/6
