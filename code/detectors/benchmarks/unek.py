from . components.clustering_algorithms import k_means_clustering
from . components.segmentation_algorithms import UNET, unet_train
import numpy as np
import torch


def unek_algorithm(training_data, validation_data, testing_data, use_pretrained=False,
                   pretrained_model_file="./benchmarks/pre_trained_models/unet.pt"):

    # SEMANTIC SEGMENTATION

    # Pre-trained parameter determines if we should use a U-Net I have already trained on data on create a new U-Net
    # and train it on the training/validation data

    if use_pretrained is False:

        # Train U-Net on data
        model = unet_train(train_data=training_data, test_data=validation_data)

        # Save model
        torch.save(model.state_dict(), pretrained_model_file)
        # shutil.move(pretrained_model_file, "./benchmarks/")

    else:

        # Use pre-trained model
        model = UNET(5, 16, 1, padding=1, downhill=4)
        model.load_state_dict(torch.load(pretrained_model_file, weights_only=True))

    # Set to model evaluation model to ensure not accidently continuing training
    model.eval()

    # Feed test count maps to trained U-Net model to perform semantic segmentation

    testing_inputs = []

    for i, vdata in enumerate(testing_data):

        testing_inputs.append(vdata[0])

    with torch.no_grad():

        unet_predictions = np.array([model(i) for i in testing_inputs][0])

    # CLUSTERING (SOURCE LOCALISATION)

    unet_predictions = torch.from_numpy(unet_predictions)

    # Determine the number of sources present within each U-Net segmented image and return the location of their centres
    source_locations = k_means_clustering(unet_predictions)

    # Return the segmented images returned by U-Net and locations of source centres returned by K-means
    return unet_predictions, source_locations


# REFERENCES

# ID8 - followed their theory/mathematical definition to implement my own version
