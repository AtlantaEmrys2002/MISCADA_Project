from .components.classification_algorithms import SourceClassifier, classifier_train
from . components.clustering_algorithms import blob_detection
from .components.data_preparation import prepare_classifier_data
from . components.segmentation_algorithms import UNET, unet_train
import numpy as np
import torch


def uneb_algorithm(training_data, validation_data, testing_data, use_pretrained_detector=False,
                   use_pretrained_classifier=False,
                   pretrained_model_file="./benchmarks/pre_trained_models/unet.pt",
                   pretrained_classifier_file="./benchmarks/pre_trained_models/classifier.pt"):

    # SEMANTIC SEGMENTATION

    # Pre-trained parameter determines if we should use a U-Net I have already trained on data on create a new U-Net
    # and train it on the training/validation data

    if use_pretrained_detector is False:

        # Train U-Net on data
        model = unet_train(train_data=training_data, test_data=validation_data)

    else:

        # Use pre-trained model
        model = UNET(5, 16, 1, padding=1, downhill=4)
        model.load_state_dict(torch.load(pretrained_model_file, weights_only=True))

    # Set to model evaluation model to ensure not accidentally continuing training
    model.eval()

    # Feed test count maps to trained U-Net model to perform semantic segmentation

    testing_inputs = []
    test_patch_ids = []

    for i, vdata in enumerate(testing_data):
        test_patch_ids.append(vdata[0])
        testing_inputs.append(vdata[1])

    test_patch_ids = test_patch_ids[0].detach().cpu().numpy()

    with torch.no_grad():

        unet_predictions = np.array([model(i) for i in testing_inputs][0])

    # CLUSTERING

    unet_predictions = torch.from_numpy(unet_predictions)

    # Determine the number of sources present within each U-Net segmented image and return the location of their centres
    predicted_source_locations = blob_detection(unet_predictions)

    # CLASSIFICATION OF SOURCES

    if use_pretrained_classifier is False:

        # Train classifier on training and validation data outputs

        training_inputs = []
        validation_inputs = []

        train_patch_ids = []
        validation_patch_ids = []

        for i, vdata in enumerate(validation_data):
            validation_patch_ids.append(vdata[0])
            validation_inputs.append(vdata[1])

        for i, vdata in enumerate(training_data):
            train_patch_ids.append(vdata[0])
            training_inputs.append(vdata[1])

        validation_patch_ids = validation_patch_ids[0].detach().cpu().numpy()
        training_patch_ids = train_patch_ids[0].detach().cpu().numpy()

        # GET PREDICTED LOCATIONS FOR TRAINING AND VALIDATION DATA

        with torch.no_grad():

            training_unet_predictions = np.array([model(i) for i in training_inputs][0])
            validation_unet_predictions = np.array([model(i) for i in validation_inputs][0])

        # CLUSTERING (SOURCE LOCALISATION)

        training_unet_predictions = torch.from_numpy(training_unet_predictions)
        validation_unet_predictions = torch.from_numpy(validation_unet_predictions)

        # Determine the number of sources present within each U-Net segmented image and return the location of their
        # centres
        train_predicted_source_locations = blob_detection(training_unet_predictions)
        validation_predicted_source_locations = blob_detection(validation_unet_predictions)

        training_inputs = training_inputs[0].detach().cpu().numpy()
        validation_inputs = validation_inputs[0].detach().cpu().numpy()

        # Prepare detected sources for classification (effectively, data prep)
        train_batches_class = (
            prepare_classifier_data(patches=training_inputs, predicted_locations=train_predicted_source_locations,
                                    patch_ids=training_patch_ids))

        validation_batches_class = (
            prepare_classifier_data(patches=validation_inputs,
                                    predicted_locations=validation_predicted_source_locations,
                                    patch_ids=validation_patch_ids))

        # Need to BALANCE THESE ABOVE DATASETS - FIND GREATEST NUMBER OF OCCURRENCES AND THEN DUPLICATE AS MANY AS POSSIBLE TO FILL UP!!!!!!!

        # Train classifier on data
        classifier_model, best_epoch_classifier = classifier_train(train_data=train_batches_class,
                                                                   test_data=validation_batches_class,
                                                                   save_file=pretrained_classifier_file)

        print("BEST EPOCH: {}".format(best_epoch_classifier))

    else:

        # Use pre-trained model
        classifier_model = SourceClassifier()
        classifier_model.load_state_dict(torch.load(pretrained_classifier_file, weights_only=True))

    # TEST CLASSIFIER

    testing_inputs = []

    for i, vdata in enumerate(testing_data):
        testing_inputs.append(vdata[1])

    testing_inputs = testing_inputs[0].detach().cpu().numpy()

    test_batches_class = prepare_classifier_data(patches=testing_inputs, predicted_locations=predicted_source_locations,
                                                 patch_ids=test_patch_ids)

    # Set to model evaluation model to ensure not accidentally continuing training
    classifier_model.eval()

    # Feed test count maps to trained U-Net model to perform semantic segmentation

    testing_inputs_class = []
    testing_actual_labels = []

    for i, vdata in enumerate(test_batches_class):
        testing_inputs_class.append(vdata[0])
        testing_actual_labels.append(vdata[1])

    actual_classes = testing_actual_labels[0].detach().cpu().numpy()

    with torch.no_grad():

        classifier_predictions = np.array([classifier_model(j) for j in testing_inputs_class][0])

    # Return the segmented images returned by U-Net and locations of source centres returned by K-means for the TEST
    # data, as well as the predictions of the class of each predicted source
    return (unet_predictions.detach().cpu().numpy(), predicted_source_locations, classifier_predictions, actual_classes,
            test_patch_ids)

    # # Return the segmented images returned by U-Net and locations of source centres returned by K-means
    # return unet_predictions.detach().cpu().numpy(), source_locations, test_patch_ids

# REFERENCES

# Data Types - https://stackoverflow.com/questions/67456368/pytorch-getting-runtimeerror-found-dtype-double-but-expected
# -float
# ID25 - followed their theory/mathematical definition to implement my own version
