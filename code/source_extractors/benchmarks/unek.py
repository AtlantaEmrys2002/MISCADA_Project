import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from . components.classification_algorithms import classifier_train
from . components.clustering_algorithms import k_means_clustering
from . components.segmentation_algorithms import UNET, unet_train
import numpy as np
from . utils import get_lb_from_pixel, pixel_id
import torch
from xml.dom import minidom, Node


def xml_parser_locations(xml_file: str, coordinate_system='G'):

    # COULD CALL THIS FROM OTHER FUNCTION MAYBE??

    # GET NAME AND LOCATION OF SOURCE IN SKY

    docs = minidom.parse(xml_file)

    sources = docs.getElementsByTagName("source")

    coordinates = []

    # Remove diffuse sources - only processing point sources with this function
    sources = [sources[k] for k in range(len(sources)) if sources[k].getAttribute("type") != "DiffuseSource"]

    source_ids = []

    # Parse XML
    for source in sources:

        source_ids.append(source.getAttribute("name"))

        # PARSE SPATIAL PARAMETERS

        spatial_model = source.getElementsByTagName("spatialModel")[0]

        parameters = spatial_model.getElementsByTagName("parameter")

        coordinate = [0, 0]

        for param in parameters:
            name = param.getAttribute("name")

            if name == "RA":
                coordinate[0] = float(param.getAttribute("value"))
            else:
                coordinate[1] = float(param.getAttribute("value"))

        coordinates.append(coordinate)

    # Have coordinates in format [RA, DEC] - need to convert them to Lat-lon

    # Convert coordinates to np array
    coordinates = np.array(coordinates)

    if coordinate_system == 'G':

        # if want in galactic coordinates then convert

        # Get coordinates into numpy array then separate into list of lats and lons

        coordinates = SkyCoord(ra=coordinates[:, 0] * u.degree, dec=coordinates[:, 1] * u.degree, frame='icrs').galactic

        coordinates = np.array([coordinates.l.value, coordinates.b.value]).T

        return coordinates, source_ids

    elif coordinate_system == "C":

        # if want celestial coordinates, just return

        return coordinates, source_ids

    else:

        raise TypeError("Coordinate system not supported.")


def source_boxes(patches, predicted_source_locations):

    # Select 7 x 7 boxes around each patch location - all in Cartesian coordinates

    num_patches = patches.shape[0]

    boxes_for_each_patch = []

    for n in range(num_patches):

        classification_sub_patches = []

        predicted_locations_in_patch = predicted_source_locations[n].astype(int)

        patch = patches[n]

        for loc in predicted_locations_in_patch:

            x, y = loc[0], loc[1]

            # Select 7 x 7 grid around predicted location
            rows = np.arange(x - 3, x + 4)
            cols = np.arange(y - 3, y + 4)

            # if cannot create a 7 x 7 grid, ignore during classification
            if rows[0] < 0 or cols[0] < 0 or rows[-1] > 64 or cols[-1] > 64:

                continue

            else:

                new_patch = patch[:, rows, :][:, :, cols]

                classification_sub_patches.append(new_patch)

        boxes_for_each_patch.append(np.array(classification_sub_patches))

    return boxes_for_each_patch


def source_box_labels(patch_ids, predicted_source_locations):

    labels = []

    num_patches = patch_ids.shape[0]

    source_information = pd.read_csv("./../data_simulation/simulated_data/patches/patch_metadata.csv")

    # Gives the ID of the catalog that each patch is based on
    patch_catalogs = dict(zip(source_information["patch_id"].to_numpy(), source_information["catalog_id"].to_numpy() +
                              1))

    patch_centres = np.array(list(zip(source_information["centre_lon"].to_numpy(), source_information["centre_lat"].
                                      to_numpy())))

    # Figure out catalog for each patch - MAYBE CREATED A CSV FILE

    agn_coordinates_per_catalog = []
    pulsar_coordinates_per_catalog = []

    for catalog_id in range(1, np.max(source_information["catalog_id"].to_numpy()) + 2):

        actual_agn_coordinates, actual_agn_ids = xml_parser_locations(xml_file="./../data_simulation/simulated_data/"
                                                                               "catalogs/catalog_{}/agns.xml".
                                                                      format(catalog_id), coordinate_system='C')

        actual_psr_coordinates, actual_psr_ids = xml_parser_locations(xml_file="./../data_simulation/simulated_data/"
                                                                               "catalogs/catalog_{}/pulsars.xml".
                                                                      format(catalog_id), coordinate_system='C')

        agn_coordinates_per_catalog.append(dict(zip(actual_agn_ids, actual_agn_coordinates)))
        pulsar_coordinates_per_catalog.append(dict(zip(actual_psr_ids, actual_psr_coordinates)))

    for n in range(num_patches):

        patch_id = patch_ids[n]

        predicted_locs_for_patch = predicted_source_locations[n]

        catalog_of_patch = patch_catalogs[patch_id]

        # Get actual locations of sources in patch

        patch_information = pd.read_csv("./../data_simulation/simulated_data/patches/patch_{}/metadata.csv".format(n))

        # actual_sources_in_patch = patch_information["source_id"].to_numpy()

        actual_agn_in_patch = patch_information[patch_information["source_type"] == "AGN"]["source_id"].to_numpy()
        actual_psr_in_patch = patch_information[patch_information["source_type"] == "PSR"]["source_id"].to_numpy()

        # FROM SOURCE IDS ABOVE GET COORDINATES OF EACH SOURCE WITH THAT ID IN THE CATALOG ID FOUND ABOVE
        # THEN CALCULATE DISTANCE BETWEEN COORDS OF PIXEL IN WHICH PREDICTED (USInG HEALPY FUNCTION TO CONVERT TO
        # LAT LON THEN GO TO RA DEC) SOURCE IS IN AND THE ACTUAL LOCATIONS. IF the CLOSEST ACTUAL IS LESS THAN 0.3
        # DEGREES FROM PREDICTED THEN thAT PREDICtEd IS THAT SOURCE (IF MORE THAN ONE, PICK THE ONE THAT IS MORE
        # PREVALENT). IF THERE ARE NO SOURCES CLOSE TO PREDICTED SOURCE THEN LABEL AS FAKE

        # N.B. FOR ABOVE - ITERATE OVER PREDICTED SOURCES

        actual_agn_locations_in_celestial = [agn_coordinates_per_catalog[catalog_of_patch - 1][actual_agn_in_patch[k]]
                                             for k in range(len(actual_agn_in_patch))]

        actual_psr_locations_in_celestial = [pulsar_coordinates_per_catalog[catalog_of_patch - 1]
                                             [actual_psr_in_patch[k]] for k in range(len(actual_psr_in_patch))]

        # Since we have no way of knowing where in patch model will predict, we take centre of pixel to be the point at
        # which the model believes there is a source

        center_of_patch = patch_centres[patch_id]

        predicted_locs_for_patch_galactic = []

        for pred in predicted_locs_for_patch:

            y_val = pred[0]
            x_val = pred[1]

            # We are calculating location in 128 x 128 instead of 64 x 64 image. Remember to keep this way round - x,y
            # becomes y,x for images.

            pixel_id_val = pixel_id(x_val * 2, y_val * 2, 128)

            l_ps, b_ps = get_lb_from_pixel(pixel_id_val, center_of_patch)

            predicted_locs_for_patch_galactic.append([l_ps, b_ps])

        predicted_locs_for_patch_galactic = np.array(predicted_locs_for_patch_galactic)

        # Convert prediction locs to celestial coordinates

        predicted_locs_for_patch_celestial = SkyCoord(l=predicted_locs_for_patch_galactic[:, 0] * u.degree,
                                                      b=predicted_locs_for_patch_galactic[:, 1] * u.degree,
                                                      frame='galactic').icrs

        predicted_locs_for_patch_celestial = np.array([predicted_locs_for_patch_celestial.ra.value,
                                                       predicted_locs_for_patch_celestial.dec.value]).T

        # FIND SEPARATION OF PREDICTED AND GALACTIC COORDINATES

        # Iterate over predictions and return whether they are AGN, PSR, FAKE - N.B. NEED NUMBER SYSTEM FOR THIS





        # USE SEPARATATION NOT CARTESIAN DISTANCE WHEN DOING RA AND DEC - SEE DISTANCE FUNCTION


        # DON'T FORGET TO NORAMLISE

    return labels


def unek_algorithm(training_data, validation_data, testing_data, use_pretrained_detector=False,
                   use_pretrained_classifier=False,
                   pretrained_model_file="./benchmarks/pre_trained_models/unet.pt"):

    # SEMANTIC SEGMENTATION

    # Pre-trained parameter determines if we should use a U-Net I have already trained on data on create a new U-Net
    # and train it on the training/validation data

    if use_pretrained_detector is False:

        # Train U-Net on data
        model, best_epoch = unet_train(train_data=training_data, test_data=validation_data,
                                       save_file=pretrained_model_file)

        print("BEST EPOCH: {}".format(best_epoch))

        # Save model
        # torch.save(model.state_dict(), pretrained_model_file)

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

    # CLUSTERING (SOURCE LOCALISATION)

    unet_predictions = torch.from_numpy(unet_predictions)

    # Determine the number of sources present within each U-Net segmented image and return the location of their centres
    predicted_source_locations = k_means_clustering(unet_predictions)

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

        # Determine the number of sources present within each U-Net segmented image and return the location of their centres
        train_predicted_source_locations = k_means_clustering(training_unet_predictions)
        validation_predicted_source_locations = k_means_clustering(validation_unet_predictions)

        training_inputs = training_inputs[0].detach().cpu().numpy()
        validation_inputs = validation_inputs[0].detach().cpu().numpy()

        # Get 7 x 7 boxes around each predicted source in each patch
        training_sub_boxes = source_boxes(training_inputs, train_predicted_source_locations)
        validation_sub_boxes = source_boxes(validation_inputs, validation_predicted_source_locations)

        # Get labels for each of the boxes (i.e. AGN, PSR, FAKE)

        train_labels = source_box_labels(patch_ids=training_patch_ids,
                                         predicted_source_locations=train_predicted_source_locations)

        validation_labels = source_box_labels(patch_ids=validation_patch_ids,
                                              predicted_source_locations=validation_predicted_source_locations)



        # Fetch training patch and validation patch data











    # if use_pretrained_classifier is False:
    #
    #     # PREPARE DATA FOR CLASSIFIER
    #
    #     test_patches = np.array(testing_inputs)[0]
    #
    #     num_test_patches = test_patches.shape[0]
    #
    #     classification_patches = []
    #
    #     for n in range(num_test_patches):
    #
    #         predicted_locations_in_patch = predicted_source_locations[n].astype(int)
    #
    #         test_patch = test_patches[n]
    #
    #         for loc in predicted_locations_in_patch:
    #
    #             x, y = loc[0], loc[1]
    #
    #             # Select 7 x 7 grid around predicted location
    #             rows = np.arange(x - 3, x + 4)
    #             cols = np.arange(y - 3, y + 4)
    #
    #             # if cannot create a 7 x 7 grid, ignore during classification
    #             if rows[0] < 0 or cols[0] < 0 or rows[-1] > 64 or cols[-1] > 64:
    #
    #                 continue
    #
    #             else:
    #
    #                 new_patch = test_patch[:, rows, :][:, :, cols]
    #
    #                 classification_patches.append(new_patch)














        # classifier_train()






    # Return the segmented images returned by U-Net and locations of source centres returned by K-means for the TEST
    # data
    return unet_predictions.detach().cpu().numpy(), predicted_source_locations, test_patch_ids


# REFERENCES

# Detach - https://stackoverflow.com/questions/49768306/pytorch-tensor-to-numpy-array
# ID8 - followed their theory/mathematical definition to implement my own version
# Slicing Numpy Arrays - https://stackoverflow.com/questions/4257394/slicing-of-a-numpy-2d-array-or-how-do-i-extract-an-
# mxm-submatrix-from-an-nxn-ar
