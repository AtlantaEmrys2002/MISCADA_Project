import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from . components.classification_algorithms import classifier_train
from . components.clustering_algorithms import k_means_clustering
from . components.segmentation_algorithms import UNET, unet_train
import numpy as np
from . utils import get_lb_from_pixel, pixel_id
import torch
from xml.dom import minidom


def image_cartesian_coordinates_to_galactic_coordinates(coordinates, patch_centre, coordinate_system='G'):
    # Converts x, y index into 64 x 64 image into the longitude and latitude of a source in that image, given
    # the galactic coordinates of the centre of that image.

    y_vals = coordinates[:, 0]
    x_vals = coordinates[:, 1]

    # We are calculating location in 128 x 128 instead of 64 x 64 image. Remember to keep this way round - x,y
    # becomes y,x for images. Then flattening the image into 1D array - this pixel value gives index into the
    # 1D array.
    pixel_id_values = pixel_id(x_vals * 2, y_vals * 2, 128)

    new_coords = get_lb_from_pixel(pixel_id_values, patch_centre)

    if coordinate_system == 'G':

        return np.array(new_coords).T

    elif coordinate_system == 'C':

        new_coords = (SkyCoord(l=new_coords[0] * u.degree, b=new_coords[1] * u.degree, frame='galactic').
                      transform_to('icrs'))

        return np.array([new_coords.ra.value, new_coords.dec.value]).T

    else:

        raise TypeError("Coordinate system not supported.")


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


def source_box_labels(patch_ids, predicted_source_locations, localisation_threshold=0.3):

    catalog_directory = "./../data_simulation/simulated_data/catalogs/catalog_{}/{}.xml"
    patches_metadata_file = "./../data_simulation/simulated_data/patches/patch_metadata.csv"
    individual_patch_metadata_file = "./../data_simulation/simulated_data/patches/patch_{}/metadata.csv"

    num_patches = patch_ids.shape[0]

    source_information = pd.read_csv(patches_metadata_file)

    # Get catalog IDs for each patch
    catalog_ids = (source_information["catalog_id"] + 1).to_numpy()

    # Calculates the number of catalogs that patches are drawn from
    num_catalogs = np.max(catalog_ids)

    # Gives the ID of the catalog that each patch is drawn from
    patch_catalogs = dict(zip(source_information["patch_id"].to_numpy(), catalog_ids))

    # Get centre of each patch
    patch_centres = (
        np.stack((source_information["centre_lon"].to_numpy(), source_information["centre_lat"].to_numpy()), axis=1))

    # Gen number of AGN and pulsars in each patch
    nagn = source_information["num_agn"].to_numpy()
    npsr = source_information["num_psr"].to_numpy()

    agn_coordinates_per_catalog = []
    pulsar_coordinates_per_catalog = []

    for catalog_id in range(1, num_catalogs + 1):

        actual_agn_coordinates, actual_agn_ids = (
            xml_parser_locations(xml_file=catalog_directory.format(catalog_id, "agns"), coordinate_system='C'))

        actual_psr_coordinates, actual_psr_ids = (
            xml_parser_locations(xml_file=catalog_directory.format(catalog_id, "pulsars"), coordinate_system='C'))

        agn_coordinates_per_catalog.append(dict(zip(actual_agn_ids, actual_agn_coordinates)))
        pulsar_coordinates_per_catalog.append(dict(zip(actual_psr_ids, actual_psr_coordinates)))

    labels = []

    for n in range(num_patches):

        patch_id = patch_ids[n]

        # Number of each type of source in patch
        num_agn_in_patch = nagn[patch_id]
        num_psr_in_patch = npsr[patch_id]

        # Centre coordinates of patch in galactic coordiante system
        center_of_patch = patch_centres[patch_id]

        # Catalog from which sources in patch are drawn from (minus 1, as this is for indexing)
        catalog_of_patch = patch_catalogs[patch_id] - 1

        # Get actual locations of sources in patch
        patch_information = pd.read_csv(individual_patch_metadata_file.format(patch_id))

        # Source IDs of AGN and pulsars in patch
        actual_agn_in_patch = patch_information[patch_information["source_type"] == "AGN"]["source_id"].to_numpy()
        actual_psr_in_patch = patch_information[patch_information["source_type"] == "PSR"]["source_id"].to_numpy()

        # N.B. FOR ABOVE - ITERATE OVER PREDICTED SOURCES

        # Get celestial locations of AGN in patch
        actual_agn_locations_in_celestial = np.array([agn_coordinates_per_catalog[catalog_of_patch]
                                                      [actual_agn_in_patch[k]] for k in
                                                      range(num_agn_in_patch)])

        actual_psr_locations_in_celestial = np.array([pulsar_coordinates_per_catalog[catalog_of_patch]
                                                      [actual_psr_in_patch[k]] for k in
                                                      range(num_psr_in_patch)])

        # Convert to SkyCoords

        if num_agn_in_patch > 0:

            actual_agn_locations_in_celestial = SkyCoord(ra=actual_agn_locations_in_celestial[:, 0] * u.degree,
                                                         dec=actual_agn_locations_in_celestial[:, 1] * u.degree,
                                                         frame='icrs')

        if num_psr_in_patch > 0:

            actual_psr_locations_in_celestial = SkyCoord(ra=actual_psr_locations_in_celestial[:, 0] * u.degree,
                                                         dec=actual_psr_locations_in_celestial[:, 1] * u.degree,
                                                         frame='icrs')

        # Convert the predicted locations of sources from coordinates within 64 x 64 patch to RA-DEC
        predicted_locs_for_patch_celestial = (
            image_cartesian_coordinates_to_galactic_coordinates(predicted_source_locations[n],
                                                                patch_centre=center_of_patch, coordinate_system='C'))

        # FIND SEPARATION OF PREDICTED AND GALACTIC COORDINATES

        # Iterate over predictions and return whether they are AGN, PSR, FAKE - N.B. NEED NUMBER SYSTEM FOR THIS

        labels_for_patch = []

        for pred in predicted_locs_for_patch_celestial:

            pred_skycoord = SkyCoord(ra=pred[0] * u.degree, dec=pred[1] * u.degree, frame="icrs")

            # Calculate distance between this predicted source and all other sources in the source

            if num_agn_in_patch > 0:

                separation_agn = pred_skycoord.separation(actual_agn_locations_in_celestial).degree

            else:

                separation_agn = np.array([])

            if num_psr_in_patch > 0:

                separation_psr = pred_skycoord.separation(actual_psr_locations_in_celestial).degree

            else:

                separation_psr = np.array([])

            if separation_agn.shape[0] == 0 and separation_psr.shape[0] == 0:

                labels_for_patch.append("FAKE")

            else:

                if separation_agn.shape[0] == 0:

                    closest_psr = np.argmin(separation_psr)

                    if separation_psr[closest_psr] < localisation_threshold:

                        labels_for_patch.append("PSR")

                    else:

                        labels_for_patch.append("FAKE")

                elif separation_psr.shape[0] == 0:

                    closest_agn = np.argmin(separation_agn)

                    if separation_agn[closest_agn] < localisation_threshold:

                        labels_for_patch.append("AGN")

                    else:

                        labels_for_patch.append("FAKE")

                else:

                    closest_psr = np.argmin(separation_psr)
                    closest_agn = np.argmin(separation_agn)

                    if (separation_psr[closest_psr] < separation_agn[closest_agn] and separation_psr[closest_psr] <
                            localisation_threshold):

                        labels_for_patch.append("PSR")

                    elif (separation_agn[closest_agn] < separation_psr[closest_psr] and separation_agn[closest_agn] <
                          localisation_threshold):

                        labels_for_patch.append("AGN")

                    else:

                        labels_for_patch.append("FAKE")

        labels.append(np.array(labels_for_patch))












    # lABELS SHOULD BE VECTORS WITH 1 BEING SORUCE TYPE AND 0 BEING NOT OSURCE TYPE, E.G. [1, 0, 0] is AGN, [0, 1, 0] is
    # PSR and [0, 0, 1] is FAKE












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

                        # DON'T FORGET TO NORAMLISE THE IMAGES - THIS HASN'T BEEN DONE YET!!!!!
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
