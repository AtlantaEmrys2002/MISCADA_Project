from astropy.coordinates import SkyCoord
from astropy import units as u
import copy
import numpy as np
from operator import itemgetter
import pandas as pd
from torch.utils.data import DataLoader, Subset
from ..utils import get_lb_from_pixel, pixel_id
import warnings
from xml.dom import minidom


def balance_dataset(data):

    num_sources = len(data)

    agns = np.array([data[k][0] for k in range(num_sources) if np.all(np.equal(data[k][1], np.array([1., 0., 0.])))])
    psrs = np.array([data[k][0] for k in range(num_sources) if np.all(np.equal(data[k][1], np.array([0., 1., 0.])))])
    fakes = np.array([data[k][0] for k in range(num_sources) if np.all(np.equal(data[k][1], np.array([0., 0., 1.])))])

    if agns.shape[0] == 0 or psrs.shape[0] == 0:
        raise ValueError("No successful detections of specific source type. AGN in Dataset: {}. PSR in Dataset: {}".
                         format(agns.shape[0], psrs.shape[0]))

    if fakes.shape[0] == 0:
        warnings.warn("No FAKE sources were detected - only genuine AGN and PSR. Segmentation and localisation was "
                      "highly successful.")

    if agns.shape[0] > psrs.shape[0] and agns.shape[0] > fakes.shape[0]:
        num_sources_to_sample_of_each_type = agns.shape[0]
    elif psrs.shape[0] > agns.shape[0] and psrs.shape[0] > fakes.shape[0]:
        num_sources_to_sample_of_each_type = psrs.shape[0]
    else:
        num_sources_to_sample_of_each_type = fakes.shape[0]

    print("No. Each Source in Dataset: {}".format(num_sources_to_sample_of_each_type))

    num_agns_to_sample = num_sources_to_sample_of_each_type - agns.shape[0]
    num_psrs_to_sample = num_sources_to_sample_of_each_type - psrs.shape[0]
    num_fakes_to_sample = num_sources_to_sample_of_each_type - fakes.shape[0]

    if num_agns_to_sample > 0:
        indices_to_take = np.random.choice(agns.shape[0], size=num_agns_to_sample)
        agns = np.vstack((agns, copy.deepcopy(agns[indices_to_take])))

    if num_psrs_to_sample > 0:
        indices_to_take = np.random.choice(psrs.shape[0], size=num_psrs_to_sample)
        psrs = np.vstack((psrs, copy.deepcopy(psrs[indices_to_take])))

    if num_fakes_to_sample > 0 and fakes.shape[0] > 0:
        indices_to_take = np.random.choice(fakes.shape[0], size=num_fakes_to_sample)
        fakes = np.stack((fakes, copy.deepcopy(fakes[indices_to_take])))

    # Combine and return new dataset - no need to shuffle, as we are calling this before the end of
    # prepare_classifier_data()

    data = []

    for k in agns:
        data.append([k, np.array([1., 0., 0.])])

    for k in psrs:
        data.append([k, np.array([0., 1., 0.])])

    for k in fakes:
        data.append([k, np.array([0., 0., 1.])])

    return data


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


def ml_segmentation_data_prep(train_data, validation_data, test_data):
    # Formats count map patches and masks for compatibility with ML (as opposed to DL) segmentation algorithms.

    # EXTRACT TRAINING DATA
    train_patch_ids = []
    training_maps = []
    training_masks = []

    for i, x in enumerate(train_data):

        for k in range(x[0].shape[0]):
            train_patch_ids.append(x[0][k])

        for k in range(x[1].shape[0]):
            training_maps.append(x[1][k])

        for j in range(x[2].shape[0]):
            training_masks.append(x[2][j][0])

    train_patch_ids = np.array(train_patch_ids).astype(int)
    training_maps = np.array(training_maps)
    training_masks = np.array(training_masks)

    # EXTRACT VALIDATION DATA
    validation_patch_ids = []
    validation_maps = []
    validation_masks = []

    for i, x in enumerate(validation_data):

        for k in range(x[0].shape[0]):
            validation_patch_ids.append(x[0][k])

        for k in range(x[1].shape[0]):
            validation_maps.append(x[1][k])

        for j in range(x[2].shape[0]):
            validation_masks.append(x[2][j][0])

    validation_patch_ids = np.array(validation_patch_ids).astype(int)
    validation_maps = np.array(validation_maps)
    validation_masks = np.array(validation_masks)

    # EXTRACT TEST DATA
    test_patch_ids = np.array([k[0] for k in test_data])
    testing_maps = np.array([k[1] for k in test_data])
    testing_masks = np.array([k[2][0] for k in test_data])

    return (train_patch_ids, training_maps, training_masks, validation_patch_ids, validation_maps, validation_masks,
            test_patch_ids, testing_maps, testing_masks)


def normalise_sub_patches(sub_patches, num_bins: int = 5):
    # Assumes sub_patches are passed as array with shape [n, m, 5, 7, 7] where n is number of patches and m is num
    # of sub-patches within patch n

    # Normalises each patch independently - assume format of sub-patches is n patches each with m sub-patches

    num_patches = len(sub_patches)

    normalised_sub_patches_arr = []

    for p in range(num_patches):

        num_sub_patches = sub_patches[p].shape[0]

        current_patch = sub_patches[p]

        sub_patches_in_patch = []

        for s in range(num_sub_patches):
            sigmas = np.array([np.std(current_patch[s][b]) for b in range(num_bins)])
            means = np.array([np.mean(current_patch[s][b]) for b in range(num_bins)])

            sub_patch = np.array([(current_patch[s][b] - means[b]) / sigmas[b] if sigmas[b] != 0 else
                                  current_patch[s][b] for b in range(num_bins)])

            sub_patches_in_patch.append(sub_patch)

        normalised_sub_patches_arr.append(np.array(sub_patches_in_patch))

    return normalised_sub_patches_arr


def prepare_classifier_data(patches, predicted_locations, patch_ids, shuffle_data=True, test=False):
    # Remove all patches with no predicted sources
    # ids_to_remove = [p for p in range(patch_ids.shape[0]) if len(predicted_locations[p]) == 0]
    ids_to_remove = [p for p in range(patch_ids.shape[0]) if predicted_locations[p].shape[0] == 0]

    predicted_locations = [predicted_locations[p] for p in range(patch_ids.shape[0]) if p not in ids_to_remove]
    patches = np.delete(patches, np.array(ids_to_remove).astype(int), 0)
    patch_ids = np.delete(patch_ids, np.array(ids_to_remove).astype(int), 0)

    # Get 7 x 7 boxes around each predicted source in each patch
    sub_boxes, predicted_locations = source_boxes(patches, predicted_locations)

    # Normalise each patch (normalise each energy bin separately)
    normalised_sub_boxes = normalise_sub_patches(sub_boxes)

    # Get labels for each patch (i.e. AGN, PSR, FAKE)
    labels = source_box_labels(patch_ids=patch_ids, predicted_source_locations=predicted_locations)

    # Format labels such that they are in vector format, e.g. AGN is equivalent to [1., 0., 0.]
    vector_labels = str_labels_to_vector_labels(labels)

    data = []

    # Combine data such that each sub-patch is associated with its equivalent label

    num_patches = len(sub_boxes)

    for patch in range(num_patches):

        num_predicted_sources = len(sub_boxes[patch])

        # N.B. conversion to float 32 from float 64 - Apple GPUs cannot work with float64
        for pred_source in range(num_predicted_sources):
            if isinstance(vector_labels[patch][pred_source], np.ndarray):

                data.append([normalised_sub_boxes[patch][pred_source].astype(np.float32),
                             vector_labels[patch][pred_source].astype(np.float32)])

    # print([str(label) for patch in labels for label in patch])

    if len(data) == 0:
        raise RuntimeError("Not enough sources were localised - no data is available for the classifier to train on.")

    # Balance dataset

    if test:

        return data

    else:

        # Only balance dataset if training or validating (not when testing or applying to real data).
        data = balance_dataset(data)

        # Reformat as DataSet
        split = Subset(data, np.arange(0, len(data)))

        batches = DataLoader(split, batch_size=128, shuffle=shuffle_data)

        return batches


def source_boxes(patches, predicted_source_locations):
    # Select 7 x 7 boxes around each patch location - all in Cartesian coordinates

    boxes_for_each_patch = []

    new_predicted_locations = []

    for n in range(patches.shape[0]):
        predicted_locations_in_patch = predicted_source_locations[n].astype(int)

        patch = patches[n]

        xs = predicted_locations_in_patch[:, 0]
        ys = predicted_locations_in_patch[:, 1]

        # Check if a 7 x 7 grid can be made with x, y at centre for each source location (i.e. check detected source is
        # not too close to edge of patch).
        mask = np.logical_not(((xs - 3) < 0) | ((xs + 4) > 63) | ((ys - 3) < 0) | ((ys + 4) > 63))

        locs = np.stack((copy.deepcopy(xs[mask]), copy.deepcopy(ys[mask]))).T

        classification_sub_patches = (
            np.array([copy.deepcopy(patch[:, np.arange(l[0] - 3, l[0] + 4), :][:, :, np.arange(l[1] - 3, l[1] + 4)])
                      for l in locs]))

        boxes_for_each_patch.append(classification_sub_patches)

        # As we are not necessarily constructing box around each patch - may disqualify some sources
        new_predicted_locations.append(copy.deepcopy(predicted_locations_in_patch[mask]))

    return boxes_for_each_patch, new_predicted_locations


def source_box_labels(patch_ids, predicted_source_locations, localisation_threshold=0.3):
    catalog_directory = "./../data_simulation/simulated_data/catalogs/catalog_{}/{}.xml"
    patches_metadata_file = "./../data_simulation/simulated_data/patches/patch_metadata.csv"
    individual_patch_metadata_file = "./../data_simulation/simulated_data/patches/patch_{}/metadata.csv"

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

    for n in range(patch_ids.shape[0]):

        patch_id = patch_ids[n]

        # Number of each type of source in patch
        num_agn_in_patch = nagn[patch_id]
        num_psr_in_patch = npsr[patch_id]

        # Centre coordinates of patch in galactic coordinate system
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

        labels_for_patch = []

        if len(predicted_source_locations[n]) > 0:

            # Convert the predicted locations of sources from coordinates within 64 x 64 patch to RA-DEC
            predicted_locs_for_patch_celestial = (
                image_cartesian_coordinates_to_galactic_coordinates(predicted_source_locations[n],
                                                                    patch_centre=center_of_patch,
                                                                    coordinate_system='C'))

            # FIND SEPARATION OF PREDICTED AND GALACTIC COORDINATES

            # Iterate over predictions and return whether they are AGN, PSR, FAKE - N.B. NEED NUMBER SYSTEM FOR THIS

            for pred in predicted_locs_for_patch_celestial:

                pred_skycoord = SkyCoord(ra=pred[0] * u.degree, dec=pred[1] * u.degree, frame="icrs")

                # Calculate distance between this predicted source and all other sources in the source

                if num_agn_in_patch == 0 and num_psr_in_patch == 0:
                    labels_for_patch.append("FAKE")

                elif num_agn_in_patch == 0:
                    separation_psr = pred_skycoord.separation(actual_psr_locations_in_celestial).degree

                    if separation_psr[np.argmin(separation_psr)] < localisation_threshold:
                        labels_for_patch.append("PSR")
                    else:
                        labels_for_patch.append("FAKE")

                elif num_psr_in_patch == 0:
                    separation_agn = pred_skycoord.separation(actual_agn_locations_in_celestial).degree

                    if separation_agn[np.argmin(separation_agn)] < localisation_threshold:
                        labels_for_patch.append("AGN")
                    else:
                        labels_for_patch.append("FAKE")
                else:

                    separation_psr = pred_skycoord.separation(actual_psr_locations_in_celestial).degree
                    separation_agn = pred_skycoord.separation(actual_agn_locations_in_celestial).degree

                    closest_agn = separation_agn[np.argmin(separation_agn)]
                    closest_psr = separation_psr[np.argmin(separation_psr)]

                    if closest_agn < closest_psr and closest_agn < localisation_threshold:
                        labels_for_patch.append("AGN")
                    elif closest_psr < closest_agn and closest_psr < localisation_threshold:
                        labels_for_patch.append("PSR")
                    else:
                        labels_for_patch.append("FAKE")

        labels.append(np.array(labels_for_patch))

    return labels


def str_labels_to_vector_labels(labels):
    num_patches = len(labels)

    str_to_vector = {"AGN": np.array([1., 0., 0.]), "PSR": np.array([0., 1., 0.]), "FAKE": np.array([0., 0., 1.])}

    vector_labels = [np.array(itemgetter(*labels[n])(str_to_vector)) if len(labels[n]) > 0 else np.array([])
                     for n in range(num_patches)]

    return vector_labels


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

        # Converts to galactic coordinates

        # Get coordinates into numpy array then separate into list of latitudes and longitudes

        coordinates = SkyCoord(ra=coordinates[:, 0] * u.degree, dec=coordinates[:, 1] * u.degree, frame='icrs').galactic

        coordinates = np.array([coordinates.l.value, coordinates.b.value]).T

        return coordinates, source_ids

    elif coordinate_system == "C":

        # Returns in celestial coordinates

        return coordinates, source_ids

    else:

        raise TypeError("Coordinate system not supported.")

# REFERENCES

# Check if Empty - https://stackoverflow.com/questions/11295609/how-can-i-check-whether-a-numpy-array-is-empty-or-not
# Data Loader Formatting - https://discuss.pytorch.org/t/how-to-get-input-data-from-a-dataloader/121216
# Delete Rows - https://stackoverflow.com/questions/40426697/is-there-any-way-to-delete-the-specific-elements-of-an-
# numpy-array-in-place-in
# Dictionary Mapping - https://stackoverflow.com/questions/63145423/how-to-create-a-numpy-array-based-on-the-values-of-
# another-numpy-array
# Dictionary Mapping - https://stackoverflow.com/questions/18453566/get-list-of-values-for-list-of-keys
# Error Debugging - https://stackoverflow.com/questions/76494637/pytorch-dataloader-runtimeerror-stack-expects-each-
# tensor-to-be-equal-size
# Match vs If-Else - https://www.reddit.com/r/learnpython/comments/1by6vht/should_i_use_match_case_instead_of_if_else/
# Partial Func - https://stackoverflow.com/questions/15331726/how-does-functools-partial-do-what-it-does
# Stacking 2D Arrays - https://stackoverflow.com/questions/72473949/stacking-2d-arrays-into-a-3d-array
# Vstack - https://stackoverflow.com/questions/62340746/numpy-stack-multidimensional-arrays
