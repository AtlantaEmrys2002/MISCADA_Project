from astropy.coordinates import SkyCoord
from astropy import units as u
import copy
from functools import partial
import numpy as np
from operator import itemgetter
import pandas as pd
from skimage import feature
from ..utils import get_catalog_data, get_lb_from_pixel, pixel_id
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

    num_sources_to_sample_of_each_type = np.max([agns.shape[0], psrs.shape[0], fakes.shape[0]])

    # Do not want too much oversampling
    num_sources_to_sample_of_each_type = min(20000, num_sources_to_sample_of_each_type)

    print(f"No. AGN: {agns.shape[0]} No. PSR: {psrs.shape[0]} No. FAKE: {fakes.shape[0]}")
    print("No. Each Source in Dataset: {}".format(num_sources_to_sample_of_each_type))

    if num_sources_to_sample_of_each_type > agns.shape[0]:
        indices_to_take = np.random.choice(agns.shape[0], size=num_sources_to_sample_of_each_type - agns.shape[0])
        agns = np.vstack((agns, copy.deepcopy(agns[indices_to_take])))
    elif num_sources_to_sample_of_each_type == agns.shape[0]:
        agns = agns
    else:
        indices_to_remove = np.random.choice(agns.shape[0], size=agns.shape[0] - num_sources_to_sample_of_each_type,
                                             replace=False)
        agns = np.delete(agns, indices_to_remove, axis=0)

    if num_sources_to_sample_of_each_type > psrs.shape[0]:
        indices_to_take = np.random.choice(psrs.shape[0], size=num_sources_to_sample_of_each_type - psrs.shape[0])
        psrs = np.vstack((psrs, copy.deepcopy(psrs[indices_to_take])))
    elif num_sources_to_sample_of_each_type == psrs.shape[0]:
        psrs = psrs
    else:
        indices_to_remove = np.random.choice(psrs.shape[0], size=psrs.shape[0] - num_sources_to_sample_of_each_type,
                                             replace=False)
        psrs = np.delete(psrs, indices_to_remove, axis=0)

    if num_sources_to_sample_of_each_type > fakes.shape[0]:
        indices_to_take = np.random.choice(fakes.shape[0], size=num_sources_to_sample_of_each_type - fakes.shape[0])
        fakes = np.vstack((fakes, copy.deepcopy(fakes[indices_to_take])))
    elif num_sources_to_sample_of_each_type == fakes.shape[0]:
        fakes = fakes
    else:
        indices_to_remove = np.random.choice(fakes.shape[0], size=fakes.shape[0] - num_sources_to_sample_of_each_type,
                                             replace=False)
        fakes = np.delete(fakes, indices_to_remove, axis=0)

    # Combine and return new dataset - no need to shuffle at this point
    data = ([[k, np.array([1., 0., 0.])] for k in agns] + [[k, np.array([0., 1., 0.])] for k in psrs] +
            [[k, np.array([0., 0., 1.])] for k in fakes])

    return data


def extract_features(maps, masks, sigma_min=1, sigma_max=4):

    features_func = partial(
        feature.multiscale_basic_features,
        intensity=True,
        workers=4,
        edges=True,
        texture=True,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        channel_axis=0,
    )

    # Extract local features from training data
    features = np.array([features_func(m) for m in maps])

    # Stitch all training images together - format correct for classifier

    num_features = features[0].shape[-1]

    features = np.reshape(features, shape=(64 * maps.shape[0], 64, num_features))

    labels = np.reshape(masks, shape=(64 * maps.shape[0], 64)) if masks.size != 0 else np.array([])

    return features, labels


def image_cartesian_coordinates_to_galactic_coordinates(coordinates, patch_centre, coordinate_system='G'):
    # Converts x, y index into 64 x 64 image into the longitude and latitude of a source in that image, given
    # the galactic coordinates of the centre of that image.

    # N.B. use 128 x 128 instead of 64 x 64 for conversions.
    new_coords = get_lb_from_pixel(pixel_id(coordinates[:, 0] * 2, coordinates[:, 1] * 2, 128), patch_centre)

    if coordinate_system == 'G':

        return np.array(new_coords).T

    elif coordinate_system == 'C':

        new_coords = SkyCoord(l=new_coords[0] * u.degree, b=new_coords[1] * u.degree, frame='galactic').icrs

        return np.array([new_coords.ra.value, new_coords.dec.value]).T

    else:

        raise TypeError("Coordinate system not supported.")


def prepare_classifier_data(patches, predicted_locations, patch_ids, test=False, real_data=False):

    # REMOVE PATCHES WITH NO PREDICTED SOURCES

    ids_to_remove = [p for p in range(patch_ids.shape[0]) if predicted_locations[p].shape[0] == 0]

    predicted_locations = [predicted_locations[p] for p in range(patch_ids.shape[0]) if p not in ids_to_remove]
    patches = np.delete(patches, np.array(ids_to_remove).astype(int), 0)
    patch_ids = np.delete(patch_ids, np.array(ids_to_remove).astype(int), 0)

    # DRAW 7 x 7 BOUNDING BOXES AROUND EACH DETECTED SOURCE IN PATCH AND LABEL

    sub_boxes, predicted_locations, patch_ids = source_boxes(patches, predicted_locations, patch_ids)

    labels = source_box_labels(patch_ids=patch_ids, predicted_source_locations=predicted_locations, real_data=real_data)

    # Combine data such that each sub-patch is associated with its equivalent label
    num_patches = len(sub_boxes)

    data = [[sub_boxes[p][r], labels[p][r]] for p in range(num_patches) for r in range(sub_boxes[p].shape[0])]

    if len(data) == 0:
        raise RuntimeError("Not enough sources were localised - no data is available for the classifier to train on.")

    return data if test else balance_dataset(data)


def source_boxes(patches, predicted_source_locations, patch_ids):
    # Select 7 x 7 boxes around each patch location - all in Cartesian coordinates

    boxes_for_each_patch = []

    new_predicted_locations = []

    patch_ids_to_remove = []

    for n in range(patches.shape[0]):
        predicted_locations_in_patch = predicted_source_locations[n].astype(int)

        patch = patches[n]

        xs = predicted_locations_in_patch[:, 0]
        ys = predicted_locations_in_patch[:, 1]

        # Check if a 7 x 7 grid can be made with x, y at centre for each source location (i.e. check detected source is
        # not too close to edge of patch).
        mask = np.logical_not(((xs - 3) < 0) | ((xs + 4) > 63) | ((ys - 3) < 0) | ((ys + 4) > 63))

        locs = np.stack((copy.deepcopy(xs[mask]), copy.deepcopy(ys[mask]))).T

        if predicted_locations_in_patch[mask].shape[0] != 0:

            classification_sub_patches = (
                np.array([copy.deepcopy(patch[:, np.arange(l[0] - 3, l[0] + 4), :][:, :, np.arange(l[1] - 3, l[1] + 4)])
                          for l in locs]))

            boxes_for_each_patch.append(classification_sub_patches)

            # As we are not necessarily constructing box around each patch - may disqualify some sources
            new_predicted_locations.append(copy.deepcopy(predicted_locations_in_patch[mask]))

        else:

            patch_ids_to_remove.append(n)

    new_patch_ids = np.delete(patch_ids, np.array(patch_ids_to_remove).astype(int), 0)

    return boxes_for_each_patch, new_predicted_locations, new_patch_ids


def source_box_labels(patch_ids, predicted_source_locations, localisation_threshold:float=0.3, real_data: bool=False):

    # GET PATCH INFORMATION

    directory = "./real_data/real_patches/patches/" if real_data else "./../data_simulation/simulated_data/patches/"

    patches_metadata_file = directory + "patch_metadata.csv"
    individual_patch_metadata_file = directory + "patch_{}/metadata.csv"

    source_information = pd.read_csv(patches_metadata_file)

    # Get centre of each patch
    patch_centres = (
        np.stack((source_information["centre_lon"].to_numpy(), source_information["centre_lat"].to_numpy()),
                 axis=1))

    # Gen number of AGN and pulsars in each patch
    nagn = source_information["num_agn"].to_numpy()
    npsr = source_information["num_psr"].to_numpy()

    if real_data:

        num_catalogs = 1

        patch_catalogs = dict(zip(source_information["patch_id"].to_numpy(), np.full(shape=len(source_information["patch_id"].to_numpy()), fill_value=1)))

    else:

        # Get catalog IDs for each patch
        catalog_ids = (source_information["catalog_id"] + 1).to_numpy()

        # Calculates the number of catalogs that patches are drawn from
        num_catalogs = np.max(catalog_ids)

        # Gives the ID of the catalog that each patch is drawn from
        patch_catalogs = dict(zip(source_information["patch_id"].to_numpy(), catalog_ids))

    if real_data:
        agn_coordinates_per_catalog, pulsar_coordinates_per_catalog = get_catalog_data("/Volumes/T7/data/catalog/4FGL_DR4.fit")

    else:

        catalog_directory = "./../data_simulation/simulated_data/catalogs/catalog_{}/{}.xml"

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

    return str_labels_to_vector_labels(labels)


def str_labels_to_vector_labels(labels):
    num_patches = len(labels)

    str_to_vector = {"AGN": np.array([1., 0., 0.]), "PSR": np.array([0., 1., 0.]), "FAKE": np.array([0., 0., 1.])}

    vector_labels = [np.array(itemgetter(*labels[n])(str_to_vector)) if len(labels[n]) > 0 else np.array([])
                     for n in range(num_patches)]

    vector_labels = [k if k.size != 3 else np.array([k]) for k in vector_labels]

    return vector_labels


def xml_parser_locations(xml_file: str, coordinate_system='G'):

    # GET NAME AND LOCATION OF SOURCE IN SKY

    sources = minidom.parse(xml_file).getElementsByTagName("source")

    # Remove diffuse sources - only processing point sources with this function
    sources = [sources[k] for k in range(len(sources)) if sources[k].getAttribute("type") != "DiffuseSource"]

    source_ids = [source.getAttribute("name") for source in sources]

    parameters = [source.getElementsByTagName("spatialModel")[0].getElementsByTagName("parameter") for source in sources]

    # Get RA and DEC of each source - i.e. parse spatial parameters
    coordinates = np.array([[float(params[0].getAttribute("value")), float(params[1].getAttribute("value"))] for params in parameters])

    if coordinate_system == 'G':

        # Converts to galactic coordinates

        coordinates = SkyCoord(ra=coordinates[:, 0] * u.degree, dec=coordinates[:, 1] * u.degree, frame='icrs').galactic

        return np.array([coordinates.l.value, coordinates.b.value]).T, source_ids

    elif coordinate_system == "C":

        # Returns celestial coordinates

        return coordinates, source_ids

    else:

        raise TypeError("Coordinate system not supported.")

# REFERENCES

# Check if Empty - https://stackoverflow.com/questions/11295609/how-can-i-check-whether-a-numpy-array-is-empty-or-not
# Delete Rows - https://stackoverflow.com/questions/40426697/is-there-any-way-to-delete-the-specific-elements-of-an-
# numpy-array-in-place-in
# Dictionary Mapping - https://stackoverflow.com/questions/63145423/how-to-create-a-numpy-array-based-on-the-values-of-
# another-numpy-array
# Dictionary Mapping - https://stackoverflow.com/questions/18453566/get-list-of-values-for-list-of-keys
# Match vs If-Else - https://www.reddit.com/r/learnpython/comments/1by6vht/should_i_use_match_case_instead_of_if_else/
# Partial Func - https://stackoverflow.com/questions/15331726/how-does-functools-partial-do-what-it-does
# Stacking 2D Arrays - https://stackoverflow.com/questions/72473949/stacking-2d-arrays-into-a-3d-array
# vstack - https://stackoverflow.com/questions/62340746/numpy-stack-multidimensional-arrays
