from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from ..utils import get_lb_from_pixel, pixel_id
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


def normalise_sub_patches(sub_patches):

    # Assumes sub_patches are passed as array with shape [n, m, 5, 7, 7] where n is number of patches and m is num
    # of sub-patches within patch n

    # Normalises each patch independently - assume format of sub-patches is n patches each with m sub-patches

    num_patches = len(sub_patches)

    num_bins = 5  # sub_patches[0][0].shape[0]

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
    ids_to_remove = [p for p in range(len(patch_ids)) if len(predicted_locations[p]) == 0]

    predicted_locations = [predicted_locations[p] for p in range(len(patch_ids)) if p not in ids_to_remove]
    patches = np.delete(patches, np.array(ids_to_remove).astype(int), 0)
    patch_ids = np.delete(patch_ids, np.array(ids_to_remove).astype(int), 0)

    # Get 7 x 7 boxes around each predicted source in each patch
    sub_boxes = source_boxes(patches, predicted_locations)

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

        for pred_source in range(num_predicted_sources):
            data.append([normalised_sub_boxes[patch][pred_source], vector_labels[patch][pred_source]])

    if len(data) == 0:
        raise RuntimeError("Not enough sources were localised - no data is available for the classifier to train on.")

    if test:

        return data

    else:

        # Reformat as DataSet
        split = Subset(data, np.arange(0, len(data)))

        if shuffle_data:

            batches = DataLoader(split, batch_size=128, shuffle=True)

        else:

            batches = DataLoader(split, batch_size=128, shuffle=False)

        return batches


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

        if len(predicted_source_locations[n]) > 0:

            predicted_locs_for_patch_celestial = (
                image_cartesian_coordinates_to_galactic_coordinates(predicted_source_locations[n],
                                                                    patch_centre=center_of_patch,
                                                                    coordinate_system='C'))

        else:

            predicted_locs_for_patch_celestial = np.array([])

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

    return labels


def str_labels_to_vector_labels(labels):
    num_patches = len(labels)

    vector_labels = []

    for n in range(num_patches):

        patch = labels[n]

        patch_labels = []

        for p in patch:

            if p == "AGN":

                patch_labels.append(np.array([1., 0., 0.]))

            elif p == "PSR":

                patch_labels.append(np.array([0., 1., 0.]))

            else:

                patch_labels.append(np.array([0., 0., 1.]))

        vector_labels.append(np.array(patch_labels))

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
