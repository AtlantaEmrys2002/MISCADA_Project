from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import pandas as pd
from xml.dom import minidom


def actual_source_locations_for_patch(patch_ids, agn_coordinates_per_catalog, pulsar_coordinates_per_catalog,
                                      simulated_data_directory="./../data_simulation/simulated_data/"):

    patches_metadata_file = simulated_data_directory + "patches/patch_metadata.csv"
    source_information = pd.read_csv(patches_metadata_file)

    # Gives the ID of the catalog that each patch is drawn from
    patch_catalogs = dict(zip(source_information["patch_id"].to_numpy(), source_information["catalog_id"].to_numpy()))

    # Gen number of AGN and pulsars in each patch
    nagn = source_information["num_agn"].to_numpy()
    npsr = source_information["num_psr"].to_numpy()

    individual_patch_metadata_file = simulated_data_directory + "patches/patch_{}/metadata.csv"

    actual_agn_locations_in_celestial_per_patch = []
    actual_psr_locations_in_celestial_per_patch = []

    for n in range(patch_ids.shape[0]):

        patch_id = patch_ids[n]

        # Get actual locations of sources in patch
        patch_information = pd.read_csv(individual_patch_metadata_file.format(patch_id))

        # Source IDs of AGN and pulsars in patch
        actual_agn_in_patch = patch_information[patch_information["source_type"] == "AGN"]["source_id"].to_numpy()
        actual_psr_in_patch = patch_information[patch_information["source_type"] == "PSR"]["source_id"].to_numpy()

        # Catalog from which sources in patch are drawn from (minus 1, as this is for indexing)
        catalog_of_patch = patch_catalogs[patch_id]

        # Get celestial locations of AGN in patch
        actual_agn_in_patch_celestial = np.array([agn_coordinates_per_catalog[catalog_of_patch]
                                                   [actual_agn_in_patch[k]] for k in range(nagn[patch_id])])

        actual_psr_in_patch_celestial = np.array([pulsar_coordinates_per_catalog[catalog_of_patch]
                                                  [actual_psr_in_patch[k]] for k in range(npsr[patch_id])])

        if npsr[patch_id] == 0:

            actual_psr_in_patch_celestial = np.array([])

        if nagn[patch_id] == 0:

            actual_agn_in_patch_celestial = np.array([])

        actual_agn_locations_in_celestial_per_patch.append(actual_agn_in_patch_celestial)
        actual_psr_locations_in_celestial_per_patch.append(actual_psr_in_patch_celestial)

    return actual_agn_locations_in_celestial_per_patch, actual_psr_locations_in_celestial_per_patch


def coordinates_of_sources_per_catalog(simulated_data_directory="./../data_simulation/simulated_data/"):

    # READ IN ACTUAL COORDINATES OF EACH SOURCE IN EACH CATALOG

    source_information = pd.read_csv(simulated_data_directory + "patches/patch_metadata.csv")

    # Get catalog IDs for each patch
    catalog_ids = (source_information["catalog_id"] + 1).to_numpy()

    # Calculates the number of catalogs that patches are drawn from
    num_catalogs = np.max(catalog_ids)

    catalog_directory = simulated_data_directory + "catalogs/catalog_{}/{}.xml"

    agn_coords_and_ids = [xml_parser_locations(xml_file=catalog_directory.format(catalog_id, "agns"),
                                               coordinate_system='C') for catalog_id in range(1, num_catalogs + 1)]

    pulsar_coords_and_ids = [xml_parser_locations(xml_file=catalog_directory.format(catalog_id, "pulsars"),
                                                  coordinate_system='C') for catalog_id in range(1, num_catalogs + 1)]

    agn_coordinates_per_catalog = [dict(zip(c[1], c[0])) for c in agn_coords_and_ids]
    pulsar_coordinates_per_catalog = [dict(zip(c[1], c[0])) for c in pulsar_coords_and_ids]

    return agn_coordinates_per_catalog, pulsar_coordinates_per_catalog


def get_patch_centres(patches_metadata_file="./../data_simulation/simulated_data/patches/patch_metadata.csv"):

    source_information = pd.read_csv(patches_metadata_file)

    # Get centre of each patch
    patch_centres = (
        np.stack((source_information["centre_lon"].to_numpy(), source_information["centre_lat"].to_numpy()), axis=1))

    return patch_centres


def vector_labels_to_str(labels):
    str_labels = []

    agn_vector = np.array([1., 0., 0.])
    psr_vector = np.array([0., 1., 0.])

    for p in labels:

        # One-hot encording ensured - some classifications are based on probability

        p_one_hot = np.zeros((3,))

        p_one_hot[np.argmax(p)] = 1

        if np.all(np.equal(p_one_hot, agn_vector)):

            str_labels.append("AGN")

        elif np.all(np.equal(p_one_hot, psr_vector)):

            str_labels.append("PSR")

        else:

            str_labels.append("FAKE")

    return str_labels


def localisation_metadata(patch_ids):

    # patch_ids - numpy array of all patches used to test model

    agn_coordinates, pulsar_coordinates = coordinates_of_sources_per_catalog()

    actual_locations_celestial_agn, actual_locations_celestial_psr = (
        actual_source_locations_for_patch(patch_ids=patch_ids, agn_coordinates_per_catalog=agn_coordinates,
                                          pulsar_coordinates_per_catalog=pulsar_coordinates))

    return actual_locations_celestial_agn, actual_locations_celestial_psr


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
