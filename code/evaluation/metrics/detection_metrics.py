from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import pandas as pd
import warnings
from xml.dom import minidom
from scipy.integrate import quad

def agn_photon_flux(E, E_0, F_0, alpha, beta):

    division = E/E_0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exponent = - alpha - (beta * np.log(division))

    dF_dE = F_0 * np.power(division, exponent)

    return dF_dE

def pulsar_photon_flux(E, F_0, E_0, Gamma, a, b):

    division = E / E_0

    exponent = a * (np.power(E_0, b) - np.power(E, b))

    dF_dE = F_0 * np.power(division, -Gamma) * np.exp(exponent)

    return dF_dE

def integral_photon_flux_agn(pivot_energy, flux_density, spectral_slope, curvature, min_energy=100.0, max_energy=100000.0):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        energy = quad(agn_photon_flux, min_energy, max_energy, args=(pivot_energy, flux_density, spectral_slope, curvature))[0]

    return energy


def integral_photon_flux_pulsar(pivot_energy, flux_density, spectral_slope, exponential_index, exponential_factor,
                       min_energy=100.0, max_energy=100000.0):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Integrate over 0.1 - 100 GeV (100 - 100000 MeV) by default
        energy = quad(pulsar_photon_flux, min_energy, max_energy, args=(flux_density, pivot_energy, spectral_slope,
                                                                        exponential_factor, exponential_index))[0]

    return energy


def get_photon_flux(patch_to_catalog_id):

    catalog_file = "./../data_simulation/simulated_data/catalogs/catalog_{}/"

    num_catalogs = max(patch_to_catalog_id.items())[1] + 1

    catalog_separated_coordinates = []
    catalog_separated_fluxes = []
    catalog_separated_source_ids = []

    for c in range(num_catalogs):

        # Get coordinates and integral photon fluxes of each source in catalog

        agn_coordinates, agn_fluxes, agn_ids = photon_flux_xml_parser(xml_file=catalog_file.format(c + 1) + "/agns.xml", give_ids=True)

        psr_coordinates, psr_fluxes, psr_ids = photon_flux_xml_parser(xml_file=catalog_file.format(c + 1) + "/pulsars.xml", give_ids=True)

        coordinates = np.concatenate((agn_coordinates, psr_coordinates), axis=0)

        fluxes = np.concatenate((agn_fluxes, psr_fluxes), axis=0)

        ids = np.concatenate((agn_ids, psr_ids))

        catalog_separated_coordinates.append(coordinates)

        catalog_separated_fluxes.append(fluxes)

        catalog_separated_source_ids.append(ids)

    return catalog_separated_coordinates, catalog_separated_fluxes, catalog_separated_source_ids

def pulsar_photon_flux(E, F_0, E_0, Gamma, a, b):

    division = E / E_0

    exponent = a * (np.power(E_0, b) - np.power(E, b))

    dF_dE = F_0 * np.power(division, -Gamma) * np.exp(exponent)

    return dF_dE


def photon_flux_xml_parser(xml_file: str, give_ids=False):
    """ Fetches the location, integral photon flux, and (optionally) unique ID of each source stored in a given XML
    file.

    IMPORTANT DIFFERENCE - THIS RETURNS INTEGRAL PHOTON FLUXES RATHER THAN ENERgY INTEGRAL FLUXES.

    WE INTEGRATE OVER 1 ENERGY BIN - 300 - 200000

    """
    # Read XML files to get latitude and longitude of each source (separate into AGN, pulsars, and background - if they
    # are in the same file), as well as the flux of the source

    docs = minidom.parse(xml_file)

    sources = docs.getElementsByTagName("source")

    num_sources = len(sources)

    coordinates = []
    fluxes = []

    # Remove diffuse sources - only processing point sources with this function
    sources = [sources[k] for k in range(num_sources) if sources[k].getAttribute("type") != "DiffuseSource"]

    source_ids = []

    # Parse XML
    for source in sources:

        source_type = source.getAttribute("name")[:3]

        # PARSE SPECTRAL FEATURES AND CALCULATE FLUX

        spectral_model = source.getElementsByTagName("spectrum")[0]

        spectral_parameter_dictionary = dict()

        # Get spectral parameters

        parameters = spectral_model.getElementsByTagName("parameter")

        for param in parameters:
            name = param.getAttribute("name")

            scale = float(param.getAttribute("scale"))

            value = float(param.getAttribute("value"))

            actual_value = scale * value

            spectral_parameter_dictionary[name] = actual_value

        binned_fluxes = []

        if source_type == "AGN":

            # Parse spectral parameters

            # N.B. Integral photon flux is not the same as energy flux
            flux = integral_photon_flux_agn(pivot_energy=spectral_parameter_dictionary["Eb"],
                                            flux_density=spectral_parameter_dictionary["norm"],
                                            spectral_slope=spectral_parameter_dictionary["alpha"],
                                            curvature=spectral_parameter_dictionary["beta"],
                                            min_energy=300,
                                            max_energy=200000)

            binned_fluxes.append(flux)

        elif source_type == "PSR":

            # N.B. Integral photon flux is not the same as energy flux
            flux = integral_photon_flux_pulsar(pivot_energy=spectral_parameter_dictionary["Scale"],
                                               flux_density=spectral_parameter_dictionary["Prefactor"],
                                               spectral_slope=spectral_parameter_dictionary["Index1"],
                                               exponential_index=spectral_parameter_dictionary["Index2"],
                                               exponential_factor=spectral_parameter_dictionary["Expfactor"],
                                               min_energy=300, max_energy=200000)

            binned_fluxes.append(flux)

        else:

            # Unrecognised point source type - allows for debugging when adding in new source types to simulation
            raise TypeError("Cannot recognise source type {}".format(source_type))

        # PARSE SPATIAL PARAMETERS

        spatial_model = source.getElementsByTagName("spatialModel")[0]

        spatial_parameters = spatial_model.getElementsByTagName("parameter")

        coordinate = [0, 0]

        for param in spatial_parameters:
            name = param.getAttribute("name")

            if name == "RA":
                coordinate[0] = float(param.getAttribute("value"))
            else:
                coordinate[1] = float(param.getAttribute("value"))

        coordinates.append(coordinate)

        source_ids.append(source.getAttribute("name"))

        fluxes.append(binned_fluxes)

    # Convert coordinates to np array
    coordinates = np.array(coordinates)

    # Convert fluxes to numpy
    fluxes = np.array(fluxes)

    return (coordinates, fluxes, source_ids) if give_ids else (coordinates, fluxes)


def s90(predicted_source_locations, test_patch_ids, patch_in_catalog,
        distance_threshold=0.3):

    num_catalogs = max(patch_in_catalog.items())[1] + 1

    max_patch_id = max(patch_in_catalog)

    real_coordinates, integral_photon_fluxes, source_ids = get_photon_flux(patch_in_catalog)

    catalog_sep_actual_loc = []
    catalog_sep_predicted_loc = []
    catalog_sep_actual_flux = []
    catalog_sep_predicted_flux = []

    # Select all patches derived from each catalog - important to only select IDs of patches in the test set
    patch_ids_per_catalog = [np.array([p for p in range(max_patch_id) if patch_in_catalog[p] == c and p in test_patch_ids]) for c in range(num_catalogs)]

    for c in range(num_catalogs):

        actual_source_loc_per_patch = []
        actual_source_flux_per_patch = []

        catalog_integral_photon_fluxes = integral_photon_fluxes[c]
        catalog_real_coordinates = real_coordinates[c]

        for p in patch_ids_per_catalog[c]:

            # Open metadata file and select relevant source ids
            df = pd.read_csv("./../data_simulation/simulated_data/patches/patch_{}/metadata.csv".format(p))

            indices = [np.argwhere(source_ids[c] == s).flatten() for s in df["source_id"].to_numpy()]

            p_fluxes = np.array([catalog_integral_photon_fluxes[i[0]] for i in indices])
            p_locs = np.array([catalog_real_coordinates[i[0]] for i in indices])

            actual_source_loc_per_patch.append(p_locs)
            actual_source_flux_per_patch.append(p_fluxes)

        num_patches = len(actual_source_loc_per_patch)

        actual_loc = np.array([actual_source_loc_per_patch[i][j] for i in range(num_patches) for j
                               in range(actual_source_loc_per_patch[i].shape[0])])

        actual_flux = np.array([actual_source_flux_per_patch[i][j] for i in range(num_patches)
                                for j in range(actual_source_flux_per_patch[i].shape[0])]).flatten()

        predicted_loc = np.array([predicted_source_locations[c][i][j] for i in range(len(predicted_source_locations[c]))
                                  for j in range(predicted_source_locations[c][i].shape[0])])

        a_coordinates = SkyCoord(ra=actual_loc[:, 0] * u.degree, dec=actual_loc[:, 1] * u.degree, frame='icrs')

        p_coordinates = SkyCoord(ra=predicted_loc[:, 0] * u.degree, dec=predicted_loc[:, 1] * u.degree,
                                 frame='icrs')

        # Find closest source to each predicted source and set that source's flux as the predicted source's
        idx, _, _ = p_coordinates.match_to_catalog_sky(a_coordinates)

        catalog_sep_actual_flux.append(actual_flux)
        catalog_sep_predicted_flux.append(actual_flux[idx])
        catalog_sep_actual_loc.append(actual_loc)
        catalog_sep_predicted_loc.append(predicted_loc)

    # Sort unique actual photon fluxes in increasing order
    unique_fluxes = np.unique([catalog_sep_actual_flux[i][j] for i in range(num_catalogs)
                               for j in range(catalog_sep_actual_flux[i].shape[0])])
    increasing_order = np.argsort(unique_fluxes)

    for e in increasing_order:

        true_positives = 0
        false_negatives = 0
        false_positives = 0

        min_flux = unique_fluxes[e]

        for c in range(num_catalogs):

            catalog_actual_loc = catalog_sep_actual_loc[c]
            catalog_predicted_loc = catalog_sep_predicted_loc[c]

            catalog_actual_flux = catalog_sep_actual_flux[c]
            catalog_predicted_flux = catalog_sep_predicted_flux[c]

            actual_source_loc_above_min_flux = catalog_actual_loc[(catalog_actual_flux > min_flux)]
            predicted_source_loc_above_min_flux = catalog_predicted_loc[(catalog_predicted_flux > min_flux)]

            if actual_source_loc_above_min_flux.shape[0] == 1 or predicted_source_loc_above_min_flux.shape[0] == 1:
                warnings.warn(
                    "No S90 Metric could be calculated - there was never a minimum SNR above which precision and "
                    "recall were both 0.9.")

                return np.nan

            # actual_source_loc_above_min_flux = np.array([catalog_actual_loc[k] for k in range(num_actual) if catalog_actual_flux[k] > min_flux])
            # predicted_source_loc_above_min_flux = np.array([catalog_predicted_loc[k] for k in range(num_predicted) if catalog_predicted_flux[k] > min_flux])

            actual_source_loc_above_min_flux_celestial = SkyCoord(ra=actual_source_loc_above_min_flux[:, 0] * u.degree,
                                                        dec=actual_source_loc_above_min_flux[:, 1] * u.degree, frame='icrs')

            predicted_source_loc_above_min_flux_celestial = SkyCoord(ra=predicted_source_loc_above_min_flux[:, 0] * u.degree,
                                                        dec=predicted_source_loc_above_min_flux[:, 1] * u.degree, frame='icrs')

            # Find closest source to each actual source to determine true positives and false negatives
            idx, d2d, _ = actual_source_loc_above_min_flux_celestial.match_to_catalog_sky(predicted_source_loc_above_min_flux_celestial)

            true_positives += np.sum(d2d.value < distance_threshold)

            false_negatives += np.sum(d2d.value >= distance_threshold)

            idx, d2d, _ = predicted_source_loc_above_min_flux_celestial.match_to_catalog_sky(
                actual_source_loc_above_min_flux_celestial)

            false_positives += np.sum(d2d.value >= distance_threshold)

        precision = true_positives / (true_positives + false_positives)

        recall = true_positives / (true_positives + false_negatives)

        if precision > 0.9 and recall > 0.9:

            # i.e. the integral photon flux above which precision and recall for source detection is 0.9
            return min_flux

    warnings.warn("No S90 Metric could be calculated - there was never a minimum SNR above which precision and "
                       "recall were both 0.9.")

    return np.nan



# REFERENCES

# Python Documentation - https://docs.python.org/3/library/exceptions.html#RuntimeWarning
# Warnings - https://stackoverflow.com/questions/3891804/raise-warning-in-python-without-interrupting-program
