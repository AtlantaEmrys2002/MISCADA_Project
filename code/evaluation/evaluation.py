from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable

from utils import integral_photon_flux_agn, integral_photon_flux_pulsar
from metrics.utils import image_cartesian_coordinates_to_physical_coordinates
from metrics.classification_metrics import classification_confusion_matrix
from metrics.localisation_metrics import chamfer_separation, num_sources_correctly_detected
from metrics.real_data_application_metrics import (percentage_of_4fgl_sources_detected, plot_predictions_actual,
                                                   percentage_of_4fgl_source_correctly_classifier)
from metrics.segmentation_metrics import (binary_balanced_accuracy, dice_coefficient, segmentation_precision,
                                          segmentation_recall)
import numpy as np
from pathlib import Path
import pickle
from read_write_functions import get_patch_centres, localisation_metadata, vector_labels_to_str


def evaluate_classifiers(actual_class, predicted_class, method_name, directory):
    # Plot confusion matrices
    classification_confusion_matrix(ground_truth=actual_class, predicted=predicted_class, classifier_name=method_name,
                                    directory=directory)


def evaluate_localisation(actual_source_centers, predicted_source_centers):
    num_patches = len(actual_source_centers)

    average_chamfer_distance = sum([chamfer_separation(actual_source_centers[p], predicted_source_centers[p]) for p in
                                    range(num_patches)]) / num_patches

    average_percentage_of_sources_detected = (
            sum([num_sources_correctly_detected(actual_source_locations[p], predicted_locations_celestial[p]) for p in
                 range(num_patches)]) / num_patches)

    return average_chamfer_distance, average_percentage_of_sources_detected


def evaluate_detection(actual_segmentations, predicted_segmentations):
    # N.B. actual segmentations and predicted segmentations should be numpy arrays

    num_patches = actual_segmentations.shape[0]

    average_binary_balanced_accuracy = sum([binary_balanced_accuracy(actual_segmentations[p],
                                                                     predicted_segmentations[p])
                                            for p in range(num_patches)]) / num_patches

    average_dice_coefficient = sum([dice_coefficient(actual_segmentations[p], predicted_segmentations[p]) for p in
                                    range(num_patches)]) / num_patches

    average_precision = sum([segmentation_precision(actual_segmentations[p], predicted_segmentations[p]) for p in
                             range(num_patches)]) / num_patches

    average_recall = sum([segmentation_recall(actual_segmentations[p], predicted_segmentations[p]) for p in
                          range(num_patches)]) / num_patches

    return average_binary_balanced_accuracy, average_dice_coefficient, average_precision, average_recall


def get_classified_patches(predicted_locations_in_real_data_raw):
    # see if it would have been classified or not (i.e. if it was too close to the edge. If it was too close to
    # edge of 64 x 64 image, then remove.

    predicted_locations_in_real_data = []

    for p in range(len(predicted_locations_in_real_data_raw)):

        tmp = []

        for loc in predicted_locations_in_real_data_raw[p]:

            x, y = loc[0], loc[1]

            # Select 7 x 7 grid around predicted location
            rows = np.arange(x - 3, x + 4)
            cols = np.arange(y - 3, y + 4)

            # if cannot create a 7 x 7 grid, ignore during classification
            if rows[0] < 0 or cols[0] < 0 or rows[-1] > 64 or cols[-1] > 64:

                continue

            else:

                tmp.append(loc)

        predicted_locations_in_real_data.append(np.array(tmp))

    return predicted_locations_in_real_data


def evaluate_on_real_data(file_4fgl, real_data_results_directory, model):
    patch_centres = get_patch_centres(patches_metadata_file="./../source_extractors/real_data/real_patches/patches/"
                                                            "patch_metadata.csv")

    catalog = QTable.read(file_4fgl, format='fits', hdu=1)

    # Select relevant columns
    columns = ("Source_Name", "Pivot_Energy", "LP_Flux_Density", "PLEC_Flux_Density", "LP_Index", "LP_beta",
               "PLEC_IndexS", "PLEC_Exp_Index", "PLEC_ExpfactorS", "CLASS1", "GLAT", "GLON")

    catalog = catalog[columns]

    # Reformat CLASS1 column - remove empty spaces and make all lower case
    catalog["CLASS1"] = np.asarray([k.decode('utf-8').strip().lower() for k in catalog["CLASS1"].value.filled('-')])

    # Reformat Source_name column - remove empty spaces and make all lower case
    catalog["Source_Name"] = np.asarray([k.decode('utf-8').strip().lower()[5:] for k in catalog["Source_Name"].value])

    # Select all rows that describe pulsars
    pulsar_mask = (catalog["CLASS1"] == "psr")

    # Select all rows that describe AGN
    agn_mask = np.isin(catalog["CLASS1"].data, np.array(["bcu", "sey", "ssrq", "bll", "fsrq", "rdg", "nlsy1", "agn"]))

    # Delete unnecessary column
    catalog.remove_column("CLASS1")

    agn_data = catalog[agn_mask].copy()
    pulsar_data = catalog[pulsar_mask].copy()

    agn_data = (agn_data["Source_Name", "LP_Flux_Density", "Pivot_Energy", "LP_Index", "LP_beta", "GLAT", "GLON"].
                to_pandas())

    pulsar_data = pulsar_data[
        ("Source_Name", "PLEC_Flux_Density", "Pivot_Energy", "PLEC_IndexS", "PLEC_Exp_Index", "PLEC_ExpfactorS", "GLAT",
         "GLON")].to_pandas()

    # Calculate the name, flux, and celestial coordinates of each source

    agn_ids = agn_data["Source_Name"].to_numpy()

    agn_pivot_energies = agn_data["Pivot_Energy"].to_numpy()
    agn_flux_densities = agn_data["LP_Flux_Density"].to_numpy()
    agn_spectral_slopes = agn_data["LP_Index"].to_numpy()  # alpha
    agn_curvatures = agn_data["LP_beta"].to_numpy()  # beta

    pulsar_pivot_energies = pulsar_data["Pivot_Energy"].to_numpy()
    pulsar_flux_densities = pulsar_data["PLEC_Flux_Density"].to_numpy()
    pulsar_spectral_slopes = pulsar_data["PLEC_IndexS"].to_numpy()  # gamma
    pulsar_exponential_indices = pulsar_data["PLEC_Exp_Index"].to_numpy()
    pulsar_exponential_factors = pulsar_data["PLEC_ExpfactorS"].to_numpy()

    agn_integral_photon_fluxes = np.array([integral_photon_flux_agn(pivot_energy=agn_pivot_energies[s],
                                                                    flux_density=agn_flux_densities[s],
                                                                    spectral_slope=agn_spectral_slopes[s],
                                                                    curvature=agn_curvatures[s], min_energy=300.,
                                                                    max_energy=200000.)
                                           for s in range(len(agn_pivot_energies))])

    pulsar_integral_photon_fluxes = np.array([integral_photon_flux_pulsar(pivot_energy=pulsar_pivot_energies[s],
                                                                          flux_density=pulsar_flux_densities[s],
                                                                          spectral_slope=pulsar_spectral_slopes[s],
                                                                          exponential_index=
                                                                          pulsar_exponential_indices[s],
                                                                          exponential_factor=
                                                                          pulsar_exponential_factors[s])
                                              for s in range(len(pulsar_pivot_energies))])

    # N.B. THESE ARE PHOTON FLUXES, NOT ENERGY FLUXES!!!!!

    agn_glon = agn_data["GLON"].to_numpy()
    agn_glat = agn_data["GLAT"].to_numpy()

    psr_glon = pulsar_data["GLON"].to_numpy()
    psr_glat = pulsar_data["GLAT"].to_numpy()

    agn_celestial_coordinates = SkyCoord(l=agn_glon * u.degree, b=agn_glat * u.degree, frame='galactic').icrs

    agn_celestial_coordinates = np.array([agn_celestial_coordinates.ra.value, agn_celestial_coordinates.dec.value]).T

    psr_celestial_coordinates = SkyCoord(l=psr_glon * u.degree, b=psr_glat * u.degree, frame='galactic').icrs

    psr_celestial_coordinates = np.array([psr_celestial_coordinates.ra.value, psr_celestial_coordinates.dec.value]).T

    actual_source_locations_4fgl = np.vstack((agn_celestial_coordinates, psr_celestial_coordinates))

    actual_source_types = np.vstack((np.array([np.array([1., 0., 0.,]) for _ in range(len(agn_glon))]),
                                     np.array([np.array([0., 1., 0.]) for _ in range(len(psr_glon))])))

    # RESULTS

    # Get the positions of each of the detected sources in the real data

    # Get predicted locations from model (in x, y coordinates in 64 x 64 image)
    with open("./../results/real/{}/predicted_locations.data".format(model), 'rb') as f:
        predicted_locations_in_real_data_raw = pickle.load(f)

    # only take sources that are not "on the edge" of 64 x 64 patches
    predicted_locations_in_real_data = get_classified_patches(predicted_locations_in_real_data_raw)

    # Slightly different indexing here - assume that always 768 patches
    predicted_locations_in_real_data_celestial = [
        image_cartesian_coordinates_to_physical_coordinates(coordinates=predicted_locations_in_real_data[p],
                                                            patch_centre=patch_centres[p],
                                                            coordinate_system='C') if
        len(predicted_locations_in_real_data[p]) != 0 else np.array([]) for p in range(768)]

    new = []

    for k in range(768):
        for p in predicted_locations_in_real_data_celestial[k]:
            new.append(p)

    predicted_locations_in_real_data_celestial = np.array(new)

    # EVALUATE NUM oF 4FGL SOURCES DETECTED

    frac_of_4fgl_sources_detected = percentage_of_4fgl_sources_detected(
        actual_source_locations=actual_source_locations_4fgl,
        predicted_source_locations=predicted_locations_in_real_data_celestial)

    Path("./../results/plots/source_discoveries_all_sky/").mkdir(parents=True, exist_ok=True)


    # ADD THIS LINE BACK AT THE END!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # plot_predictions_actual(actual_coordinates=actual_source_locations_4fgl,
    #                         predicted_coordinates=predicted_locations_in_real_data_celestial, model=model)

    classifications = np.load("./../results/real/{}/classifications.npy".format(model))


    # THIS FRACTION IS THE NUMBER OF SOURCES CORRECTLY CLASSIFIED OF THE NUMBER OF SOURCES CORRECTLY DETECTED
    frac_correct_classed_sources = (
        percentage_of_4fgl_source_correctly_classifier(actual_source_locations=actual_source_locations_4fgl,
                                                       predicted_source_locations=
                                                       predicted_locations_in_real_data_celestial, classifications=
                                                       classifications), actual_classifications=actual_source_types)

    # print(classifications.shape)
    #
    # print(len(predicted_locations_in_real_data_celestial), len(classifications))





if __name__ == "__main__":

    # TAKE INPUTS (RECOMMENDED READ IN FILE)

    # Add name of models here
    # models = ["UNEK", "UNEB"]
    models = ["UNEK"]

    detectors = {"UNEK": "U-NET", "UNEB": "U-NET"}
    localisers = {"UNEK": "K-Means", "UNEB": "Blob Detection"}
    classifiers = {"UNEK": "CNN", "UNEB": "CNN"}

    results = []

    csv_headers = ("id,detection_algorithm,localisation_algorithm,classification_algorithm,"
                   "segmentation_balanced_binary_accuracy,segmentation_dive_coefficient,segmentation_precision,"
                   "segmentation_recall,chamfer_separation,frac_sources_detected\n")

    # Set up file
    file = open("./../results/results.csv", "w+")
    file.writelines(csv_headers)
    file.close()

    # Create directory in which to save plots
    plot_directory = "./../results/plots"
    Path(plot_directory).mkdir(parents=True, exist_ok=True)

    model_id = 0

    # # READ IN ACTUAL COORDINATES OF EACH SOURCE IN EACH CATALOG

    patch_centres = get_patch_centres(
        patches_metadata_file="./../data_simulation/simulated_data/patches/patch_metadata.csv")

    # CALCULATE METRICS FOR EACH SOURCE EXTRACTION ALGORITHM

    for m in models:

        # IDs of patches used to test model
        patch_ids = np.load("./../results/{}/patch_ids.npy".format(m))

        # DETECTION (SEGMENTATION) EVALUATION

        segments = np.load("./../results/{}/segmentations.npy".format(m))

        actual_segments = segments[:, 0]

        predicted_segments = segments[:, 1]

        av_bin_balanced_acc, av_dice, av_prec, av_rec = evaluate_detection(actual_segmentations=actual_segments,
                                                                           predicted_segmentations=predicted_segments)

        # LOCALISATION EVALUATION

        # Get predicted locations from model (in x, y coordinates in 64 x 64 image)
        with open("./../results/{}/predicted_locations.data".format(m), 'rb') as f:

            predicted_locations = pickle.load(f)

        # Convert predicted locations to celestial RA/DEC coordinates

        predicted_locations_celestial = [
            image_cartesian_coordinates_to_physical_coordinates(coordinates=predicted_locations[p],
                                                                patch_centre=patch_centres[patch_ids[p]],
                                                                coordinate_system='C') for p in range(len(patch_ids))]

        # Get locations of actual sources (in celestial coordinates) within each patch
        actual_agn_locations_celestial, actual_psr_locations_celestial = localisation_metadata(patch_ids=patch_ids)

        actual_source_locations = []

        for n in range(actual_segments.shape[0]):

            if actual_agn_locations_celestial[n].size != 0 and actual_psr_locations_celestial[n].size != 0:
                actual_source_locations_for_patch = np.vstack((actual_agn_locations_celestial[n],
                                                               actual_psr_locations_celestial[n]))
            elif actual_psr_locations_celestial[n].size == 0:
                actual_source_locations_for_patch = actual_agn_locations_celestial[n]
            else:
                actual_source_locations_for_patch = actual_psr_locations_celestial[n]

            actual_source_locations.append(actual_source_locations_for_patch)

        # Evaluate localisation
        av_chamfer_distance, av_frac_sources_detected = (
            evaluate_localisation(actual_source_centers=actual_source_locations, predicted_source_centers=
            predicted_locations_celestial))

        # CLASSIFICATION EVALUATION

        classifications = np.load("./../results/{}/classifications.npy".format(m))

        actual_classes, predicted_classes = vector_labels_to_str(classifications[:, 0]), vector_labels_to_str(
            classifications[:, 1])

        # This does not take into account any spatial distributions (at the moment!!!!!!)
        evaluate_classifiers(actual_class=actual_classes, predicted_class=predicted_classes, method_name=m,
                             directory=plot_directory)

        evaluate_on_real_data(file_4fgl="/Volumes/T7/data/catalog/4FGL_DR4.fit",
                              real_data_results_directory="./../results/real", model=m)

        # pred_classifications_for_each_patch = []
        # actual_classifications_for_each_patch = []
        #
        # starting_index = 0
        #
        # for p in range(patch_ids.shape[0]):
        #
        #     num_predicted_sources_in_patch = predicted_locations[p].shape[0]
        #
        #     pred_classes = predicted_classes[starting_index: starting_index + num_predicted_sources_in_patch]
        #
        #     actual_classes = actual_classes[starting_index: starting_index + num_predicted_sources_in_patch]
        #
        #     pred_classifications_for_each_patch.append(pred_classes)
        #
        #     starting_index += num_predicted_sources_in_patch

        # SAVE RESULTS

        results.append(f"{model_id},{detectors[m]},{localisers[m]},{classifiers[m]},{av_bin_balanced_acc},{av_dice},"
                       f"{av_prec},{av_rec},{av_chamfer_distance},{av_frac_sources_detected}\n")

        model_id += 1

    # EVALUATE DIFFERENT STAGES FOR EACH MODEL WITH METRICS

    # SAVE RESULTS TO FILE

    file = open("./../results/results.csv", "a")
    file.writelines(results)
    file.close()

# NEED TO MAKE SURE PLOTS DIRECTORY EXISTS AND CONFUSION MATRIX FILE EXISTS (SEE CLASSIFICATION METRICS FILE)

# FOR s90 - figure out the number of degrees (suggested is 0.3 degrees) before reject as true source representation and
# convert to pixels on image

# REFERENCES

# Cartesian Products - https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-
# single-array-of-2d-points
