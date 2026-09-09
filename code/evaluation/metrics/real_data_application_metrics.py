from astropy.coordinates import SkyCoord
from astropy.table import QTable
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pickle


def get_catalog_data(catalog_file: str = "/Volumes/T7/data/catalog/4FGL_DR4.fit"):
    catalog = QTable.read(catalog_file, format='fits', hdu=1)

    # Select relevant columns
    columns = ("RAJ2000", "DEJ2000")

    catalog = catalog[columns]

    catalog_locations = catalog.to_pandas().to_numpy()

    return catalog_locations


def percentage_of_4fgl_source_correctly_classified(actual_source_locations, predicted_source_locations, classifications,
                                                   actual_classifications, separation_threshold=0.3):

    # print(actual_source_locations.shape, predicted_source_locations.shape)
    #
    # print(np.unique(actual_source_locations, axis=0).shape, np.unique(predicted_source_locations, axis=0).shape)

    # Convert predicted source_centres to sky coordinates
    predicted_source_centres_sky = SkyCoord(ra=predicted_source_locations[:, 0] * u.degree,
                                            dec=predicted_source_locations[:, 1] * u.degree, frame='icrs')

    num_sources_correctly_classified = 0

    # classifications = classifications[:, 1]

    num_detected_sources = 0

    for a in range(actual_source_locations.shape[0]):

        actual_loc = actual_source_locations[a]

        coordinate = SkyCoord(ra=actual_loc[0] * u.degree, dec=actual_loc[1] * u.degree, frame='icrs')

        separations = coordinate.separation(predicted_source_centres_sky).degree

        separation_from_closest_predicted_source = separations[np.argmin(separations)]

        if separation_from_closest_predicted_source < separation_threshold:

            num_detected_sources += 1

            if np.all(np.equal(classifications[np.argmin(separations)], actual_classifications[a])):

                num_sources_correctly_classified += 1

    if num_detected_sources == 0 or num_sources_correctly_classified == 0:

        return 0

    return num_sources_correctly_classified / num_detected_sources


def percentage_of_4fgl_sources_detected(actual_source_locations, predicted_source_locations, separation_threshold=5):

    # Convert predicted source_centres to sky coordinates
    predicted_source_centres_sky = SkyCoord(ra=predicted_source_locations[:, 0] * u.degree,
                                            dec=predicted_source_locations[:, 1] * u.degree, frame='icrs')

    num_sources_detected = 0

    for a in actual_source_locations:

        coordinate = SkyCoord(ra=a[0] * u.degree, dec=a[1] * u.degree, frame='icrs')

        separations = coordinate.separation(predicted_source_centres_sky).degree

        if separations[np.argmin(separations)] < separation_threshold:
            num_sources_detected += 1

    print(num_sources_detected / actual_source_locations.shape[0])

    return num_sources_detected / actual_source_locations.shape[0]


def plot_predictions_actual(actual_coordinates, predicted_coordinates, model, detection_threshold: float=0.3, label:list=[]):

    # Find out which of the predicted coordinates correspond to actual sources

    # Convert predicted source_centres to sky coordinates
    predicted_source_centres_sky = SkyCoord(ra=predicted_coordinates[:, 0] * u.degree,
                                            dec=predicted_coordinates[:, 1] * u.degree, frame='icrs')

    # Convert predicted source_centres to sky coordinates
    actual_source_centres_sky = SkyCoord(ra=actual_coordinates[:, 0] * u.degree,
                                         dec=actual_coordinates[:, 1] * u.degree, frame='icrs')

    correctly_detected_sources = []
    not_detected = []

    for a in actual_coordinates:

        coordinate = SkyCoord(ra=a[0] * u.degree, dec=a[1] * u.degree, frame='icrs')

        separations = coordinate.separation(predicted_source_centres_sky).degree

        separation_from_closest_predicted_source = separations[np.argmin(separations)]

        if separation_from_closest_predicted_source <= detection_threshold:
            correctly_detected_sources.append(a)
        else:
            not_detected.append(a)

    candidate_sources_not_in_4fgl = []

    for p in predicted_coordinates:

        coordinate = SkyCoord(ra=p[0] * u.degree, dec=p[1] * u.degree, frame='icrs')

        separations = coordinate.separation(actual_source_centres_sky).degree

        separation_from_closest_predicted_source = separations[np.argmin(separations)]

        if separation_from_closest_predicted_source > detection_threshold:
            candidate_sources_not_in_4fgl.append(p)

    correctly_detected_sources = np.array(correctly_detected_sources)
    not_detected = np.array(not_detected)
    candidate_sources_not_in_4fgl = np.array(candidate_sources_not_in_4fgl)

    # PLOT

    fig, ax = plt.subplots(figsize=(12, 6.3), subplot_kw=dict(projection="aitoff"))

    ax.grid(True)

    labels = ["Detected 4FGL Sources", "Undetected 4FGL Sources", "New Candidate Sources"]
    colours = ["red", "gray", "blue"]
    markers = ['P', 'o', '*']
    sizes = [5, 2, 50]
    data_points = [correctly_detected_sources, not_detected, candidate_sources_not_in_4fgl]

    for x in range(len(data_points)):

        if data_points[x].size != 0:

            coords = SkyCoord(ra=data_points[x][:, 0] * u.degree, dec=data_points[x][:, 1] * u.degree, frame='icrs')

            ra_rad = coords.ra.wrap_at(180 * u.deg).radian
            dec_rad = coords.dec.radian

            ax.scatter(ra_rad, dec_rad, label=labels[x], color=colours[x], marker=markers[x], s=sizes[x])

    ax.set_xlabel("RA [$\\degree$]", fontsize=13)
    ax.set_ylabel("Dec [$\\degree$]", fontsize=13)

    ax.legend(loc='lower left', bbox_to_anchor=(0.25, -0.4) , fontsize=14)

    ax.tick_params(axis="both", which="major", labelsize=12)

    # bbox_to_anchor=(0.3, -.3)

    fig.suptitle("{}-{}-{} Algorithm \n Applied to Fermi-LAT Observations".format(label[0], label[1], label[2]), fontsize=18)

    fig.tight_layout()

    plt.savefig("./../results/plots/source_discoveries_all_sky/{}_discoveries.png".format(model))

    plt.close()


def save_candidate_sources(actual_source_locations, predicted_source_locations, classifications, model,
                           detection_threshold=0.3):

    # print(classifications.shape)

    # classifications = classifications[:, 1]

    # Convert predicted source_centres to sky coordinates
    actual_source_centres_sky = SkyCoord(ra=actual_source_locations[:, 0] * u.degree,
                                         dec=actual_source_locations[:, 1] * u.degree, frame='icrs')

    potential_new_sources = []

    for p in range(len(predicted_source_locations)):

        coordinate = SkyCoord(ra=predicted_source_locations[p][0] * u.degree,
                              dec=predicted_source_locations[p][1] * u.degree, frame='icrs')

        separations = coordinate.separation(actual_source_centres_sky).degree

        separation_from_closest_actual_source = separations[np.argmin(separations)]

        if separation_from_closest_actual_source > detection_threshold:

            potential_new_sources.append([classifications[p], predicted_source_locations[p][0],
                                          predicted_source_locations[p][1]])

    with open("./../results/real/new_sources_{}.data".format(model), 'wb') as f:
        pickle.dump(potential_new_sources, f)


def compare_candidate_sources():

    models = ["unet_dbscan_cnn", "unet_dbscan_random_forest", "pspnet_dbscan_cnn", "pspnet_dbscan_random_forest",
              "random_forest_dbscan_cnn", "random_forest_dbscan_random_forest", "random_forest_kmeans_cnn",
              "random_forest_kmeans_random_forest"]

    detection_models = ["unet_dbscan", "pspnet_dbscan", "random_forest_dbscan", "random_forest_kmeans"]

    # Only do the detection models
    candidate_locations_per_model = []

    candidate_types_per_model = []

    coordinate_locations_same_classifier_size = []

    model_id = 0

    for model in models:

        with open("./../results/real/new_sources_{}.data".format(model), 'rb') as f:
            candidates = pickle.load(f)

            candidate_types_per_model.append(np.array([c[0] for c in candidates]))

            # Convert locations into sky coordinates
            coordinates = np.array([[c[1], c[2]] for c in candidates])

            coordinates = SkyCoord(ra=coordinates[:, 0] * u.degree, dec=coordinates[:, 1] * u.degree, frame='icrs')

            if model_id % 2 == 0:

                candidate_locations_per_model.append(coordinates)

            coordinate_locations_same_classifier_size.append(coordinates)

        model_id += 1

    print(sum([len(k) for k in candidate_locations_per_model]))

    # Sources detected by two or more models

    sources_detected_by_two_or_more_models = []

    for m in range(len(detection_models)):

        for n in range(m + 1, len(detection_models)):

            # Check if each source detected by two or more

            idx, d2d, _ = candidate_locations_per_model[m].match_to_catalog_sky(candidate_locations_per_model[n])

            # Select all sources that are close

            tmp = candidate_locations_per_model[m][d2d.value < 0.3]

            for t in tmp:
                sources_detected_by_two_or_more_models.append(t)

    # combine into one sky coordinate

    sources_detected_by_two_or_more_models = np.array([[k.ra.value, k.dec.value] for k in sources_detected_by_two_or_more_models])

    num_sources = sources_detected_by_two_or_more_models.shape[0]

    distinct_candidates = []

    for a in range(sources_detected_by_two_or_more_models.shape[0]):

        sources_detected_by_two_or_more_models_coords = SkyCoord(
            ra=sources_detected_by_two_or_more_models[:, 0][[k for k in range(num_sources) if k != a]] * u.degree,
            dec=sources_detected_by_two_or_more_models[:, 1][[k for k in range(num_sources) if k != a]] * u.degree, frame='icrs')

        coord = SkyCoord(ra=sources_detected_by_two_or_more_models[a][0] * u.degree,
                         dec=sources_detected_by_two_or_more_models[a][1] * u.degree, frame='icrs')

        if np.min(coord.separation(sources_detected_by_two_or_more_models_coords).value) > 0.3:
            distinct_candidates.append(sources_detected_by_two_or_more_models[a])
        else:

            if np.argmin(coord.separation(sources_detected_by_two_or_more_models_coords).value) >= a:
                distinct_candidates.append(sources_detected_by_two_or_more_models[a])

    distinct_candidates = np.unique(np.array(distinct_candidates), axis=0)

    print(distinct_candidates.shape[0])

    # Read in locations of 4FGL sources
    locations_4fgl = get_catalog_data()

    locations_4fgl = SkyCoord(ra=locations_4fgl[:, 0] * u.degree,
                                                             dec=locations_4fgl[:, 1] * u.degree, frame='icrs')

    locations_new = SkyCoord(ra=distinct_candidates[:, 0] * u.degree,
                                                             dec=distinct_candidates[:, 1] * u.degree, frame='icrs')

    idx, d2d, _ = locations_new.match_to_catalog_sky(locations_4fgl)

    no_counterparts = distinct_candidates[d2d.value > 0.3]

    types_of_no_counterparts = []

    # Majority vote wins - if tied, then say unassociated
    for a in no_counterparts:

        model_id = 0

        associated_types = []

        for m in coordinate_locations_same_classifier_size:

            # FIND CLOSEST SOURCE IN EACH MODEL AND ASSOCIATED CLASSIFICATION
            source = SkyCoord(ra=a[0] * u.degree, dec=a[1] * u.degree, frame='icrs')

            separations = source.separation(m)

            minimum_sep = np.argmin(separations.value)

            if separations.value[minimum_sep] < 0.3:

                associated_types.append(candidate_types_per_model[model_id][minimum_sep])

            model_id += 1

        associated_types = np.array(associated_types)

        num_votes_agn = int(np.sum([1 for k in associated_types if np.all(np.equal(k, np.array([1., 0., 0.])))]))
        num_votes_psr = int(np.sum([1 for k in associated_types if np.all(np.equal(k, np.array([0., 1., 0.])))]))
        num_votes_fake = int(np.sum([1 for k in associated_types if np.all(np.equal(k, np.array([0., 0., 1.])))]))

        if (num_votes_fake > num_votes_agn) and (num_votes_fake > num_votes_psr):
            types_of_no_counterparts.append("FAKE")
        elif (num_votes_agn > num_votes_fake) and (num_votes_agn > num_votes_psr):
            types_of_no_counterparts.append("AGN")
        elif (num_votes_psr > num_votes_fake) and (num_votes_psr > num_votes_agn):
            types_of_no_counterparts.append("PSR")
        else:
            types_of_no_counterparts.append("UNASSOC")

    print(len(no_counterparts))

    print("No. AGN: {}".format(sum([1 for k in types_of_no_counterparts if k == "AGN"])))
    print("No. PSR: {}".format(sum([1 for k in types_of_no_counterparts if k == "PSR"])))
    print("No. FAKE: {}".format(sum([1 for k in types_of_no_counterparts if k == "FAKE"])))
    print("No. UNASSOC: {}".format(sum([1 for k in types_of_no_counterparts if k == "UNASSOC"])))

    agn_coordinates = np.array([no_counterparts[k] for k in range(len(no_counterparts)) if types_of_no_counterparts[k] == "AGN"]).T
    psr_coordinates = np.array([no_counterparts[k] for k in range(len(no_counterparts)) if types_of_no_counterparts[k] == "PSR"]).T
    unassoc_coordinates = np.array([no_counterparts[k] for k in range(len(no_counterparts)) if types_of_no_counterparts[k] == "UNASSOC"]).T

    # PLOT

    fig, ax = plt.subplots(figsize=(12, 6.3), subplot_kw=dict(projection="aitoff"))

    ax.grid(True)

    labels = ["AGN", "PSR", "Unassociated"]
    colours = ["red", "blue", "gray"]
    markers = ['*', '*', 'o']
    sizes = [50, 50, 10]

    count = 0

    for x in [agn_coordinates, psr_coordinates, unassoc_coordinates]:

        coords = SkyCoord(ra=x[0] * u.degree, dec=x[1] * u.degree, frame='icrs')

        print(len(coords))

        ra_rad = coords.ra.wrap_at(180 * u.deg).radian
        dec_rad = coords.dec.radian

        ax.scatter(ra_rad, dec_rad, label=labels[count], color=colours[count], marker=markers[count], s=sizes[count])

        count += 1

    ax.set_xlabel("RA [$\\degree$]", fontsize=13)
    ax.set_ylabel("Dec [$\\degree$]", fontsize=13)

    ax.legend(loc='lower left', bbox_to_anchor=(0.25, -0.4), fontsize=14)

    ax.tick_params(axis="both", which="major", labelsize=12)

    fig.suptitle("Candidate $\gamma$-ray Sources Detected with \n Novel Source Extraction Algorithms", fontsize=18)

    fig.tight_layout()

    plt.savefig("./../results/plots/candidate_sources.png".format(model))

    plt.close()

# REFERENCES

# Add Rows - https://stackoverflow.com/questions/3881453/numpy-add-row-to-array
# Legend Position - https://stackoverflow.com/questions/77958304/reliably-avoiding-legend-overlapping-other-elements-in-
# matplotlib-pie-chart
# Legend Position - https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot/43439132#43439132
# Nearest Neighbours - https://stackoverflow.com/questions/12923586/nearest-neighbor-search-python
# Nearest Neighbours - https://stackoverflow.com/questions/52166769/astropy-coordinates-second-nearest-neighbour
# Pickle Files - https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict-or-any-other-python-
# object
