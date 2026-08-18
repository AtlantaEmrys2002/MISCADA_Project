from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pickle


def percentage_of_4fgl_source_correctly_classified(actual_source_locations, predicted_source_locations, classifications,
                                                   actual_classifications, separation_threshold=0.3):

    # Convert predicted source_centres to sky coordinates
    predicted_source_centres_sky = SkyCoord(ra=predicted_source_locations[:, 0] * u.degree,
                                            dec=predicted_source_locations[:, 1] * u.degree, frame='icrs')

    num_sources_correctly_classified = 0

    classifications = classifications[:, 1]

    num_detected_sources = 0

    for a in range(actual_source_locations.shape[0]):

        actual_loc = actual_source_locations[a]

        coordinate = SkyCoord(ra=actual_loc[0] * u.degree, dec=actual_loc[1] * u.degree, frame='icrs')

        separations = coordinate.separation(predicted_source_centres_sky).degree

        closest_predicted_index = np.argmin(separations)

        separation_from_closest_predicted_source = separations[closest_predicted_index]

        if separation_from_closest_predicted_source < separation_threshold:

            num_detected_sources += 1

            if np.all(np.equal(classifications[closest_predicted_index], actual_classifications[a])):

                num_sources_correctly_classified += 1

    return num_sources_correctly_classified / num_detected_sources


def percentage_of_4fgl_sources_detected(actual_source_locations, predicted_source_locations, separation_threshold=0.3):

    # Convert predicted source_centres to sky coordinates
    predicted_source_centres_sky = SkyCoord(ra=predicted_source_locations[:, 0] * u.degree,
                                            dec=predicted_source_locations[:, 1] * u.degree, frame='icrs')

    num_sources_detected = 0

    for a in actual_source_locations:

        coordinate = SkyCoord(ra=a[0] * u.degree, dec=a[1] * u.degree, frame='icrs')

        separations = coordinate.separation(predicted_source_centres_sky).degree

        separation_from_closest_predicted_source = separations[np.argmin(separations)]

        if separation_from_closest_predicted_source < separation_threshold:
            num_sources_detected += 1

    return num_sources_detected / len(actual_source_locations)


def plot_predictions_actual(actual_coordinates, predicted_coordinates, model):

    # Find out which of the predicted coordinates correspond to actual sources

    # print(actual_coordinates[:10])
    # print(predicted_coordinates[:10])

    # print(np.min(actual_coordinates[:, 0]), np.max(actual_coordinates[:, 0]))
    # print(np.min(predicted_coordinates[:, 0]), np.max(predicted_coordinates[:, 0]))
    #
    # print(np.min(actual_coordinates[:, 1]), np.max(actual_coordinates[:, 1]))
    # print(np.min(predicted_coordinates[:, 1]), np.max(predicted_coordinates[:, 1]))

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

        if separation_from_closest_predicted_source < 0.3:
            correctly_detected_sources.append(a)
        else:
            not_detected.append(a)

    candidate_sources_not_in_4fgl = []

    for p in predicted_coordinates:

        coordinate = SkyCoord(ra=p[0] * u.degree, dec=p[1] * u.degree, frame='icrs')

        separations = coordinate.separation(actual_source_centres_sky).degree

        separation_from_closest_predicted_source = separations[np.argmin(separations)]

        if separation_from_closest_predicted_source > 0.3:
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

        coords = SkyCoord(ra=data_points[x][:, 0] * u.degree, dec=data_points[x][:, 1] * u.degree, frame='icrs')

        ra_rad = coords.ra.wrap_at(180 * u.deg).radian
        dec_rad = coords.dec.radian

        ax.scatter(ra_rad, dec_rad, label=labels[x], color=colours[x], marker=markers[x], s=sizes[x])


    #
    #
    #
    # if correctly_detected_sources.shape[0] != 0:
    #
    #     coords = SkyCoord(ra=correctly_detected_sources[:, 0] * u.degree, dec=correctly_detected_sources[:, 1] * u.degree, frame='icrs')
    #
    #     ra_rad = coords.ra.wrap_at(180 * u.deg).radian
    #     dec_rad = coords.dec.radian
    #
    #     ax.scatter(ra_rad, dec_rad, label="Detected 4FGL Sources", color="red", marker='P', s=2)
    #
    # if not_detected.shape[0] != 0:
    #     coords = SkyCoord(ra=not_detected[:, 0] * u.degree, dec=not_detected[:, 1] * u.degree, frame='icrs')
    #
    #     ra_rad = coords.ra.wrap_at(180 * u.deg).radian
    #     dec_rad = coords.dec.radian
    #
    #     ax.scatter(ra_rad, dec_rad, label="Undetected 4FGL Sources", s=2, color="gray", marker='o')
    # #
    # if candidate_sources_not_in_4fgl.shape[0] != 0:
    #     candidate_sources_not_in_4fgl_ra = Angle(candidate_sources_not_in_4fgl[:, 0] * u.degree).wrap_at(180 * u.degree)
    #     candidate_sources_not_in_4fgl_dec = Angle(candidate_sources_not_in_4fgl[:, 1] * u.degree)
    #
    #     ax.scatter(candidate_sources_not_in_4fgl_ra.radian, candidate_sources_not_in_4fgl_dec.radian,
    #                label="New Candidate Sources", s=2, color="blue", marker='*')
    #








    ax.set_xlabel("RA [$\\degree$]")
    ax.set_ylabel("Dec [$\\degree$]")

    # ax.legend(bbox_to_anchor=(1.05, 1), loc='lower center')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -.3))

    fig.suptitle("Plot of {} Algorithm Applied to Fermi-LAT Observations".format(model))

    fig.tight_layout()

    plt.savefig("./../results/plots/source_discoveries_all_sky/{}_discoveries.png".format(model))


def save_candidate_sources(actual_source_locations, predicted_source_locations, classifications, model,
                           detection_threshold=0.3):

    classifications = classifications[:, 1]

    # Convert predicted source_centres to sky coordinates
    actual_source_centres_sky = SkyCoord(ra=actual_source_locations[:, 0] * u.degree,
                                         dec=actual_source_locations[:, 1] * u.degree, frame='icrs')

    potential_new_sources = []

    for p in range(len(predicted_source_locations)):

        coordinate = SkyCoord(ra=predicted_source_locations[p][0] * u.degree,
                              dec=predicted_source_locations[p][1] * u.degree, frame='icrs')

        separations = coordinate.separation(actual_source_centres_sky).degree

        separation_from_closest_actual_source = separations[np.argmin(separations)]

        if separation_from_closest_actual_source > 0.3:
            potential_new_sources.append([classifications[p], predicted_source_locations[p][0],
                                          predicted_source_locations[p][1]])

    with open("./../results/real/new_sources_{}.data".format(model), 'wb') as f:
        pickle.dump(potential_new_sources, f)

    # np.save("./../results/real/new_sources_{}.npy".format(model), potential_new_sources)


# REFERENCES

# Legend Position - https://stackoverflow.com/questions/77958304/reliably-avoiding-legend-overlapping-other-elements-in-
# matplotlib-pie-chart
# Legend Position - https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot/43439132#43439132
# Pickle Files - https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict-or-any-other-python-
# object
