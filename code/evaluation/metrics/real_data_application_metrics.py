from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np


def percentage_of_4fgl_sources_detected(actual_source_locations, predicted_source_locations, separation_threshold=0.3):

    # Convert predicted source_centres to sky coordinates
    predicted_source_centres_sky = SkyCoord(ra=predicted_source_locations[:, 0] * u.degree,
                                            dec=predicted_source_locations[:, 1] * u.degree, frame='icrs')

    num_sources_detected = 0

    for a in actual_source_locations:

        coordinate = SkyCoord(ra=a[0] * u.degree, dec=a[1] * u.degree, frame='icrs')

        separation_from_closest_predicted_source = np.argmin(coordinate.separation(predicted_source_centres_sky).degree)

        if separation_from_closest_predicted_source < separation_threshold:
            num_sources_detected += 1

    return num_sources_detected / len(actual_source_locations)


def plot_predictions_actual(actual_coordinates, predicted_coordinates, model):

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

        separation_from_closest_predicted_source = np.argmin(coordinate.separation(predicted_source_centres_sky).degree)

        if separation_from_closest_predicted_source < 0.3:
            correctly_detected_sources.append(a)
        else:
            not_detected.append(a)

    candidate_sources_not_in_4fgl = []

    for p in predicted_coordinates:

        coordinate = SkyCoord(ra=p[0] * u.degree, dec=p[1] * u.degree, frame='icrs')

        separation_from_closest_predicted_source = np.argmin(coordinate.separation(actual_source_centres_sky).degree)

        if separation_from_closest_predicted_source > 0.3:

            candidate_sources_not_in_4fgl.append(p)

    correctly_detected_sources = np.array(correctly_detected_sources)
    not_detected = np.array(not_detected)
    candidate_sources_not_in_4fgl = np.array(candidate_sources_not_in_4fgl)

    fig = plt.figure(figsize=(8, 7))

    ax = fig.add_subplot(111, projection="mollweide")

    if correctly_detected_sources.shape[0] != 0:

        correctly_detected_ra = Angle(correctly_detected_sources[:, 0] * u.degree).wrap_at(180 * u.degree)
        correctly_detected_dec = Angle(correctly_detected_sources[:, 1] * u.degree)

        ax.scatter(correctly_detected_ra.radian, correctly_detected_dec.radian, label="Detected 4FGL Sources",
                   color="green", marker='P')

    if not_detected.shape[0] != 0:

        not_detected_ra = Angle(not_detected[:, 0] * u.degree).wrap_at(180 * u.degree)
        not_detected_dec = Angle(not_detected[:, 1] * u.degree)

        ax.scatter(not_detected_ra.radian, not_detected_dec.radian, label="Undetected 4FGL Sources", s=2, color="gray",
                   marker='o')

    if candidate_sources_not_in_4fgl.shape[0] != 0:

        candidate_sources_not_in_4fgl_ra = Angle(candidate_sources_not_in_4fgl[:, 0] * u.degree).wrap_at(180 * u.degree)
        candidate_sources_not_in_4fgl_dec = Angle(candidate_sources_not_in_4fgl[:, 1] * u.degree)

        ax.scatter(candidate_sources_not_in_4fgl_ra.radian, candidate_sources_not_in_4fgl_dec.radian,
                   label="New Candidate Sources", s=2, color="gold", marker='*')

    ax.set_xlabel("RA [$\\degree$]")
    ax.set_ylabel("Dec [$\\degree$]")

    # ax.legend(bbox_to_anchor=(1.05, 1), loc='lower center')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -.3))

    ax.grid(True)

    fig.suptitle("Plot of {} Algorithm Applied to Fermi-LAT Observations".format(model))

    fig.tight_layout()

    plt.savefig("./../results/plots/source_discoveries_all_sky/{}_discoveries.png".format(model))




    # actual_ra = Angle(actual_coordinates[:, 0] * u.degree).wrap_at(180 * u.degree)
    #
    # actual_dec = Angle(actual_coordinates[:, 1] * u.degree)
    #
    # predicted_ra = Angle(predicted_coordinates[:, 0] * u.degree).wrap_at(180 * u.degree)
    #
    # predicted_dec = Angle(predicted_coordinates[:, 1] * u.degree)
    #
    # fig = plt.figure(figsize=(8, 6))
    #
    # ax = fig.add_subplot(111, projection="mollweide")
    #
    # ax.scatter(not_detected_ra.radian, not_detected_dec.radian, label="Undetected 4FGL Sources", s=2, color='gray')
    # ax.scatter(candidate_sources_not_in_4fgl_ra.radian, candidate_sources_not_in_4fgl_dec.radian,
    #            label="Candidate Sources (not in 4FGL)", s=2, color='yellow', marker=(10, 1, 0))
    #
    # # ax.scatter(actual_ra.radian, actual_dec.radian, label="Actual 4FGL Sources", s=2, color='gray')
    # #
    # # ax.scatter(predicted_ra.radian, predicted_dec.radian, label="Predicted Sources", s=2, color='green')
    #
    # ax.legend()
    #
    # fig.tight_layout()
    #
    # ax.grid(True)
    #
    # plt.show()




