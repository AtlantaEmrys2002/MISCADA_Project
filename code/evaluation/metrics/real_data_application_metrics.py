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


def plot_predictions_actual(actual_coordinates, predicted_coordinates):

    actual_ra = Angle(actual_coordinates[:, 0] * u.degree).wrap_at(180 * u.degree)

    actual_dec = Angle(actual_coordinates[:, 1] * u.degree)

    predicted_ra = Angle(predicted_coordinates[:, 0] * u.degree).wrap_at(180 * u.degree)

    predicted_dec = Angle(predicted_coordinates[:, 1] * u.degree)

    fig = plt.figure(figsize=(8, 6))

    ax = fig.add_subplot(111, projection="mollweide")

    ax.scatter(actual_ra.radian, actual_dec.radian, label="Actual 4FGL Sources", s=2, color='gray')

    ax.scatter(predicted_ra.radian, predicted_dec.radian, label="Predicted Sources", s=2, color='green')

    ax.legend()

    fig.tight_layout()

    ax.grid(True)

    plt.show()




