from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np


def chamfer_separation(actual_source_centres, predicted_source_centres):
    # ID11 - evaluates localisation power of faint source detection (how close predicted point sources are to actual
    # point source). I have adapted Chamfer distance to work with RA/DEC instead of cartesian coordinates in image.
    # In fact, this metric is more similar to ID12 - which deems nearest neighbours separated by 1-2 arcmin to be a good
    # match)

    # It is REALLY important to notice the difference between Chamfer distance and this new metric which relies on
    # angular separation

    # A = Predicted source centres
    # B = Ground truth source centres

    if actual_source_centres.shape[0] == 0 or predicted_source_centres.shape[0] == 0:
        return np.inf

    # Convert predicted source_centres to sky coordinates
    predicted_source_centres_sky = SkyCoord(ra=predicted_source_centres[:, 0] * u.degree,
                                            dec=predicted_source_centres[:, 1] * u.degree, frame='icrs')

    actual_source_centres_sky = SkyCoord(ra=actual_source_centres[:, 0] * u.degree,
                                         dec=actual_source_centres[:, 1] * u.degree, frame='icrs')

    dist_ab = 0

    for a in actual_source_centres:

        coordinate = SkyCoord(ra=a[0] * u.degree, dec=a[1] * u.degree, frame='icrs')

        separations = coordinate.separation(predicted_source_centres_sky).degree

        dist_ab += separations[np.argmin(separations)]

    dist_ba = 0

    for p in predicted_source_centres:

        coordinate = SkyCoord(ra=p[0] * u.degree, dec=p[1] * u.degree, frame='icrs')

        separations = coordinate.separation(actual_source_centres_sky).degree

        dist_ba += separations[np.argmin(separations)]

    dist_chamfer = dist_ab + dist_ba

    return dist_chamfer


# CHANGED TO 0.46 (CHANGE BACK TO 0.3)
def num_sources_correctly_detected(actual_source_centres, predicted_source_centres, separation_threshold=0.3):
    # This determines how many sources in patch are correctly detected (it does not matter what source type they are,
    # but whether they are correctly detected and localised as a source). A fraction is returned - the proportion of
    # sources in patch actually detected.

    num_catalogs = len(actual_source_centres)

    total_sources = sum([actual_source_centres[b].shape[0] for b in range(num_catalogs)])

    num_sources_detected = 0

    for b in range(num_catalogs):

        if predicted_source_centres[b].shape[0] == 0:

            continue

        else:

            # Convert predicted source_centres to sky coordinates
            predicted_source_centres_sky = SkyCoord(ra=predicted_source_centres[b][:, 0] * u.degree,
                                                    dec=predicted_source_centres[b][:, 1] * u.degree, frame='icrs')

            for a in actual_source_centres[b]:

                separations = (SkyCoord(ra=a[0] * u.degree, dec=a[1] * u.degree, frame='icrs').
                               separation(predicted_source_centres_sky).degree)

                separation_from_closest_predicted_source = separations[np.argmin(separations)]

                if separation_from_closest_predicted_source < separation_threshold:
                    num_sources_detected += 1

    return num_sources_detected / total_sources

# REFERENCES

# Chamfer Distance - https://medium.com/@sim30217/chamfer-distance-4207955e8612
# Distance Calculations - https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-
# numpy
# Mesh grids - https://www.geeksforgeeks.org/python/numpy-meshgrid-function/
# Scipy Documentation - https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.
# spatial.distance.cdist
