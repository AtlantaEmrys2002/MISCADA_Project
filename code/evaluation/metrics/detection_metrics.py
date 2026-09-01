from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from scipy.spatial.distance import cdist
import warnings


def s90(actual_source_locations, predicted_source_locations, actual_source_snr, predicted_source_snr,
        distance_threshold=0.3):

    actual_source_locations = np.unique(actual_source_locations, axis=0)
    predicted_source_locations = np.unique(predicted_source_locations, axis=0)

    increasing_order = np.argsort(actual_source_snr)

    for i in increasing_order:

        minimum_snr = actual_source_snr[i]

        # Select all sources with flux above minimum SNR

        mask_actual = (actual_source_snr >= minimum_snr)
        mask_predicted = (predicted_source_snr >= minimum_snr)

        actual_sources = actual_source_locations[mask_actual]
        predicted_sources = predicted_source_locations[mask_predicted]

        predicted_sources_celestial = SkyCoord(ra=predicted_sources[:, 0] * u.degree,
                                     dec=predicted_sources[:, 1] * u.degree, frame='icrs')

        actual_sources_celestial = SkyCoord(ra=actual_sources[:, 0] * u.degree,
                                     dec=actual_sources[:, 1] * u.degree, frame='icrs')

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for a in actual_sources:

            separations = (SkyCoord(ra=a[0] * u.degree, dec=a[1] * u.degree, frame='icrs').
                           separation(predicted_sources_celestial).degree)

            if separations[np.argmin(separations)] < 0.3:

                true_positives += 1

            else:

                false_negatives += 1

        for p in predicted_sources:

            separations = (SkyCoord(ra=p[0] * u.degree, dec=p[1] * u.degree, frame='icrs').
                           separation(actual_sources_celestial).degree)

            if separations[np.argmin(separations)] > 0.3:

                false_positives += 1

        precision = true_positives / (true_positives + false_positives)

        recall = true_positives / (true_positives + false_negatives)

        if precision > 0.9 and recall > 0.9:

            # i.e. the SNR above which precision and recall for source detection is 0.9
            return minimum_snr

    raise RuntimeError("No S90 Metric could be calculated - there was never a minimum SNR above which precision and "
                       "recall were both 0.9.")

# REFERENCES

# Python Documentation - https://docs.python.org/3/library/exceptions.html#RuntimeWarning
# Warnings - https://stackoverflow.com/questions/3891804/raise-warning-in-python-without-interrupting-program
