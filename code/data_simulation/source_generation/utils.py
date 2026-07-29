from math import floor
import numpy as np


def luminosity_function_calculator(energy_fluxes_4fgl, n_min, detection_threshold: np.float64, num_bins: int = 50):
    # The minimum energy flux of our generated sources is an order of magnitude less than the 4FGL
    our_threshold = detection_threshold / 10

    # Following method detailed in ID8

    # Bin 4FGL data
    min_bin_val = np.log10(np.min(energy_fluxes_4fgl))
    max_bin_val = np.log10(np.max(energy_fluxes_4fgl))
    log_linspace = np.linspace(min_bin_val, max_bin_val, num=num_bins)

    bin_edges = 10 ** log_linspace
    counts, bin_intervals = np.histogram(energy_fluxes_4fgl, bins=bin_edges)

    # Calculate width of bins
    bin_width = log_linspace[1] - log_linspace[0]

    # FLAT EXTRAPOLATION TO FAINTER DETECTION THRESHOLD

    # Extend to one order of magnitude less than the detection threshold of the 4FGL (similar premise to ID8) -
    # assume constant below given threshold (not Gaussian)

    # Number of bins between current lowest energy bin and our faint source threshold
    num_extra_bins = floor((np.log10(bin_intervals[0]) - np.log10(our_threshold)) / bin_width)

    extra_intervals = [10 ** (np.log10(bin_intervals[0]) - (bin_width * x)) for x in range(num_extra_bins, 0, -1)]

    # Create extra bin intervals

    # Calculate number of random

    # Find bin with the most AGNs
    peak = np.argmax(counts)

    # For any bin to the right of the peak that has 0 or 1 expected counts, set to 2
    for k in range(peak, len(counts)):
        if counts[k] == 1 or counts[k] == 0:
            counts[k] = 2

    # Generate random numbers for number of energy flux bins to the right of the peak
    n_noise = list(np.random.uniform(low=0.8, high=1.3, size=len(bin_intervals) - 1 - peak))

    # Create some noise in energy bins greater than peak
    for k in range(peak, len(n_noise)):
        counts[k + peak] = counts[k + peak] * n_noise[k]

    bin_intervals = np.array(extra_intervals + list(bin_intervals))

    # Set number of counts equal to peak for original 4FGL bins to the left of the peak
    for k in range(0, peak):
        counts[k] = n_min

    counts = [n_min for _ in range(num_extra_bins)] + list(counts)

    # Convert counts to int
    counts = [int(k) for k in counts]

    peak = peak + num_extra_bins

    return np.array([counts])[0], np.array(bin_intervals), peak


# REFERENCES

# Trapezoidal Rule - https://en.wikipedia.org/wiki/Trapezoidal_rule
