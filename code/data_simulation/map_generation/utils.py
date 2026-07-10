import healpy as hp
import numpy as np
from scipy.integrate import quad


def angle_to_healpix_pixels(coordinates, nside: int):

    # Transform shape from num_sources x 2 to 2 x num_sources
    coordinates = coordinates.T

    # Extract two lists - one of galactic latitudes and one of galactic longitudes
    latitudes = coordinates[0]
    longitudes = coordinates[1]

    pixels = hp.pixelfunc.ang2pix(nside=nside, theta=latitudes, phi=longitudes, lonlat=True)

    return pixels


def get_nside(healpix_exposure_map):

    # Healpix parameter nside can be calculated with this function

    nside = hp.pixelfunc.get_nside(healpix_exposure_map)

    return nside


def exponential_func(energy):
    return energy ** -2.4


def integrate_over_energy(maps, energy_bins, num_bins, energy_weighted=False):

    # if energy_weighted is true, then we use a weighting factor when integrating over the energy range
    # This is used when integrating over exposure maps
    if energy_weighted is True:

        for k in range(len(maps)):

            maps[k] *= (energy_bins[k] ** -2.4)

    # Use the trapezium rule for numerical integration here

    # Always create num_bins + 1 (to get num bins you want - these are the boundaries
    new_energy_bins = np.logspace(np.log(energy_bins[0]), np.log(energy_bins[-1]), num=num_bins + 1, base=np.e)

    # Find the interval in the previous energy_bins in which the new energy bin sits, then calculate the exposure at
    # that energy bin
    new_maps = []

    for n in range(0, num_bins):

        new_energy_value = new_energy_bins[n]

        # Check to see if new energy bin value already in previous energy bins
        already_calculated = np.isclose(energy_bins, new_energy_value)

        if np.any(already_calculated):

            energy_bin = np.argwhere(already_calculated)[0, 0]

            new_maps.append(maps[energy_bin])

        else:

            pos = np.searchsorted(energy_bins, new_energy_value, side="left")

            # Fit straight line between points at energy bins surrounding new energy value
            m = (maps[pos] - maps[pos - 1]) / (energy_bins[pos] - energy_bins[pos - 1])
            c = maps[pos - 1] - (m * energy_bins[pos - 1])

            new_map = (m * new_energy_value) + c

            new_maps.append(new_map)

    integrated_maps = []

    for n in range(0, num_bins):

        lower_new_energy_value = new_energy_bins[n]

        # Check to see if new energy bin value already in previous energy bins
        already_calculated = np.isclose(energy_bins, lower_new_energy_value)

        if np.any(already_calculated):

            lower_addition = np.zeros_like(maps[0])

        else:

            pos = np.searchsorted(energy_bins, lower_new_energy_value, side="left")

            lower_addition = (maps[pos] + new_maps[n]) * (energy_bins[pos] - lower_new_energy_value) / 2

        upper_new_energy_value = new_energy_bins[n + 1]

        # Check to see if new energy bin value already in previous energy bins
        already_calculated = np.isclose(energy_bins, upper_new_energy_value)

        if np.any(already_calculated):

            upper_addition = 0

        else:

            pos = np.searchsorted(energy_bins, upper_new_energy_value, side="left")

            upper_addition = (new_maps[n + 1] + maps[pos - 1]) * (upper_new_energy_value - energy_bins[pos - 1]) / 2

        upper_pos = np.searchsorted(energy_bins, upper_new_energy_value, side="left") - 1  # THIS IS VERY IMPORTANT - THE -1
        lower_pos = np.searchsorted(energy_bins, lower_new_energy_value, side="left")

        middle_addition = np.zeros_like(maps[0])

        for k in range(lower_pos + 1, upper_pos + 1):

            middle_addition += ((maps[k] + maps[k - 1]) * (energy_bins[k] - energy_bins[k - 1])) / 2

        integrated_map = lower_addition + middle_addition + upper_addition

        integrated_maps.append(integrated_map)

    integrated_maps = np.array(integrated_maps)

    if energy_weighted is True:

        integrated_energies = [quad(exponential_func, new_energy_bins[k], new_energy_bins[k + 1])[0] for k in range(num_bins)]

        for x in range(num_bins):

            integrated_maps[x] /= integrated_energies[x]

    # print(integrated_m)

    return integrated_maps, new_energy_bins



# REFERENCES

# New Position - https://math.stackexchange.com/questions/143932/calculate-point-given-x-y-angle-and-distance/
# 3534251#3534251