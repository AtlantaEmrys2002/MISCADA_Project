import healpy as hp


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


def integrate_over_energy(maps, energy_bins, num_bins, energy_weighted=False):

    # print(len(maps))
    # print(len(energy_bins))

    # Use the trapezium rule for numerical integration here

    import numpy as np

    # Always create num_bins + 1 (to get num bins you want - these are the boundaries
    new_energy_bins = np.logspace(np.log(energy_bins[0]), np.log(energy_bins[-1]), num=num_bins + 1, base=np.e)

    # Find the interval in the previous energy_bins in which the new energy bin sits, then calculate the exposure at
    # that energy bin
    new_maps = []

    # Will be the same - based on logarithmic spacing
    # new_maps.append(maps[0])

    # print(energy_bins)
    # print(new_energy_bins)

    for n in range(0, num_bins):

        new_energy_value = new_energy_bins[n]

        # Check to see if new energy bin value already in previous energy bins
        already_calculated = np.isclose(energy_bins, new_energy_value)

        if np.any(already_calculated):

            energy_bin = np.argwhere(already_calculated)[0, 0]

            new_maps.append(maps[energy_bin])

        else:

            pos = np.searchsorted(energy_bins, new_energy_value, side="left")

            # print(pos)
            #
            # print(energy_bins[pos - 1], new_energy_value, energy_bins[pos])

            # Fit straight line between points at energy bins surrounding new energy value
            m = (maps[pos] - maps[pos - 1]) / (energy_bins[pos] - energy_bins[pos - 1])
            c = maps[pos - 1] - (m * energy_bins[pos - 1])

            new_map = (m * new_energy_value) + c

            new_maps.append(new_map)

    # print(energy_bins)
    # print(new_energy_bins)

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

        upper_pos = np.searchsorted(energy_bins, upper_new_energy_value, side="left") - 1 # THIS IS VERY IMPORTANT - THE -1
        lower_pos = pos = np.searchsorted(energy_bins, lower_new_energy_value, side="left")

        print('\n')

        middle_addition = np.zeros_like(maps[0])

        for k in range(lower_pos + 1, upper_pos + 1):

            # print(energy_bins[k - 1], energy_bins[k])

            middle_addition += ((maps[k] + maps[k - 1]) * (energy_bins[k] - energy_bins[k - 1])) / 2

        integrated_map = lower_addition + middle_addition + upper_addition

        integrated_maps.append(integrated_map)




        # print(energy_bins[upper_pos])

        # print(lower_new_energy_value, energy_bins[lower_pos], energy_bins[upper_pos], upper_new_energy_value)

        #
        # # Chose int to ensure consitency
        # pos = np.searchsorted(energy_bins.astype(int), new, side="right")
        #
        # # draw all lines between exposure interval values surrounding it (e.g. ///////) - y is exposure values and x
        # # are energy values
        #
        # m = (maps[pos + 1] - maps[pos]) / (energy_bins[pos + 1] - energy_bins[pos])
        #
        # c = maps[pos] - (m * energy_bins[pos])
        #
        # map_at_new_bin = (m * new) + c
        #
        # new_maps.append(map_at_new_bin)

    # Final exposure map at upper bound - will be the same - based on logarithmic spacing
    # new_maps.append(maps[-1])
    #
    # for n in range(num_bins):
    #
    #     lower = new_energy_bins[n]
    #     upper = new_energy_bins[n + 1]
    #     map_at_new_bin_lower = new_maps[n]
    #     map_at_new_bin_upper = new_maps[n + 1]
    #
    #     # print(energy_bins)
    #     #
    #     # print(lower, upper)
    #
    #     pos_lower = np.searchsorted(energy_bins.astype(int), lower, side="right")
    #     pos_upper = np.searchsorted(energy_bins.astype(int), upper, side="right")
    #
    #     if pos_upper >= len(maps):
    #         pos_upper = len(maps) - 1
    #
    #     # print(pos_lower, pos_upper)
    #
    #     # area under curve between this newly approximated exposure and the next known exposure bin (from the energy
    #     # bins fed to this function)
    #
    #     # Just for this area between this new energy and the next known energy - will need to do for other side and also
    #     # all bins inbetween
    #     lower_addition = (energy_bins[pos_lower + 1] - lower) * (map_at_new_bin_lower + maps[pos_lower + 1]) / 2
    #
    #
    #
    #     # THIS IS NOT RIGHT
    #     upper_addition = (upper - energy_bins[pos_upper]) * (map_at_new_bin_upper + maps[pos_upper]) / 2
    #
    #     print(upper)
    #     print(energy_bins[pos_upper])
    #    #  print(upper - energy_bins[pos_upper])
    #
    #     # Middle between all bins between
    #
    #     # middle_addition = np.zeros_like(maps[0])
    #     #
    #     # for k in range(pos_lower + 1, pos_upper - 1):
    #     #
    #     #     middle_addition += (energy_bins[k + 1] - energy_bins[k]) * (maps[k] + maps[k + 1]) / 2
    #
    #     # DON'T FORGET TO DO THE ENERGY MULTIPLICATION BEFORE INTEGRATION - SEE ROBUST FOR DIVISTION TOO

    return integrated_maps, new_energy_bins



# REFERENCES

# New Position - https://math.stackexchange.com/questions/143932/calculate-point-given-x-y-angle-and-distance/
# 3534251#3534251