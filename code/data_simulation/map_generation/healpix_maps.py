from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
import healpy as hp
from map_generation.utils import angle_to_healpix_pixels
import numpy as np
from psfs.utils import dual_function, monte_carlo_sampler


def create_expected_counts_map(infinite_counts_map):

    binned_count_maps = []

    num_bins = len(infinite_counts_map)

    for b in range(num_bins):

        # Poisson sample each pixel in the infinite statistics count map to get a count map c
        sampled_counts = np.random.poisson(lam=infinite_counts_map[b])

        binned_count_maps.append(sampled_counts)

    return binned_count_maps


def create_exposure_map(exposure_file: str):

    # Read and plot binned exposure files

    with fits.open(exposure_file) as hdul:

        num_bins = hdul[1].header["TFIELDS"] - 1

        energy_bins = np.array([k[0] for k in hdul[2].data])

        exposure_maps = [hp.read_map(exposure_file, hdu="HPXEXPOSURES", field=b) for b in range(num_bins)]

    return exposure_maps, energy_bins


def create_infinite_statistics_map(exposure_maps, fluxes, pixels):

    # Pixel x corresponds to location of source x in the sky with flux x

    binned_infinite_statistics = []

    num_bins = len(exposure_maps)

    for b in range(num_bins):

        healpix_exposure_map = exposure_maps[b]

        # Copy for infinite statistics
        infinite_statistics_counts = np.zeros_like(healpix_exposure_map)

        # Calculate infinite statistics realisation of sky (c bar)
        for x in range(len(pixels)):
            pixel = pixels[x]

            # Important to add - Poisson value is additive
            infinite_statistics_counts[pixel] += healpix_exposure_map[pixel] * fluxes[x, b]

        # Add infinite statistics bin to list
        binned_infinite_statistics.append(infinite_statistics_counts)

    return binned_infinite_statistics


def new_position(ra, dec, radius, angle):

    # N.B. Convert to celestial (RA/Dec coords for PSF) before passing to this function -
    # https://iopscience.iop.org/article/10.1088/0067-0049/203/1/4/pdf

    # Make sure you have converted lat, lon to ra, dec before passing to this function

    # coordinates = SkyCoord(l=lat * u.degree, b=lon * u.degree, frame='galactic').icrs

    # ra = coordinates.ra.value
    # dec = coordinates.dec.value

    # Calculate new position using equations from here https://math.stackexchange.com/questions/143932/calculate-point-
    # given-x-y-angle-and-distance/3534251#3534251

    # Angle must be in radians
    new_ra = (ra + (radius * np.cos(angle))) % 360  # % 360 to ensure wrap-around
    new_dec = (((dec + (radius * np.sin(angle))) + 90) % 180) - 90  # ensure wrap-around

    # COORDINATES ALSO RETURNED IN RA DEC FORMAT - REMEMBER TO CONVERT

    # Convert back to galactic latitude and longitude

    # coordinates = SkyCoord(ra=new_ra * u.degree, dec=new_dec * u.degree, frame='icrs').galactic
    #
    # new_position = [coordinates.l.value, coordinates.b.value]

    # return new_position

    return np.array([new_ra, new_dec])


def coordinates_galactic_to_celestial(coordinates):

    # Converts list of coordinates in l,b format to ra, dec format - ALL IN DEGREES

    new_coordinates = []

    for c in coordinates:

        celestial = SkyCoord(l=c[0] * u.degree, b=c[1] * u.degree, frame='galactic').icrs

        new_coordinates.append([celestial.ra.value, celestial.dec.value])

    return np.array(new_coordinates)


def coordinates_celestial_to_galactic(coordinates):

    # Converts list of coordinates in ra, dec format to l, b format - ALL IN DEGREES

    new_coordinates = []

    for c in coordinates:

        coordinates = SkyCoord(ra=c[0] * u.degree, dec=c[1] * u.degree, frame='icrs').galactic

        new_coordinates.append([coordinates.l.value, coordinates.b.value])

    return np.array(new_coordinates)


import time


def create_point_source_map(coordinates, exposure_maps, psf_parameters, fluxes, nside):

    num_bins = len(exposure_maps)
    source_num = len(coordinates)

    # new_position(lat=coordinates[0][0], lon=coordinates[0][1], radius=100, angle=np.pi)

    original_pixels = angle_to_healpix_pixels(coordinates, nside=nside)

    point_source_maps = []

    # Convert all galactic coordinates to celestial
    celestial_coordinates = coordinates_galactic_to_celestial(coordinates)

    for b in range(num_bins):

        start = time.time()

        print("BIN {}".format(b))

        start_prep = time.time()

        bin_fluxes = fluxes.T[b]
        exposure_map = exposure_maps[b]
        point_source_map = np.zeros_like(exposure_map)

        # Calculate the number of photons to sample for each pixel

        # BELOW ARE 2 NEW LINES

        # infinite counts
        infinite_cs = [exposure_map[original_pixels[source]] * bin_fluxes[source] for source in range(source_num)]

        # expected counts
        cs = np.random.poisson(lam=infinite_cs)

        print("PREP TIME: {} s". format(time.time() - start_prep))

        for source in range(source_num):

            c = cs[source]

            if c > 0:

                # SAMPLE DISPLACEMENT PARAMETERS

                radial_angle_displacements = monte_carlo_sampler(dual_function, parameters=psf_parameters[b],
                                                                 num_samples=c)
                angles = np.random.uniform(low=0, high=2*np.pi, size=c)

                # Calculate new origins

                # new_positions = np.array([new_position(ra=celestial_coordinates[source][0],
                #                                        dec=celestial_coordinates[source][1],
                #                                        radius=radial_angle_displacements[photon],
                #                                        angle=angles[photon]) for photon in range(c)])

                new_positions = new_position(ra=celestial_coordinates[source][0], dec=celestial_coordinates[source][1],
                                             radius=radial_angle_displacements, angle=angles)

                new_positions = new_positions.T

                # Convert to longitude-latitude
                new_positions_galactic = coordinates_celestial_to_galactic(new_positions)

                new_pixels = angle_to_healpix_pixels(new_positions_galactic, nside=nside)

                for p in new_pixels:

                    point_source_map[p] += 1

        point_source_maps.append(point_source_map)

        print("TIME: {}".format(time.time() - start))

    return point_source_maps





