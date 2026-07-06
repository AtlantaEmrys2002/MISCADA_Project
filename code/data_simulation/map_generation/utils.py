import healpy as hp
import numpy as np


def angle_to_healpix_pixels(coordinates, nside: int):

    # Transform shape from num_sources x 2 to 2 x num_sources
    coordinates = coordinates.T

    # Extract two lists - one of galactic latitudes and one of galactic longitudes
    latitudes = coordinates[0]
    longitudes = coordinates[1]

    pixels = hp.pixelfunc.ang2pix(nside=nside, theta=latitudes, phi=longitudes, lonlat=True)

    return pixels


def dual_function(x, sigma_core, gamma_core, sigma_tail, gamma_tail, f_core):

    first_distribution = king_function(x, sigma=sigma_core, gamma=gamma_core)

    second_distribution = king_function(x, sigma=sigma_tail, gamma=gamma_tail)

    return (f_core * first_distribution) + ((1 - f_core) * second_distribution)


def get_nside(healpix_exposure_map):

    # Healpix parameter nside can be calculated with this function

    nside = hp.pixelfunc.get_nside(healpix_exposure_map)

    return nside


def king_function(x, sigma, gamma):

    # Called King function by this paper - https://iopscience.iop.org/article/10.1088/0004-637X/765/1/54/pdf
    # However, it is often referred to as the Moffat distribution - note in final paper

    factor = 1 / (2 * np.pi * (sigma ** 2))

    first_term = (1 - (1 / gamma))

    second_term = 1 + ((1 / (2 * gamma)) * (x ** 2 / sigma ** 2))

    return factor * first_term * (second_term ** (- gamma))
