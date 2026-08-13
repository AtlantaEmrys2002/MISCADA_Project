"""
Utility functions used when creating all-sky maps.
"""

from astropy.coordinates import SkyCoord
import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad


def angle_to_healpix_pixels(coordinates, nside: int = 256):
    """Wrapper for healpy function that takes list of galactic coordinates and converts them to the HEALPix pixel that
    encompasses that coordinate.

    Parameters
    ----------
    coordinates : ndarray
        Galactic coordinates (longitude and latitude) to be converted.
    nside : int
        Order of the HEALPix map.

    Returns
    -------
    ndarray
        Returns array of pixels where value at index k corresponds to the pixel that encompasses coordinate k.

    """
    # Transform shape from num_sources x 2 to 2 x num_sources
    coordinates = coordinates.T

    # Extract two lists - one of galactic latitudes and one of galactic longitudes
    longitudes = coordinates[0]
    latitudes = coordinates[1]

    pixels = hp.pixelfunc.ang2pix(nside=nside, theta=longitudes, phi=latitudes, lonlat=True)

    return pixels


def coordinates_galactic_to_celestial(coordinates):
    """Wrapper that allows a series of galactic coordinates (in degrees) to be converted efficiently to celestial
    coordinates (also in degrees).

    Parameters
    ----------
    coordinates : ndarray
        2D array of coordinates to be converted to celestial coordinates

    Returns
    -------
    ndarray
        A 2D array of the coordinates converted to celestial coordinates.

    """
    # Converts list of coordinates in l,b format to ra, dec format

    celestial = SkyCoord(l=coordinates[:, 0] * u.degree, b=coordinates[:, 1] * u.degree, frame='galactic').icrs

    new_coordinates = np.array([celestial.ra.value, celestial.dec.value]).T

    return new_coordinates


def coordinates_celestial_to_galactic(coordinates):
    """Wrapper that allows a series of galactic coordinates (in degrees) to be converted efficiently to celestial
    coordinates (also in degrees).

    Parameters
    ----------
    coordinates : ndarray
        2D array of coordinates to be converted to celestial coordinates

    Returns
    -------
    ndarray
        A 2D array of the coordinates converted to celestial coordinates.

    """

    coordinates = SkyCoord(ra=coordinates[:, 0] * u.degree, dec=coordinates[:, 1] * u.degree, frame='icrs').galactic

    new_coordinates = np.array([coordinates.l.value, coordinates.b.value]).T

    return new_coordinates


def get_nside(healpix_map) -> int:
    """Wrapper for healpy function that returns the order of a given healpix map.

    Parameters
    ---------
    healpix_map : ndarray
        1D array that represents a HEALPix map.

    Returns
    -------
    int
        Order of HEALPix map

    """
    # Healpix parameter nside can be calculated with this function

    return hp.pixelfunc.get_nside(healpix_map)


def exponential_func(energy):
    """The exponential func used when calculating the weighted integral of the exposure maps.

    Parameters
    ----------
    energy
        Photon energies to pass to function.

    """
    return energy ** -2.4


def integrate_over_energy(maps, energy_bins, num_bins: int, energy_weighted=False, predefined: bool = False,
                          min_e: float = 300., max_e: float = 200000):
    """Numerically integrates HEALPix map over each of a series of logarithmically-spaced energy bins using the
    trapezium rule (also referred to as the trapezoidal rule). If energy weighted then it divides the integral of the
    map by an exponential function then divides by the integral of said energy function.

    Parameters
    ----------
    maps
        Binned HEALPix maps to integrate over - do not have to be binned according to the logarithmic scheme eventually
        followed.
    energy_bins : ndarray
        Energy bins that the HEALPix maps were originally binned into.
    num_bins : int
        The number of new logarithmically-spaced bins to integrate between (i.e. the number of integrated maps to
        produce).
    energy_weighted : bool
        Indicates whether to calculate the energy weighted integral or not.
    predefined : bool
        Indicates whether to integrate between the pre-defined boundaries of the provided energy bins or to calculate
        new energy bin boundaries and integrate between them.
    min_e : float
        The minimum photon energy from which the new energy bins are to start from.
    max_e : float
        The maximum photon energy to which the new energy bins are to get to.

    """
    # Extrapolate a map for the upper energy bin to ensure maps and energy bins have the same size

    m = ((maps[-1] - maps[-2]) /
         (energy_bins[-1] - energy_bins[-2]))

    c = maps[-1] - (m * energy_bins[-1])

    maps = list(maps)

    maps.append((m * energy_bins[-1]) + c)

    maps = np.array(maps)

    # if energy_weighted is true, then we use a weighting factor when integrating over the energy range
    # This is used when integrating over exposure maps
    if energy_weighted is True:

        exponential_values = exponential_func(energy_bins)

        for k in range(maps.shape[0]):
            maps[k] *= exponential_values[k]

    # Use the trapezium rule for numerical integration here

    # If upper and lower boundaries of energy values not specified, take them from the original energy bins fed to
    # function always create num_bins + 1 - this creates the boundaries (need 6 boundaries to create 5 bins).
    if predefined is False:
        min_e, max_e = energy_bins[0], energy_bins[-1]

    new_energy_bins = np.logspace(np.log(min_e), np.log(max_e), num=num_bins + 1, base=np.e)

    new_maps = []

    # Calculate new maps at each of the new energy bin boundaries
    for n in range(num_bins):

        new_energy_boundary = new_energy_bins[n]

        # Check to see if new energy bin value already in previous energy bins
        already_calculated = np.isclose(energy_bins, new_energy_boundary)

        if np.any(already_calculated):

            new_maps.append(maps[np.argwhere(already_calculated)[0, 0]])

        else:

            # pos indicates where new_energy boundary exists such that energy_bin[pos - 1] <= new_energy_boundary <
            # energy_bins[pos]
            pos = np.searchsorted(energy_bins, new_energy_boundary)

            left_boundary = pos - 1
            right_boundary = pos

            # Fit straight line between points at energy bins surrounding new energy value
            m = ((maps[right_boundary] - maps[left_boundary]) /
                 (energy_bins[right_boundary] - energy_bins[left_boundary]))

            c = maps[left_boundary] - (m * energy_bins[left_boundary])

            new_maps.append((m * new_energy_boundary) + c)

            if n == num_bins - 1:
                # Create an additional map for upper boundary extrapolating even further with the straight line - this
                # is in cases when a passed energy bin does not match the upper energy boundary and as such
                # upper_addition must be included

                new_maps.append((m * new_energy_bins[n + 1]) + c)

    integrated_maps = []

    # Calculate numerical integrals
    for n in range(num_bins):

        lower_energy_boundary = new_energy_bins[n]
        upper_energy_boundary = new_energy_bins[n + 1]

        # Closest energy bin to lower_energy_boundary that has a greater value than lower energy value
        pos_lower = np.searchsorted(energy_bins, lower_energy_boundary)

        # Closest energy bin to upper_energy_boundary that has a lesser value than upper energy value
        pos_upper = np.searchsorted(energy_bins, upper_energy_boundary, side="right") - 1

        # Calculate the lower part of the integral between the new energy bin and the closest energy bin in the maps
        if np.isclose(lower_energy_boundary, energy_bins[pos_lower]):

            lower_addition = np.zeros_like(maps[0])

        else:

            lower_addition = (maps[pos_lower] + new_maps[n]) * (energy_bins[pos_lower] - lower_energy_boundary) / 2

        # Calculate the upper part of the integral between the new energy bin (upper) and the closest energy bin the
        # passed maps

        if np.isclose(upper_energy_boundary, energy_bins[pos_upper]):

            upper_addition = np.zeros_like(maps[0])

        else:

            upper_addition = (maps[pos_upper] + new_maps[n + 1]) / (upper_energy_boundary - energy_bins[pos_upper]) / 2

        # Calculate the middle section of the interval
        middle_addition = np.zeros_like(maps[0])

        for m in range(pos_lower, pos_upper):
            sub_integral = (maps[m] + maps[m + 1]) * (energy_bins[m + 1] - energy_bins[m]) / 2

            middle_addition += sub_integral

        integrated_map = lower_addition + middle_addition + upper_addition

        integrated_maps.append(integrated_map)

    integrated_maps = np.array(integrated_maps)

    if energy_weighted is True:

        integrated_energies = [quad(exponential_func, a=float(new_energy_bins[k]),
                                    b=float(new_energy_bins[k + 1]))[0] for k in range(num_bins)]

        for x in range(num_bins):
            integrated_maps[x] /= integrated_energies[x]

    return integrated_maps, new_energy_bins


def isotropic_func(energy, m_val: np.float64, c_val: np.float64):
    """Function utilised when energy-integrating over isotropic diffuse background.

    Parameters
    ----------
    energy : ndarray
        Energy values to calculate differential flux density from.
    m_val : np.float64
        Gradient of line showing relationship between photon energy and flux density.
    c_val : np.float64
        y-Intercept of line showing relationship between photon energy and flux density.

    """
    return (energy ** m_val) * (np.e ** c_val)


def cartesian_patch(count_map, lon, lat, xsize, lonra, latra):
    """Please note that this code was directly adapted from healpy's cartview function found here -
    https://github.com/healpy/healpy/blob/main/lib/healpy/visufunc.py#L689. To satisfy all proper referencing standards,
    I will cite the HEALPix paper in my final report.

    :param count_map:
    :param lon:
    :param lat:
    :param xsize:
    :param lonra:
    :param latra:
    :return:
    """
    # Ensure that the nside is valid

    f = plt.figure(figsize=(8.5, 5.4))

    # Starting to draw : turn interactive off

    # count_map = hp.pixelfunc.ma_to_array(count_map)

    ax = hp.projaxes.HpxCartesianAxes(
        f, (0.075, 0.05, 0.85, 0.9), coord='G', rot=(lon, lat, 0.), format='%.3g', flipconv='astro'
    )

    img = ax.projmap(
        hp.pixelfunc.ma_to_array(count_map),
        nest=False,
        coord='G',
        vmin=None,
        vmax=None,
        xsize=xsize,
        ysize=None,
        lonra=lonra,
        latra=latra,
        cmap=None,
        badcolor='gray',
        bgcolor='white',
        norm=None,
        aspect=None,
        alpha=None,
    )

    return img


def new_coordinate(ra: np.float64, dec: np.float64, radius, angle):
    """Calculates new celestial coordinate origin of a photon, given a radial displacement and the angle from its
    originally presumed celestial coordinate origin.

    Parameters
    ----------
    ra : np.float64
        Original RA (in degrees) of photon origin
    dec : np.float64
        Original Dec (in degrees) of photon origin
    radius : ndarray
        Radial displacement of photon from presumed origin
    angle : ndarray
        Angle from presumed photon origin to actual photon origin.

    """
    # Make sure you have converted lat, lon to ra, dec before passing to this function

    # Equations taken from New Position reference below

    # Angle must be in radians
    new_ra = (ra + (radius * np.cos(angle))) % 360  # % 360 to ensure wrap-around
    new_dec = (((dec + (radius * np.sin(angle))) + 90) % 180) - 90  # ensure wrap-around

    # COORDINATES ALSO RETURNED IN RA DEC FORMAT - REMEMBER TO CONVERT

    return np.array([new_ra, new_dec])

# REFERENCES

# Co-latitude and Longitude - https://mathworld.wolfram.com/SphericalCoordinates.html
# New Position - https://math.stackexchange.com/questions/143932/calculate-point-given-x-y-angle-and-distance/
# 3534251#3534251
# PSF Function
# Trapezoidal Rule - https://en.wikipedia.org/wiki/Trapezoidal_rule
