"""
Functions for creating exposure maps, infinite statistics count maps, and count maps for both diffuse and point sources.
"""

from astropy.io import fits
from astropy.wcs import WCS
import healpy as hp
from .utils import angle_to_healpix_pixels, coordinates_celestial_to_galactic, coordinates_galactic_to_celestial
import numpy as np
import numpy.typing as npt
from psfs.utils import monte_carlo_sampler
import reproject
from scipy.stats import loguniform
from scipy.integrate import quad
from .utils import isotropic_func, new_coordinate, integrate_over_energy


def create_diffuse_infinite_statistics_background(diffuse_background_file: str, exposure_map: npt.NDArray[np.float64],
                                                  nside: int, to_create_num_bins: int = 5) -> npt.NDArray[np.float64]:
    """Returns infinite statistics mean count map for the diffuse galactic background (in photons per pixel).

    Parameters
    ----------
    diffuse_background_file : str
        Location of FITS-formatted diffuse background differential flux map (N.B. the differential)
    exposure_map : ndarray
        HEALPix-formatted exposure maps
    nside : int
        Order of the exposure maps.
    to_create_num_bins : int
        Number of energy bins to create infinite statistics maps for (not necessarily the same energy bins as provided
        in the file).

    Returns
    -------
    ndarray
        Binned diffuse galactic background infinite statistics maps.

    """
    with fits.open(diffuse_background_file) as hdul:
        energy_intervals = hdul[1].data.astype(np.float64)

        num_bins = energy_intervals.shape[0]

        n_pix = (12 * nside ** 2)

        # Conversion to per pixel instead of per steradians
        per_pixel_conversion = (4 * np.pi / n_pix)

        new_header = WCS(hdul[0].header).sub(2)

        data = hdul[0].data

        healpix_data = np.array([reproject.reproject_to_healpix((data[b], new_header), 'galactic', nside=nside)[0]
                                 for b in range(num_bins)])

        healpix_data *= per_pixel_conversion

        # Integrate over energy spectrum
        diffuse_backgrounds, new_energy_bins = integrate_over_energy(maps=np.array(healpix_data),
                                                                     energy_bins=energy_intervals,
                                                                     num_bins=to_create_num_bins, predefined=True)

        # Multiply by energy-averaged exposure
        diffuse_backgrounds *= exposure_map

    return np.array(diffuse_backgrounds)


def create_diffuse_source_map(expected_counts_isotropic_background: npt.NDArray[np.float64],
                              expected_counts_diffuse_background: npt.NDArray[np.float64], psfs: list) \
        -> npt.NDArray[np.float64]:
    """When passed binned HEALPix representations of the infinite statistics mean count map for the diffuse isotropic
    background, applies (binned) PSF convolution to the maps, before multiplying backgrounds by normalisation constants
    to vary brightness, then Poisson samples to get background count map for each energy bin.

    Parameters
    ----------
    expected_counts_isotropic_background : ndarray
        Binned infinite statistics expected counts of diffuse isotropic background in HEALPix format.
    expected_counts_diffuse_background : ndarray
        Binned infinite statistics expected counts of diffuse galactic background in HEALPix format.
    psfs : list
        Point spread function binned for application to different photon energies.

    Returns
    -------
    ndarray
        Energy-binned diffuse source count maps (both for isotropic and galactic diffuse sources).

    """
    num_bins = expected_counts_diffuse_background.shape[0]

    expected_counts_isotropic_background = (
        np.array([hp.sphtfunc.smoothing(map_in=expected_counts_isotropic_background[k], beam_window=psfs[k]) for k in
                  range(num_bins)]))

    expected_counts_diffuse_background = (
        np.array([hp.sphtfunc.smoothing(map_in=expected_counts_diffuse_background[k], beam_window=psfs[k]) for k in
                  range(num_bins)]))

    # Clip negative values (ID50)
    expected_counts_isotropic_background = np.clip(a=expected_counts_isotropic_background, a_min=0, a_max=None)
    expected_counts_diffuse_background = np.clip(a=expected_counts_diffuse_background, a_min=0, a_max=None)

    # Sample random normalisation coefficients - random brightness of background components
    a_diff = loguniform.rvs(a=0.1, b=2)
    a_iso = loguniform.rvs(a=0.1, b=2)

    # Normalise expected count maps
    expected_counts_isotropic_background *= a_iso
    expected_counts_diffuse_background *= a_diff

    # Mean background
    background = expected_counts_diffuse_background + expected_counts_isotropic_background

    # Poisson sample expected counts to get realisation
    background_realisation = np.random.poisson(lam=background)

    return np.array(background_realisation)


def create_exposure_map(exposure_file: str, num_bins_to_create: int = 5) \
        -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Reads in and energy integrates (in MeV) over calculated exposure maps, binning into the specified number of bins.

    Parameters
    ----------
    exposure_file : str
        Location of FITS-formatted exposure maps calculated using fermitools from real Fermi LAT data.
    num_bins_to_create : int
        Number of energy bins (and hence the number of exposure maps) to create.

    Returns
    -------
    tuple
        A tuple of the binned HEALPix-formatted exposure maps and corresponding energy bins.

    """
    with fits.open(exposure_file) as hdul:
        num_bins = hdul[1].header["TFIELDS"] - 1

        energy_bins = hdul[2].data.astype(np.float64)

        # Read in exposure files
        exposure_maps = np.array([hp.read_map(exposure_file, hdu="HPXEXPOSURES", field=b) for b in range(num_bins)])

        # Integrate over energy - removes energy dependence
        exposure_maps, energy_bins = integrate_over_energy(maps=exposure_maps, energy_bins=energy_bins,
                                                           num_bins=num_bins_to_create, energy_weighted=True)

    return exposure_maps, energy_bins


def create_isotropic_infinite_statistics_background(isotropic_background_file: str, nside: int,
                                                    exposure_map: npt.NDArray[np.float64],
                                                    energy_bins: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Returns infinite statistics mean count map for the diffuse isotropic background (in photons per pixel).

    Parameters
    ----------
    isotropic_background_file : str
        Location of FITS-formatted isotropic background differential flux map (N.B. the differential)
    exposure_map : ndarray
        HEALPix-formatted exposure maps
    nside : int
        Order of the exposure maps.
    energy_bins : ndarray
        Energy bins to create infinite statistics maps for (not necessarily the same energy bins as provided
        in the file).

    Returns
    -------
    ndarray
        Energy-binned, HEALPix-formatted infinite statistics maps for the isotropic background.

    """
    num_bins = energy_bins.shape[0] - 1

    # Shape tells you length of numpy array representing healpix maps - every pixel will be the same

    with open(isotropic_background_file) as f:
        lines = f.readlines()

    # Format
    lines = np.array([line.strip("\n").split(" ") for line in lines]).astype(np.float64).T

    central_energies = lines[0]
    differential_flux = lines[1]

    # Convert differential fluxes from per steradians to per pixel - https://arxiv.org/html/2302.01947v2
    n_pix = (12 * nside ** 2)

    differential_flux *= (4 * np.pi / n_pix)

    # Fit line to log-log plot of isotropic background spectrum
    m, c = np.polyfit(np.log(central_energies), np.log(differential_flux), deg=1)

    # Integrate energy spectrum of isotropic background
    isotropic_values = [quad(func=isotropic_func, a=float(energy_bins[b]), b=float(energy_bins[b + 1]), args=(m, c))[0]
                        for b in range(num_bins)]

    isotropic_maps = np.array([isotropic_values[e_map] * exposure_map[e_map] for e_map in range(num_bins)])

    return isotropic_maps


def create_infinite_counts_maps(source_pixels, exposure_maps, fluxes):
    """Calculates an infinite statistics count map for a given type of point source, given the telescope exposure map,
    as well as the location and flux of each source.

    Parameters
    ----------
    source_pixels : ndarray
        The location of each source in the HEALPix-formatted sky map
    exposure_maps : ndarray
        The HEALPix-formatted exposure maps of the sky, calculated using fermitools.
    fluxes : ndarray
        The corresponding fluxes of each source in source_pixels.

    Returns
    -------
    ndarray
        Array of binned infinite statistics in HEALPix format.

    """
    # Calculates infinite counts map (to be Poisson sampled) for each catalog

    infinite_counts_maps = np.zeros_like(exposure_maps)

    # Transform fluxes
    fluxes = fluxes.T

    for b in range(exposure_maps.shape[0]):
        # Calculate the number of photons of a given energy to sample for each pixel - we add as more than one source
        # may be in each pixel
        infinite_counts_maps[b, source_pixels] += exposure_maps[b, source_pixels] * fluxes[b]

    return infinite_counts_maps


def create_count_map(coordinates: npt.NDArray[np.float64], exposure_maps: npt.NDArray[np.float64],
                     psf_parameters: npt.NDArray[np.float64], fluxes: npt.NDArray[np.float64], nside: int,
                     infinite_stats_file: str) -> npt.NDArray[np.float64]:
    """Creates photon count map for point sources, provided that the galactic coordinates of the sources, their fluxes,
    the exposure maps calculated using fermitools, and PSF parameters (also calculated using a mixture of fermitools and
    custom code) are provided. These count maps will be binned and realistic.

    Parameters
    ----------
    coordinates : ndarray
        2D array of galactic coordinates of relevant point sources.
    exposure_maps : ndarray
        2D array of binned, HEALPix-formatted exposure maps.
    psf_parameters : list
        Energy-binned point spread function parameters that cause photons to be "smeared" across the sky when observed
        by the telescope.
    fluxes : ndarray
        Array of integral photon fluxes of relevant point sources
    nside : int
        Order of the HEALPix maps
    infinite_stats_file : str
        Location of pre-computed infinite statistics mean count maps for relevant sources.

    """
    num_bins = exposure_maps.shape[0]
    source_num = coordinates.shape[0]

    fluxes = fluxes.T

    # Convert all galactic coordinates to celestial
    celestial_coordinates = coordinates_galactic_to_celestial(coordinates)
    original_pixels = angle_to_healpix_pixels(coordinates, nside=nside)

    unique, counts = np.unique(original_pixels, return_counts=True)

    source_in_pixel = dict(zip(unique, counts))

    # Fetch infinite stats maps
    infinite_stats_maps = np.array([hp.fitsfunc.read_map(infinite_stats_file.format(b)) for b in range(num_bins)])

    count_maps = np.zeros_like(exposure_maps)

    for source in range(source_num):

        source_pixel = original_pixels[source]

        celestial_coord = celestial_coordinates[source]

        # If only one source is in the pixel, take the infinite statistics map value
        if source_in_pixel[source_pixel] == 1:

            c_per_bin = infinite_stats_maps[:, source_pixel]

        else:

            c_per_bin = exposure_maps[:, source_pixel] * fluxes[:, source]

        # RANDOM SAMPLE

        # Get expected counts
        # Poisson sample infinite statistics map to get expected counts from each POINT source
        sampled_counts = np.random.poisson(lam=c_per_bin)

        for b in range(num_bins):
            # Sample radial displacement
            radial_angle_displacements = monte_carlo_sampler(parameters=psf_parameters[b],
                                                             num_samples=sampled_counts[b])

            angles = np.random.uniform(low=0, high=2 * np.pi, size=sampled_counts[b])

            # Calculate new origins
            new_positions = new_coordinate(ra=celestial_coord[0], dec=celestial_coord[1],
                                           radius=radial_angle_displacements, angle=angles).T

            # Convert to longitude-latitude
            new_positions_galactic = coordinates_celestial_to_galactic(new_positions)

            new_pixels = angle_to_healpix_pixels(new_positions_galactic, nside=nside)

            count_maps[b, new_pixels] += 1

    return count_maps

# REFERENCES

# Astropy Affiliated - https://www.astropy.org/affiliated/
# Background Model Information - https://fermi.gsfc.nasa.gov/ssc/data/access/lat/BackgroundModels.html
# Map Cube Formats - https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/other_sources.html
# Reproject Suggestion - https://stackoverflow.com/questions/54715123/converting-a-map-in-cartesian-projection-with-
# spherical-coordinate-to-healix-p
# Sky-coordinates and Arrays - https://stackoverflow.com/questions/36146183/astropy-skycoord-extremely-slow-how-to-
# resovle-it
# Unique Counts - https://stackoverflow.com/questions/28663856/how-do-i-count-the-occurrence-of-a-certain-item-in-an-
# ndarray
