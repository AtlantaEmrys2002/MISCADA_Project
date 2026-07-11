from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
import healpy as hp
from map_generation.utils import angle_to_healpix_pixels
import numpy as np
from psfs.utils import dual_function, monte_carlo_sampler
from reproject import reproject_to_healpix
from scipy.stats import loguniform
from scipy.integrate import quad
from . utils import integrate_over_energy


def create_expected_counts_map(infinite_counts_map):

    binned_count_maps = []

    num_bins = len(infinite_counts_map)

    for b in range(num_bins):

        # Poisson sample each pixel in the infinite statistics count map to get a count map c
        sampled_counts = np.random.poisson(lam=infinite_counts_map[b])

        binned_count_maps.append(sampled_counts)

    return np.array(binned_count_maps)


def create_background_counts_map(expected_counts_isotropic_background, expected_counts_diffuse_background):

    # MAPS PASSED IN HEALPIX FORMAT - EXPECTED COUntS SMOOTHED BY PSF

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


def create_diffuse_background(diffuse_background_file, exposure_map, nside, to_create_num_bins=5):

    with fits.open(diffuse_background_file) as hdul:

        # In MeV
        energy_intervals = hdul[1].data

        energy_intervals = np.array([k[0] for k in energy_intervals])

        num_bins = len(energy_intervals)

        healpix_data = []

        for b in range(num_bins):

            data = hdul[0].data[b]

            # REFORMAT USING REPROJECT

            new_header = hdul[0].header

            new_header = WCS(new_header).sub(2)

            data, _ = reproject_to_healpix((data, new_header), 'galactic', nside=nside)

            # N.B. May have to do the same thing for galactic diffuse background - PSF IS NOT DONE THE SAME WAY AS IN THE ABOVE
            # PAPER SO IT IS NOT IN sr^-1 and EXPOSURE IS IN cm2s
            n_pix = (12 * nside ** 2)

            data *= (4 * np.pi / n_pix)

            healpix_data.append(data)

        # Integrate over energy spectrum
        diffuse_backgrounds, new_energy_bins = integrate_over_energy(maps=np.array(healpix_data),
                                                                     energy_bins=energy_intervals,
                                                                     num_bins=to_create_num_bins, predefined=True)

        # Multiply by ENERGY-AVERAGED exposure
        for k in range(len(exposure_map)):
            diffuse_backgrounds[k] *= exposure_map[k]

    return np.array(diffuse_backgrounds)


def create_exposure_map(exposure_file: str, num_bins_to_create=5):

    # Num bins is the number of energy bins that to create (here, we want 5)

    # Read and plot binned exposure files

    with fits.open(exposure_file) as hdul:

        num_bins = hdul[1].header["TFIELDS"] - 1

        energy_bins = np.array([k[0] for k in hdul[2].data])

        exposure_maps = np.array([hp.read_map(exposure_file, hdu="HPXEXPOSURES", field=b) for b in range(num_bins)])

        # EXPERIMENTATION BELOW

        # Integrate over energy - see robust paper and arxiv paper as well
        exposure_maps, energy_bins = integrate_over_energy(maps=exposure_maps, energy_bins=energy_bins, num_bins=num_bins_to_create,
                                                           energy_weighted=True)

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

    return np.array(binned_infinite_statistics)


def isotropic_func(energy, m_val, c_val):

    return (energy ** m_val) * (np.e ** c_val)


def create_isotropic_background(isotropic_background_file: str, nside: int, exposure_map, energy_bins):

    num_bins = len(energy_bins) - 1

    # Shape tells you length of numpy array representing healpix maps - every pixel will be the same

    with open(isotropic_background_file) as f:

        lines = f.readlines()

    # Format
    lines = np.array([line.strip("\n").split(" ") for line in lines]).astype(np.float64).T

    central_energies = lines[0]
    differential_flux = lines[1]

    # Convert differential fluxes from per steradian to per pixel - https://arxiv.org/html/2302.01947v2

    # N.B. May have to do the same thing for galactic diffuse background - PSF IS NOT DONE THE SAME WAY AS IN THE ABOVE
    # PAPER SO IT IS NOT IN sr^-1 and EXPOSURE IS IN cm2s
    n_pix = (12 * nside ** 2)

    differential_flux *= (4 * np.pi / n_pix)

    # Fit line to log-log plot of isotropic background spectrum
    m, c = np.polyfit(np.log(central_energies), np.log(differential_flux), deg=1)

    # Integrate energy spectrum of isotropic background

    isotropic_values = []

    for b in range(num_bins):

        # integrate energy spectrum over energy range - CHECK THIS WITH ANTHONY
        isotropic_constant = quad(func=isotropic_func, a=energy_bins[b], b=energy_bins[b + 1], args=(m, c))[0]

        isotropic_values.append(isotropic_constant)

    print(isotropic_values)

    isotropic_maps = []

    for e_map in range(num_bins):

        isotropic_maps.append(isotropic_values[e_map] * exposure_map[e_map])

    return np.array(isotropic_maps)





    # THINK I'VE SOLVED THIS ONE - NEED TO INTEGRATE OUT SOLID ANGLE (GO FROM DIRECTIONAL FLUX TO FLUX - DIFFUSE SOURCES HAVE sr^-1 ASPECT)
    # NEED TO INTEGRATE OUT ENERGY DEPENDENCY BY INTEGRATING OVER BIN
    #
    # import matplotlib.pyplot as plt
    #
    # plt.plot(np.log(central_energies), np.log(differential_flux))
    #
    # plt.plot(np.log(central_energies), (m * np.log(central_energies)) + c)
    #
    # plt.xscale("log")
    #
    # plt.show()
    #
    # plt.plot(central_energies, differential_flux)
    #
    # plt.plot(central_energies, (central_energies ** m) * (np.e ** c))
    #
    # plt.show()

    # NOT FINISHED YET


def new_coordinate(ra, dec, radius, angle):

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

    return np.array([new_ra, new_dec])


def coordinates_galactic_to_celestial(coordinates):

    # Converts list of coordinates in l,b format to ra, dec format - ALL IN DEGREES

    ls = coordinates.T[0]
    bs = coordinates.T[1]

    celestial = SkyCoord(l=ls * u.degree, b=bs * u.degree, frame='galactic').icrs

    new_coordinates = np.array([celestial.ra.value, celestial.dec.value]).T

    return new_coordinates


def coordinates_celestial_to_galactic(coordinates):

    # Converts list of coordinates in ra, dec format to l, b format - ALL IN DEGREES

    ras = coordinates.T[0]
    decs = coordinates.T[1]

    coordinates = SkyCoord(ra=ras * u.degree, dec=decs * u.degree, frame='icrs').galactic

    new_coordinates = np.array([coordinates.l.value, coordinates.b.value]).T

    return new_coordinates


def create_point_source_map(coordinates, exposure_maps, psf_parameters, fluxes, nside):

    num_bins = len(exposure_maps)
    source_num = len(coordinates)

    original_pixels = angle_to_healpix_pixels(coordinates, nside=nside)

    point_source_maps = []

    # Convert all galactic coordinates to celestial
    celestial_coordinates = coordinates_galactic_to_celestial(coordinates)

    for b in range(num_bins):

        bin_fluxes = fluxes.T[b]
        exposure_map = exposure_maps[b]
        point_source_map = np.zeros_like(exposure_map)

        # Calculate the number of photons to sample for each pixel

        # BELOW ARE 2 NEW LINES

        # infinite counts
        infinite_cs = np.array([exposure_map[original_pixels[source]] * bin_fluxes[source] for source in range(source_num)])

        # expected counts
        cs = np.random.poisson(lam=infinite_cs)

        for source in range(source_num):

            print(source)

            c = cs[source]

            if c > 0:

                # SAMPLE DISPLACEMENT PARAMETERS

                radial_angle_displacements = monte_carlo_sampler(dual_function, parameters=psf_parameters[b],
                                                                 num_samples=c)
                angles = np.random.uniform(low=0, high=2*np.pi, size=c)

                # Calculate new origins

                new_positions = new_coordinate(ra=celestial_coordinates[source][0], dec=celestial_coordinates[source][1],
                                             radius=radial_angle_displacements, angle=angles).T

                # Convert to longitude-latitude
                new_positions_galactic = coordinates_celestial_to_galactic(new_positions)

                new_pixels = angle_to_healpix_pixels(new_positions_galactic, nside=nside)

                for p in new_pixels:

                    point_source_map[p] += 1

        point_source_maps.append(point_source_map)

    return np.array(point_source_maps)

# REFERENCES

# Astropy Affiliated - https://www.astropy.org/affiliated/
# Map Cube Formats - https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/other_sources.html
# Reproject Suggestion - https://stackoverflow.com/questions/54715123/converting-a-map-in-cartesian-projection-with-
# spherical-coordinate-to-healix-p
# Skycoords and Arrays - https://stackoverflow.com/questions/36146183/astropy-skycoord-extremely-slow-how-to-resovle-it



