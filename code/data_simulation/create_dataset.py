from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import healpy as hp
from healpy.newvisufunc import projview
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from source_generation.agn_spectral_parameters import energy_flux_agn
from source_generation.pulsar_spectral_parameters import energy_flux_pulsar
import xml.dom.minidom


from map_generation.visualisation import plot_all_sky_map
from map_generation.healpix_maps import create_exposure_map


# def format_scientific_notation_label(numbers):
#     # Formats numbers into scientific notation for inclusion on graphs
#
#     labels = []
#
#     for number in numbers:
#
#         scientific = str(np.format_float_scientific(number, precision=2, trim='0'))
#
#         base, exponent = scientific.split('e')
#
#         if exponent[0] == "+":
#             sign = ""
#         else:
#             sign = "-"
#
#         label = base + "$\\times 10^{" + sign + str(int(exponent[1:])) + "}$"
#
#         labels.append(label)
#
#     return labels


def get_nside(healpix_exposure_map):

    # Healpix parameter nside can be calculated with this function

    nside = hp.pixelfunc.get_nside(healpix_exposure_map)

    return nside


# def plot_all_sky_map(healpix_maps, energy_bins, title: str, directory: str, logarithmic=False):
#
#     # Create directory to store results
#     Path(directory).mkdir(parents=True, exist_ok=True)
#
#     # Format energy bins
#     energy_bin_labels = format_scientific_notation_label(energy_bins)
#
#     # Plot either raw or log of data
#     if logarithmic is True:
#         data = np.log(healpix_maps)
#     else:
#         data = healpix_maps
#
#     # Formatting
#     if len(title) > 10:
#         new_line = "\n"
#     else:
#         new_line = ""
#
#     for b in range(len(healpix_maps)):
#
#         label = "{} Map for {}{} - {} MeV".format(title, new_line, energy_bin_labels[b], energy_bin_labels[b + 1])
#
#         projview(
#             data[b],
#             coord=["G"],
#             graticule=True,
#             graticule_labels=True,
#             xlabel="Galactic Longitude [$\degree$]",
#             ylabel="Galactic Latitude [$\degree$]",
#             cb_orientation="vertical",
#             projection_type="aitoff",
#             title=label,
#             cbar=False,
#             sub=(3, 2, b + 1),
#         )
#
#     plt.tight_layout()
#
#     plt.savefig(directory + title.lower().replace(" ", "_") + "_map.png")
#
#     plt.close()


def angle_to_healpix_pixels(coordinates, nside: int):

    # Transform shape from num_sources x 2 to 2 x num_sources
    coordinates = coordinates.T

    # Extract two lists - one of galactic latitudes and one of galactic longitudes
    latitudes = coordinates[0]
    longitudes = coordinates[1]

    pixels = hp.pixelfunc.ang2pix(nside=nside, theta=latitudes, phi=longitudes, lonlat=True)

    return pixels


def xml_parser(energy_bins, xml_file: str):

    # Read XML files to get latitude and longitude of each source (separate into AGN, pulsars, and background - if they
    # are in the same file)

    docs = xml.dom.minidom.parse(xml_file)

    sources = docs.getElementsByTagName("source")

    num_bins = energy_bins.shape[0]

    coordinates = []
    fluxes = []

    # Remove diffuse sources - only processing point sources with this function
    sources = [sources[k] for k in range(len(sources)) if sources[k].getAttribute("type") != "DiffuseSource"]

    # Parse XML
    for source in sources:

        source_type = source.getAttribute("name")[:3]

        # PARSE SPATIAL PARAMETERS

        spatial_model = source.getElementsByTagName("spatialModel")[0]

        parameters = spatial_model.getElementsByTagName("parameter")

        coordinate = [0, 0]

        for param in parameters:
            name = param.getAttribute("name")

            if name == "RA":
                coordinate[0] = float(param.getAttribute("value"))
            else:
                coordinate[1] = float(param.getAttribute("value"))

        coordinates.append(coordinate)

        # PARSE SPECTRAL FEATURES AND CALCULATE FLUX

        spectral_model = source.getElementsByTagName("spectrum")[0]

        spectral_parameter_dictionary = dict()

        # Get spectral parameters

        parameters = spectral_model.getElementsByTagName("parameter")

        for param in parameters:
            name = param.getAttribute("name")

            scale = float(param.getAttribute("scale"))

            value = float(param.getAttribute("value"))

            actual_value = scale * value

            spectral_parameter_dictionary[name] = actual_value

        binned_fluxes = []

        if source_type == "AGN":

            # Parse spectral parameters

            # For each energy interval, calculate corresponding flux
            for f in range(num_bins - 1):

                flux = energy_flux_agn(pivot_energy=spectral_parameter_dictionary["Eb"],
                                       flux_density=spectral_parameter_dictionary["norm"],
                                       spectral_slope=spectral_parameter_dictionary["alpha"],
                                       curvature=spectral_parameter_dictionary["beta"], min_energy=energy_bins[f],
                                       max_energy=energy_bins[f + 1])

                binned_fluxes.append(flux)

        elif source_type == "PSR":

            # For each energy interval, calculate corresponding flux
            for f in range(num_bins - 1):

                flux = energy_flux_pulsar(pivot_energy=spectral_parameter_dictionary["Scale"],
                                          flux_density=spectral_parameter_dictionary["Prefactor"],
                                          spectral_slope=spectral_parameter_dictionary["Index1"],
                                          exponential_index=spectral_parameter_dictionary["Index2"],
                                          exponential_factor=spectral_parameter_dictionary["Expfactor"],
                                          min_energy=energy_bins[f], max_energy=energy_bins[f + 1])

                binned_fluxes.append(flux)

        else:

            # Unrecognised point source type - allows for debugging when adding in new source types to simulation
            raise TypeError("Cannot recognise source type {}".format(source_type))

        fluxes.append(binned_fluxes)

    # Have coordinates in format [RA, DEC] - need to convert them to Lat-lon

    coordinates = [SkyCoord(ra=c[0] * u.degree, dec=c[1] * u.degree, frame='icrs').galactic for c in coordinates]

    # Get coordinates into numpy array then separate into list of lats and lons
    coordinates = np.array([[c.l.value, c.b.value] for c in coordinates])

    # Convert fluxes to numpy
    fluxes = np.array(fluxes)

    return coordinates, fluxes


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


def create_expected_counts_map(infinite_counts_map):

    binned_count_maps = []

    num_bins = len(infinite_counts_map)

    for b in range(num_bins):

        # Poisson sample each pixel in the infinite statistics count map to get a count map c
        sampled_counts = np.random.poisson(lam=infinite_counts_map[b])

        binned_count_maps.append(sampled_counts)

    return binned_count_maps


# MAIN PROGRAM

# Prepare exposure maps
exposure_maps, energy_bins = create_exposure_map(
    exposure_file="/Volumes/T7/project_data/real_data/fermi_filtered_gti_exposure_map.fits")

# Get NSIDE parameter from exposure map
nside = get_nside(exposure_maps[0])

# Plot exposure maps to verify correctness
plot_all_sky_map(healpix_maps=exposure_maps, energy_bins=energy_bins, title="Exposure",
                 directory="./plots/all_sky_maps/")

coordinates, binned_fluxes = xml_parser(energy_bins=energy_bins, xml_file="./simulated_data/sources.xml")

pixels = angle_to_healpix_pixels(coordinates, nside=nside)

# Calculated source locations in lon-lat, the pixels in which they are situated in the healpix map, the binned exposure
# maps of the sky, and their fluxes
infinite_statistics_maps = create_infinite_statistics_map(exposure_maps, binned_fluxes, pixels)

# Plot infinite counts maps
plot_all_sky_map(healpix_maps=infinite_statistics_maps, energy_bins=energy_bins, title="Infinite Counts",
                 directory="./plots/all_sky_maps/", logarithmic=True)

# Sample infinite statistics maps to create expected counts maps
count_maps = create_expected_counts_map(infinite_counts_map=infinite_statistics_maps)

plot_all_sky_map(healpix_maps=count_maps, energy_bins=energy_bins, title="Expected Counts",
                 directory="./plots/all_sky_maps/", logarithmic=True)




# N.B. Do celestial (RA/Dec coords for PSF) - https://iopscience.iop.org/article/10.1088/0067-0049/203/1/4/pdf