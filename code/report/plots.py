"""
This code is used to create extra plots for the report (not all plots in the report can be generated with this file
- see any file named visualisation.py in data_simulation and evaluation for other plots.
"""

from astropy.io import fits
import copy
import healpy as hp
# from matplotlib import colorizer as mcolorizer
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import numpy.typing as npt
import warnings
from scipy.integrate import quad

def exponential_func(energy):
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


def format_scientific_notation_label(numbers):
    # Formats numbers into scientific notation for inclusion on graphs

    labels = []

    for number in numbers:

        scientific = str(np.format_float_scientific(number, precision=2, trim='0'))

        base, exponent = scientific.split('e')

        if exponent[0] == "+":
            sign = ""
        else:
            sign = "-"

        label = base + "$\\times 10^{" + sign + str(int(exponent[1:])) + "}$"

        labels.append(label)

    return labels


def cartesian_patch(count_map, lon, lat):
    """Please note that this code was directly adapted from healpy's cartview function found here -
    https://github.com/healpy/healpy/blob/main/lib/healpy/visufunc.py#L689. To satisfy all proper referencing standards,
    I will cite the HEALPix paper in my final report. This code projects the map into cartesian coordinates and selects
    the 10 x 10 degree patch around lonra, latra

    :param count_map:
    :param lon:
    :param lat:
    :param xsize:
    :param lonra:
    :param latra:
    :return:
    """

    f = plt.figure(figsize=(8.5, 5.4))

    ax = hp.projaxes.HpxCartesianAxes(
        f, (0.075, 0.05, 0.85, 0.9), coord='G', rot=(lon, lat, 0.), format='%.3g', flipconv='astro'
    )

    img = ax.projmap(
        hp.pixelfunc.ma_to_array(count_map),
        nest=False,
        coord='G',
        vmin=None,
        vmax=None,
        xsize=64,
        ysize=None,
        lonra=[-5, 5],
        latra=[-5, 5],
        cmap=None,
        badcolor='gray',
        bgcolor='white',
        norm=None,
        aspect=None,
        alpha=None,
    )

    return img

def fit_diffuse_source_psf(roi_count_map: str, nside: int = 512) -> list:
    """Derives point spread function from gtmodel output that can be applied to diffuse background expected counts map.

    Parameters
    ----------
    roi_count_map : str
        Location of FITS-formatted infinite statistics expected count map of ROI in real data.
    nside : int
        Order of HEALPix maps to create beam function (used as PSF) for.

    Returns
    -------
    list
        Energy-binned PSFs for diffuse sources.

    """
    # N.B. do not energy integrate this function, as we are looking at energy-binned gtmodel output - already covers
    # the entire energy bin

    with fits.open(roi_count_map) as hdul:
        # Centre of data
        midpoint = hdul[0].data[0].shape[0] // 2

        num_bins = hdul[0].data.shape[0]

        psfs = []

        for b in range(num_bins):
            counts = hdul[0].data[b]

            # Average x and y-axis through point
            x_axis = counts[midpoint]
            y_axis = counts[:, midpoint]

            minus_included = np.arange(-x_axis.shape[0], x_axis.shape[0])

            # Arithmetic mean along the x and y-axis (remember, this is radial)
            values = (x_axis + y_axis) / 2

            # Convert from pixels to radians - approximately
            side_length_pixel = np.sqrt((4 * np.pi) / (12 * nside ** 2))

            # The minus included is in degrees, therefore multiply by pi/180 to convert to radians
            radius = side_length_pixel * minus_included * np.pi / 180

            beam = np.array(hp.sphtfunc.bl2beam(bl=values, theta=radius))

            # Normalise function (y-axis)
            beam /= np.max(beam)

            # Clip < 0 values to 0
            beam = np.clip(beam, a_min=0, a_max=np.max(beam))

            psfs.append(beam)

    return psfs


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

        num_bins = np.array(hdul[1].header["TFIELDS"]) - 1

        energy_bins = hdul[2].data.astype(np.float64)

        # Read in exposure files
        exposure_maps = np.array([hp.read_map(exposure_file, hdu="HPXEXPOSURES", field=b) for b in range(num_bins)])

        # Integrate over energy - removes energy dependence
        exposure_maps, energy_bins = integrate_over_energy(maps=exposure_maps, energy_bins=energy_bins,
                                                           num_bins=num_bins_to_create, energy_weighted=True)

    return exposure_maps, energy_bins


def plot_fermi_lat_count_map(binned_count_map_fermi: str="/Volumes/T7/project_data/real_data/fermi_filtered_gti_binned_for_evaluation.fits"):

    plt.cla()
    plt.clf()
    plt.close()

    # NOTE THAT THIS IS JUST TAKING A LOW ENERGY bin AND NO TITLE INCLUDED

    Path("./plots").mkdir(parents=True, exist_ok=True)

    count_map = np.sum(np.array([hp.read_map(binned_count_map_fermi, hdu="SKYMAP", field=b) for b in range(5)]), axis=0)

    hp.newvisufunc.projview(
        count_map, coord=["G"], graticule=True, graticule_labels=True, xlabel="Galactic Longitude, $l$ [$\\degree$]",
        ylabel="Galactic Latitude, $b$ [$\\degree$]", unit="Photons", cb_orientation="vertical",
        projection_type="aitoff",
    )

    plt.tight_layout()

    plt.savefig("./plots/fermi_count_map.png")

    plt.close()

    hp.newvisufunc.projview(
        count_map, coord=["G"], graticule=True, graticule_labels=True, xlabel="Galactic Longitude, $l$ [$\\degree$]",
        ylabel="Galactic Latitude, $b$ [$\\degree$]", unit="Photons", cb_orientation="vertical",
        projection_type="aitoff", norm='log', title="Aitoff Projection of Fermi-LAT Count Map",
    )

    plt.tight_layout()

    plt.savefig("./plots/fermi_count_map_titled.png")

    plt.close()


def plot_fermi_lat_exposure(binned_count_map_fermi: str="/Volumes/T7/project_data/real_data/fermi_filtered_gti_exposure_map.fits"):

    plt.cla()
    plt.clf()
    plt.close()

    # NOTE THAT THIS IS JUST TAKING A LOW ENERGY bin AND NO TITLE INCLUDED

    Path("./plots").mkdir(parents=True, exist_ok=True)

    count_map = np.sum(np.array([hp.read_map(binned_count_map_fermi, hdu="SKYMAP", field=b) for b in range(5)]), axis=0)

    hp.newvisufunc.projview(
        count_map, coord=["G"], graticule=True, graticule_labels=True, xlabel="Galactic Longitude, $l$ [$\\degree$]",
        ylabel="Galactic Latitude, $b$ [$\\degree$]", unit="$\\text{cm}^2\\text{s}$", cb_orientation="vertical",
        projection_type="aitoff", norm='log',
    )

    plt.tight_layout()

    plt.savefig("./plots/fermi_count_map.png")

    plt.close()

    hp.newvisufunc.projview(
        count_map, coord=["G"], graticule=True, graticule_labels=True, xlabel="Galactic Longitude, $l$ [$\\degree$]",
        ylabel="Galactic Latitude, $b$ [$\\degree$]", unit="Photons", cb_orientation="vertical",
        projection_type="aitoff", norm='log', title="Aitoff Projection of $Fermi$ Exposure Maps",
    )

    plt.tight_layout()

    plt.savefig("./plots/exposure_map.png")

def plot_exposure_map(title: str = "Exposure", directory= "./plots/", logarithmic=False):

    healpix_maps, energy_bins = create_exposure_map(exposure_file="/Volumes/T7/project_data/real_data/fermi_filtered_gti_exposure_map.fits")

    # Format energy bins
    energy_bin_labels = format_scientific_notation_label(energy_bins)

    # Plot either raw or log of data
    if logarithmic is True:

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            data = np.log(healpix_maps)

            # Replace all log0 NaNs with 0
            data = np.where(healpix_maps > 0, data, 0)

    else:
        data = healpix_maps

    # Formatting
    if len(title) > 10:
        new_line = "\n"
    else:
        new_line = ""

    for b in range(len(healpix_maps)):
        label = "{} Map \n {}{} - {} MeV".format(title, new_line, energy_bin_labels[b], energy_bin_labels[b + 1])

        hp.newvisufunc.projview(
            data[b], coord=["G"], graticule=True, graticule_labels=True, xlabel="Galactic Longitude, $l$ [$\\degree$]",
            ylabel="Galactic Latitude, $b$ [$\\degree$]", cb_orientation="vertical", projection_type="aitoff", title=label,
            sub=(3, 2, b + 1), unit="cm$^2$s", xtick_label_color='white', fontsize={"cbar_tick_label": 12, "cbar_label": 14, "xlabel":14, "ylabel":14}
        )

    plt.tight_layout()

    plt.savefig(directory + title.lower().replace(" ", "_") + "_map.png")

    plt.close()


def plot_simulated_count_map(title: str = "Simulated Count", directory= "./plots/", logarithmic=False):

    energy_bins = np.logspace(np.log(300), np.log(200000), num=6, base=np.e)

    agns_maps = np.array([hp.read_map("./skymap_0_for_plotting/agns_{}.fits".format(b)) for b in range(5)])
    psrs_maps = np.array([hp.read_map("./skymap_0_for_plotting/pulsars_{}.fits".format(b)) for b in range(5)])
    background_maps = np.array([hp.read_map("./skymap_0_for_plotting/background_{}.fits".format(b)) for b in range(5)])

    healpix_maps = agns_maps + psrs_maps + background_maps

    # Format energy bins
    energy_bin_labels = format_scientific_notation_label(energy_bins)

    # Plot either raw or log of data
    if logarithmic is True:

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            data = np.log(healpix_maps)

            # Replace all log0 NaNs with 0
            data = np.where(healpix_maps > 0, data, 0)

    else:
        data = healpix_maps

    # Formatting
    if len(title) > 10:
        new_line = "\n"
    else:
        new_line = ""

    plt.rcParams["figure.figsize"] = (40, 50)

    for b in range(len(healpix_maps)):

        label = "{} Map \n {}{} - {} MeV".format(title, new_line, energy_bin_labels[b], energy_bin_labels[b + 1])

        hp.newvisufunc.projview(
            data[b], coord=["G"], graticule=True, graticule_labels=True, xlabel="Galactic Longitude, $l$ [$\\degree$]",
            ylabel="Galactic Latitude, $b$ [$\\degree$]", cb_orientation="vertical", projection_type="aitoff", title=label,
            sub=(3, 2, b + 1), unit="log$_{10}$(photons)", xtick_label_color='white',
            fontsize={"cbar_tick_label": 12, "cbar_label": 14, "xlabel": 14, "ylabel": 14}
        )

    plt.tight_layout()

    plt.savefig(directory + title.lower().replace(" ", "_") + "_map.png")

    plt.close()


def plot_419(patch_directory="./../source_extractors/real_data/real_patches/patches/patch_419"):

    plt.cla()
    plt.clf()
    plt.close()

    plt.rcParams["figure.figsize"] = (4, 10)

    binned_patch = np.load(patch_directory + "/patch.npy")

    fig, axs = plt.subplots(6, 1, gridspec_kw={"height_ratios": [1, 1, 1, 1, 1, 0.05]})

    # create a colorizer with a predefined norm to be shared across all images
    norm = mcolors.Normalize(vmin=np.min(binned_patch), vmax=np.max(binned_patch))
    colorizer = mcolorizer.Colorizer(norm=norm)

    images = []

    count = 0

    for ax, data in zip(axs.flat, binned_patch):

        if count == 5:
            fig.colorbar(images[0], orientation='horizontal', cax=ax, fraction=0.1, label="Photons")
            continue

        images.append(ax.imshow(data, colorizer=colorizer))
        ax.set_axis_off()

        count += 1

    fig.colorbar(images[0], orientation='horizontal', cax=axs.flat[5], fraction=0.1, label="Photons")

    fig.tight_layout()

    plt.savefig("./plots/binned_patch.png")

    plt.cla()
    plt.clf()
    plt.close()

    plt.rcParams["figure.figsize"] = (4, 5)

    mask = np.load(patch_directory + "/mask.npy")

    noise = np.random.choice(2, size=(64 * 64), p=[0.99, 0.01]).reshape((64, 64))

    plt.imshow(np.logical_or(mask, noise))

    plt.savefig("./plots/mask_with_noise.png")

    plt.cla()
    plt.clf()
    plt.close()

    plt.rcParams["figure.figsize"] = (4, 5)

    plt.imshow(mask)

    plt.savefig("./plots/mask.png")

    plt.cla()
    plt.clf()
    plt.close()

    plt.rcParams["figure.figsize"] = (8, 4)

    metadata = pd.read_csv(patch_directory + "/metadata.csv")

    regions = []

    for k in zip(metadata["cartesian_y"].to_numpy(), metadata["cartesian_x"].to_numpy()):

        coord = [int(k[0]) // 2, int(k[1]) // 2]

        regions.append(copy.deepcopy(binned_patch[:, coord[0] - 5 : coord[0] + 5, coord[1] - 5 : coord[1] + 5]))

        print(regions[0][0].shape)

    regions = np.array(regions)

    num_region = 1

    for r in regions:

        fig, axs = plt.subplots(1, 5, gridspec_kw={"width_ratios":[1, 1, 1, 1, 1]})
        # fig.suptitle("Boxes Around Detected Sources")

        images = []

        count = 0

        # create a colorizer with a predefined norm to be shared across all images
        norm = mcolors.Normalize(vmin=np.min(binned_patch), vmax=np.max(binned_patch))
        colorizer = mcolorizer.Colorizer(norm=norm)

        for ax, data in zip(axs.flat, r):

            if count == 5:
                # fig.colorbar(images[0], orientation='vertical', cax=ax, label="Photons", fraction=0.046, pad=0.04)
                continue

            images.append(ax.imshow(data, colorizer=colorizer))
            ax.set_axis_off()

            count += 1

        # cax = fig.add_axes([axs.flat[3].get_position().x1 - 0.25, axs.flat[2].get_position().y0, 0.02,
        #                     axs.flat[2].get_position().y1 - axs.flat[2].get_position().y0])

        cax = axs[4].inset_axes((1.05, 0, 0.08, 1.0))

        fig.colorbar(images[0], orientation='vertical', cax=cax, label="Photons")

        fig.tight_layout()

        plt.savefig("./plots/box_{}.png".format(num_region))

        plt.cla()
        plt.clf()
        plt.close()

        num_region += 1

def plot_psf_blur(binned_count_map_fermi: str="/Volumes/T7/project_data/real_data/fermi_filtered_gti_binned_for_evaluation.fits"):

    plt.cla()
    plt.clf()
    plt.close()

    # NOTE THAT THIS IS JUST TAKING A LOW ENERGY bin AND NO TITLE INCLUDED

    Path("./plots").mkdir(parents=True, exist_ok=True)

    blank_count_map = np.zeros(hp.read_map(binned_count_map_fermi, hdu="SKYMAP", field=1).shape[0])

    pixel = hp.pixelfunc.ang2pix(nside=hp.pixelfunc.get_nside(blank_count_map), theta=90, phi=40, lonlat=True)

    blank_count_map[pixel] = 450

    binned_diffuse_source_psf = fit_diffuse_source_psf(roi_count_map="/Volumes/T7/project_data/real_data/diffuse_psf_roi/count_map.fits")

    count_map = (hp.sphtfunc.smoothing(map_in=blank_count_map, beam_window=binned_diffuse_source_psf[0]))

    count_map = np.clip(a=count_map, a_min=0, a_max=None)

    count_map = np.random.poisson(count_map)

    patch = cartesian_patch(count_map, lon=90, lat=40)

    # hp.newvisufunc.projview(
    #     count_map, coord=["G"], graticule=True, graticule_labels=True, xlabel="Galactic Longitude, $l$ [$\\degree$]",
    #     ylabel="Galactic Latitude, $b$ [$\\degree$]", unit="Photons", cb_orientation="vertical",
    #     projection_type="aitoff",
    # )

    plt.figure(figsize=(5, 4))

    plt.xticks([])
    plt.yticks([])

    plt.imshow(patch)

    plt.colorbar(label="Photon Count")

    plt.tight_layout()

    plt.savefig("./plots/psf_point_source_blur.png")

    plt.close()


def plot_integration_of_spectra():
    def agn_photon_flux(E, E_0=5752.19139, F_0=8.11 * 10 ** -13, alpha=1.7179, beta=0.027328):

        division = E / E_0

        exponent = - alpha - (beta * np.log(division))

        dF_dE = F_0 * np.power(division, exponent)

        return dF_dE

    a, b = 300, 200000  # integral limits
    x = np.linspace(300, 200000, num=1000)
    y = agn_photon_flux(x)

    fig, ax = plt.subplots()
    ax.plot(x, y, 'r', linewidth=2, color='gray')
    ax.set_ylim(bottom=0)


    # Make the shaded region
    intervals = np.logspace(np.log(a), np.log(b), num=7, base=np.e)

    print(intervals)

    colours = ['red', 'blue', 'yellow', 'green', 'purple']

    for i in range(5):

        ix = np.linspace(intervals[i], intervals[i + 1], num=1000)

        iy = agn_photon_flux(ix)
        verts = [(intervals[i], 0), *zip(ix, iy), (intervals[i + 1], 0)]
        poly = Polygon(verts, facecolor=colours[i], edgecolor='0.5', alpha=0.2)
        ax.add_patch(poly)

    # ax.text(0.5 * (a + b), 30, r"$\int_a^b f(x)\mathrm{d}x$",
    #         horizontalalignment='center', fontsize=20)

    fig.text(0.9, 0.05, '$x$')
    fig.text(0.1, 0.9, '$y$')

    ax.set_ylabel("$\\frac{dN}{dE}$ [ph cm$^2$s$^{-1}$MeV$^{-1}$]", fontsize=16)

    ax.set_xlabel("$E$ [MeV]", fontsize=13)

    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xticks(intervals, labels=np.rint(intervals).astype(int))
    ax.set_yticks([])

    ax.set_xlim(300, 20000)
    ax.set_ylim(0, 0.05 * 10 ** -10)

    plt.show()

def plot_fitted_point_source_psf(psf_file: str, function_parameters, directory: str):

    empirical = []
    energy_bins = []

    # Plot empirical data from gtpsf
    with fits.open(psf_file) as hdul:

        thetas = np.array([k[0] for k in hdul["THETA"].data])

        psf_data = hdul["PSF"].data

        num_bins = len(psf_data)

        for b in range(num_bins):
            # Lowest energy (MeV) of this bin
            energy_value = psf_data[b][0]

            # PSF values dP/dOmega - probability to find event in solid angle dOmega at offset r from point source
            psf_values = np.array(psf_data[b][2])

            psf_values = scale_psf(psf_values, energy_value)

            probs = normalise_psf(thetas, psf_values)

            empirical.append(probs)
            energy_bins.append(energy_value)

    # Create figure
    plt.rcParams["figure.figsize"] = (10, 14)

    fig, ax = plt.subplots((num_bins // 2) + (num_bins % 2), 2)

    # Plotting
    flattened_axes = ax.flatten()

    for x in range(num_bins):

        flattened_axes[x].scatter(thetas, empirical[x], marker='+', label="Empirical", color="blue")

        popt = function_parameters[x]

        flattened_axes[x].plot(thetas, dual_function(thetas, sigma_core=popt[0], gamma_core=popt[1], sigma_tail=popt[2],
                                                     gamma_tail=popt[3], f_core=popt[4]), label="Fitted King Function",
                               color="orange", linestyle='--')

        # Sample random values
        random_values = monte_carlo_sampler(popt, 5000)

        # Plot sampled values
        counts, bins = np.histogram(random_values, bins=200, density=True)
        flattened_axes[x].stairs(counts, bins, color="green", label="Random Samples")

    # Formatting

    if num_bins % 2 == 1:
        fig.delaxes(ax[-1, -1])

    labels = format_scientific_notation_label(energy_bins)

    for x in range(len(energy_bins)):

        a = flattened_axes[x]

        a.set_xlim(0, 5)
        a.set_ylim(0, )

        a.set_xlabel("Energy Scaled Angular Deviation of $\\gamma$-Ray, $x$ [$\\degree$]")
        a.set_ylabel("PSF($x, E$)")

        if x == len(energy_bins) - 1:
            a.set_title("PDF of Angular Deviation for $\\gamma$-Rays \nwith Energy {}+ MeV".format(labels[x]))
        else:
            a.set_title("PDF of Angular Deviation for $\\gamma$-Rays \nwith Energy {}-{} MeV".format(labels[x], labels[x+1]))

        a.legend()

    fig.suptitle("Point Spread Function Fitting")

    fig.tight_layout()

    fig.savefig(directory + "/fitted_point_source_psf.png")

    plt.close()


def plot_localisation_times():

    # https: // matplotlib.org / stable / gallery / lines_bars_and_markers / barchart.html

    # Increasing order of times
    algorithm = ["DBSCAN", "$k$-Means", "Blob Detection", "Spectral"]

    time_1 = [26.24301791191101, 1483.1209070682526, 12609.620107889175, 0] # REPLACE THE ZERO

    time_2 = [25.34082794189453, 1458.6058359146118, 12607.771565198898, 0]

    time_3 = [29.49886393547058, 7470.353355169296, 12587.637760162354, 0]

    average = [12600, 27, 3470] # FILL IN WITH SPECTRAL

    times = {
        "Applied to PSPNet Segmentation": (26.24301791191101, 1483.1209070682526, 12609.620107889175, 88219.83941984177), # REPLACE 0 with spectral
        "Applied to RF Segmentation": (25.34082794189453, 1458.6058359146118, 12607.771565198898, 56694.98472213745),
        "Applied to U-Net Segmentation": (29.49886393547058, 7470.353355169296, 12587.637760162354, 145519.62906694412),
        "Average": (27, 3470, 12600, 96811.484403)

    }

    clustering = ("DBSCAN", "Blob Detection", "$k$-Means", "Spectral")

    plt.rcParams["figure.figsize"] = (7, 5)

    fig, ax = plt.subplots(layout='constrained')

    ax.set_yscale('log')

    res = ax.grouped_bar(times, tick_labels=clustering, group_spacing=1, facecolor=("lightblue", "lightcoral", "palegreen", "palegoldenrod"), edgecolor='black', hatch=("/", "\\", "x", ""))

    # for container in res.bar_containers:

    # for container in res.bar_containers:
    #
    #     ax.bar_label(container, padding=3)

    ax.bar_label(res.bar_containers[-1], padding=3, fontsize=12, rotation=90)

    # Add some text for labels, title, etc.
    ax.set_ylabel('Time (s)', fontsize=14)
    ax.set_title('Execution Times of Localisation Algorithms', fontsize=18)
    ax.legend(loc='upper left', fontsize=13)
    ax.set_ylim(None, 750000)

    ax.tick_params(axis="both", which="major", labelsize=14)

    fig.savefig("./plots/test.png")




if __name__ == "__main__":

    plot_localisation_times()

    # plot_fermi_lat_count_map()
    #
    # plot_419()

    # plot_fermi_lat_exposure()
    #
    # # plot_psf_blur()
    #
    # # plot_integration_of_spectra()
    #
    # plot_exposure_map()

    # plot_simulated_count_map(logarithmic=True)


# REFERENCES

# Change Axis Size - https://www.geeksforgeeks.org/python/how-to-change-the-size-of-axis-labels-in-matplotlib/
# Colour Bar - https://stackoverflow.com/questions/13310594/positioning-the-colorbar
# Colour Bar - https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
# Figure Size - https://stackoverflow.com/questions/10540929/figure-of-imshow-is-too-small
# Hiding Xticks - https://www.geeksforgeeks.org/python/how-to-hide-axis-text-ticks-or-tick-labels-in-matplotlib/
# Matplotlib Polygons - https://stackoverflow.com/questions/65291917/how-to-change-the-opacity-transparency-alpha-of-
# patches-polygon-edge-lines
# Remove Axes Labels - https://stackoverflow.com/questions/9295026/how-to-remove-axis-legends-and-white-padding
# Shading - https://matplotlib.org/stable/gallery/showcase/integral.html
