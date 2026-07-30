from astropy.io import fits
from astropy.table import QTable
from healpy.newvisufunc import projview
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pathlib import Path
from psfs.fit_psf import normalise_psf, scale_psf
from psfs.utils import dual_function, monte_carlo_sampler
import warnings

axis_labels = {"LP_Flux_Density": "Differential Flux Density",
               "Pivot_Energy": "Pivot Energy", "LP_Index": "Spectral Slope",
               "LP_beta": "Spectral Curvature", "PLEC_IndexS": "Spectral Slope",
               "PLEC_Flux_Density": "Differential Flux Density", "PLEC_Exp_Index": "Exponential Index",
               "PLEC_ExpfactorS": "Exponential Factor", "GLAT": "Latitude"}

# Used for mathematical descriptions - gives mathematical notation equivalent to variable
mathematical_notation = {"Pivot_Energy": "$E_0$", "LP_Flux_Density": "$F_0$", "LP_Index": "$\\alpha$",
                         "LP_beta": "$\\beta$", "PLEC_Flux_Density": "$F_0$", "PLEC_IndexS": "$\\Gamma$",
                         "PLEC_Exp_Index": "$b$", "PLEC_ExpfactorS": "$a$", "GLAT": "Latitude"}

# Used for indicating units
units = {"LP_Flux_Density": "[ph cm$^{-2}$ MeV$^{-1}$ s$^{-1}$", "Pivot_Energy": "[MeV]", "LP_Index": "",
         "LP_beta": "",
         "PLEC_IndexS": "", "PLEC_Flux_Density": "[ph cm$^{-2}$ MeV$^{-1}$ s$^{-1}$", "PLEC_Exp_Index": "",
         "PLEC_ExpfactorS": "", "GLAT": "[rad]"}


def format_scientific_notation_label(numbers) -> list[str]:
    """Used for formatting very large or very small numbers into scientific notation, such that they can be neatly
    plotted on graphs.

    Parameters
    ----------
    numbers
        Values to format into scientific notation

    Returns
    -------
    list
        Labels in scientific notation and as strings

    """
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


def plot_all_sky_map(healpix_maps, energy_bins, title: str, directory: str, logarithmic=False):
    # Create directory to store results
    Path(directory).mkdir(parents=True, exist_ok=True)

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
        label = "{} Map for {}{} - {} MeV".format(title, new_line, energy_bin_labels[b], energy_bin_labels[b + 1])

        projview(
            data[b], coord=["G"], graticule=True, graticule_labels=True, xlabel="Galactic Longitude [$\\degree$]",
            ylabel="Galactic Latitude [$\\degree$]", cb_orientation="vertical", projection_type="aitoff", title=label,
            sub=(3, 2, b + 1),
        )

    plt.tight_layout()

    plt.savefig(directory + title.lower().replace(" ", "_") + "_map.png")

    plt.close()


def plot_correlation(catalog_4fgl: str, var1_name: str, var2_name: str, simulated_var1, simulated_var2,
                     source_type: str, directory: str) -> None:
    # VERIFY THAT THE SIMULATED CATALOG'S PARAMETERS HAVE SIMILAR CORRELATION TO THE ORIGINAL CATALOG

    # Read 4FGL Catalog
    catalog = QTable.read(catalog_4fgl, format='fits', hdu=1)["CLASS1", var1_name, var2_name]

    # Reformat columns
    catalog['CLASS1'] = np.asarray([k.decode('utf-8').strip().lower() for k in catalog['CLASS1'].value.filled('-')])

    if source_type == "AGN":

        # Select all rows that describe AGN
        agn_mask = np.isin(catalog['CLASS1'].data, np.array(['bcu', 'sey', 'ssrq', 'bll', 'fsrq', 'rdg', 'nlsy1',
                                                             'agn']))
        sources_4fgl = catalog[agn_mask]

    else:

        # Select all rows that describe pulsars
        pulsar_mask = (catalog["CLASS1"] == "psr")
        sources_4fgl = catalog[pulsar_mask]

    plt.rcParams["figure.figsize"] = (10, 4)

    fig, ax = plt.subplots(1, 2)

    # Plot original correlation

    original_var1, original_var2 = sources_4fgl[var1_name].value, sources_4fgl[var2_name].value

    ax[0].scatter(original_var1, original_var2, label="4FGL", color='blue', alpha=0.6)
    ax[1].scatter(np.log(original_var1), np.log(original_var2), label="4FGL", color='blue', alpha=0.6)

    # Plot simulated data's correlation

    ax[0].scatter(simulated_var1, simulated_var2, color='orange', alpha=0.6, label="Simulated")
    ax[1].scatter(np.log(simulated_var1), np.log(simulated_var2), color='orange', alpha=0.6, label="Simulated")

    # Formatting

    fig.suptitle("Comparison of Correlation between 4FGL and Simulated {} {} and {}".format(source_type,
                                                                                            axis_labels[var2_name],
                                                                                            axis_labels[var1_name]))

    ax[0].set_title("{} against {}".format(mathematical_notation[var2_name], mathematical_notation[var1_name]))
    ax[1].set_title("log {} against log {}".format(mathematical_notation[var2_name], mathematical_notation[var1_name]))

    ax[0].set_xlabel("{} {}".format(mathematical_notation[var1_name], units[var1_name]))
    ax[0].set_ylabel("{} {}".format(mathematical_notation[var2_name], units[var2_name]))

    ax[1].set_xlabel("log {}".format(mathematical_notation[var1_name]))
    ax[1].set_ylabel("log {}".format(mathematical_notation[var2_name]))

    ax[0].legend()
    ax[1].legend()

    fig.savefig(directory + "/correlation_between_{}_{}_and_{}.png".format(source_type.lower(), var1_name, var2_name))

    plt.close()


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
        counts, bins = np.histogram(random_values, bins=250, density=True)
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


def plot_luminosity_function(catalog: str, energy_fluxes: npt.NDArray[np.float64], directory: str,
                             source_type: str) -> None:
    # Data Processing

    # Read 4FGL Catalog
    catalog = QTable.read(catalog, format='fits', hdu=1)['CLASS1', 'Energy_Flux100']

    # Reformat columns
    catalog['CLASS1'] = np.asarray([k.decode('utf-8').strip().lower() for k in catalog['CLASS1'].value.filled('-')])

    if source_type == "AGN":

        # Select all rows that describe AGN
        agn_mask = np.isin(catalog['CLASS1'].data, np.array(['bcu', 'sey', 'ssrq', 'bll', 'fsrq', 'rdg', 'nlsy1',
                                                             'agn']))
        sources = catalog[agn_mask]

    else:

        # Select all rows that describe pulsars
        pulsar_mask = (catalog["CLASS1"] == "psr")
        sources = catalog[pulsar_mask]

    energy_fluxes_4fgl = sources['Energy_Flux100'].value

    # Plotting

    # Set plot size
    plt.rcParams["figure.figsize"] = (6.4, 4.8)

    fig, ax = plt.subplots(1, 1)

    # Plot 4FGL data
    bin_edges = 10 ** np.linspace(-14, -9)
    counts, bins = np.histogram(energy_fluxes_4fgl, bins=bin_edges)
    ax.stairs(counts, bins, label='4FGL')

    # Plot simulated data
    bin_edges = 10 ** np.linspace(-15, -8)
    counts, bins = np.histogram(energy_fluxes, bins=bin_edges)
    ax.stairs(counts, bins, label='Simulated')

    # Formatting

    fig.suptitle("{} Luminosity Function".format(source_type))

    ax.set_xlabel('Energy Flux')
    ax.set_ylabel('No. Sources')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(10 ** -14, 10 ** -8)
    ax.set_ylim(top=10 ** 4)

    ax.legend()

    fig.savefig(directory + "/4fgl_{}_luminosity_function.png".format(source_type.lower()))

    plt.close()


def plot_patch(binned_patches, mask, unformatted_energy_bins, directory):
    # PLOT ENERGY BINNED COUNT MAPS AND CORRESPONDING MASK

    formatted_energy_bins = format_scientific_notation_label(unformatted_energy_bins)

    plt.rcParams["figure.figsize"] = (15, 10)

    fig, ax = plt.subplots(nrows=2, ncols=3)

    axes = ax.flatten()

    num_bins = unformatted_energy_bins.shape[0] - 1

    for a in range(num_bins):
        im = axes[a].imshow(binned_patches[a])

        axes[a].set_title("Count Map {} MeV - {} MeV".format(formatted_energy_bins[a], formatted_energy_bins[a + 1]))

        fig.colorbar(im, ax=axes[a])

    axes[-1].imshow(mask)

    fig.suptitle("Binned Patch Count Map")
    fig.tight_layout()

    plt.savefig(directory + "/patch_visualised.png")


def plot_spatial_distribution(galactic_longitudes, galactic_latitudes, source_type: str, directory: str):
    # N.B. longitudes and latitudes should be passed to this function in radians (not in degrees)

    xs = galactic_longitudes
    ys = galactic_latitudes

    # CREATE FIGURE

    fig, ax = plt.subplots(figsize=(8, 4.2), subplot_kw=dict(projection="aitoff"))

    # Plot
    ax.scatter(xs, ys, marker='o', s=2, alpha=0.5)

    # FORMATTING

    ax.set_title("Spatial Distribution of Simulated " + source_type + " on the Sky", pad=20)
    ax.grid(True)
    fig.subplots_adjust(top=0.95, bottom=0.0)

    fig.savefig(directory + "/simulated_{}_spatial_distributions.png".format(source_type.lower()))

    plt.close()

# REFERENCES

# Deleting Subplot - https://stackoverflow.com/questions/14694501/delete-a-subplot
# Numpy Typing - https://stackoverflow.com/questions/35673895/type-hinting-annotation-pep-484-for-numpy-ndarray
# Odd Number Subplots - https://stackoverflow.com/questions/28738836/how-to-create-an-odd-number-of-subplots
# Scientific Notation in Plots - https://stackoverflow.com/questions/46735745/how-to-control-scientific-notation-in-
# matplotlib
