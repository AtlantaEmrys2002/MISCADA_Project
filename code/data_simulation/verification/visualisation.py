from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable
from map_generation.visualisation import format_scientific_notation_label
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


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


def plot_correlation(catalog_4fgl: str, var1_name: str, var2_name: str, simulated_var1, simulated_var2,
                     source_type: str, directory: str) -> None:

    # VERIFY THAT THE SIMULATED CATALOG'S PARAMETERS HAVE SIMILAR CORRELATION TO THE ORIGINAL CATALOG

    # Read 4FGL Catalog
    catalog = QTable.read(catalog_4fgl, format='fits', hdu=1)["CLASS1", var1_name, var2_name]

    # Reformat columns
    catalog['CLASS1'] = np.asarray([k.decode('utf-8').strip().lower() for k in catalog['CLASS1'].value.filled('-')])

    if source_type == "AGN":

        # Select all rows that describe AGN
        agn_mask = np.isin(catalog['CLASS1'].data, np.array(['bcu', 'sey', 'ssrq', 'bll', 'fsrq', 'rdg', 'nlsy1', 'agn']))
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


def plot_luminosity_function(catalog: str, energy_fluxes: npt.NDArray[np.float64], directory: str,
                             source_type: str) -> None:
    # Data Processing

    # Read 4FGL Catalog
    catalog = QTable.read(catalog, format='fits', hdu=1)['CLASS1', 'Energy_Flux100']

    # Reformat columns
    catalog['CLASS1'] = np.asarray([k.decode('utf-8').strip().lower() for k in catalog['CLASS1'].value.filled('-')])

    if source_type == "AGN":

        # Select all rows that describe AGN
        agn_mask = np.isin(catalog['CLASS1'].data, np.array(['bcu', 'sey', 'ssrq', 'bll', 'fsrq', 'rdg', 'nlsy1', 'agn']))
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
    bin_edges = 10 ** np.linspace(-14, -9, 50)
    counts, bins = np.histogram(energy_fluxes_4fgl, bins=bin_edges)
    ax.stairs(counts, bins, label='4FGL')

    # Plot simulated data
    bin_edges = 10 ** np.linspace(-15, -8, 50)
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

# Astropy Documentation - https://docs.astropy.org/en/stable/index_user_docs.html
# Numpy Documentation - https://numpy.org/doc/stable/index.html
# Numpy Typing - https://stackoverflow.com/questions/35673895/type-hinting-annotation-pep-484-for-numpy-ndarray
# Pandas Documentation - https://pandas.pydata.org/docs/index.html
# Scipy Documentation - https://docs.scipy.org/doc/scipy/index.html
