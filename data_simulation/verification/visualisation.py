from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def plot_luminosity_function(catalog: str, energy_fluxes: npt.NDArray[np.float64], directory: str,
                                 source_type="AGN") -> None:

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
    bin_edges = 10**np.linspace(-14, -9, 50)
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
