from astropy.table import QTable
from itertools import combinations, product
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.stats import lognorm, norm
from .utils import log_normal_parameter
import seaborn as sns


# Used for labelling axes and titles - gives mathematical notation equivalent to variable
mathematical_notation = {"Pivot_Energy": "$E_0$", "LP_Flux_Density": "$F_0$", "LP_Index": "$\\alpha$",
                         "LP_beta": "$\\beta$", "PLEC_Flux_Density": "$F_0$", "PLEC_IndexS": "$\Gamma$",
                         "PLEC_Exp_Index": "$b$", "PLEC_ExpfactorS": "$a$", "GLAT": "Latitude"}


def plot_agn_luminosity_function(catalog: str, energy_fluxes: npt.NDArray[np.float64], directory) -> None:

    # Data Processing

    # Read 4FGL Catalog
    catalog = QTable.read(catalog, format='fits', hdu=1)['CLASS1', 'Energy_Flux100']

    # Reformat columns
    catalog['CLASS1'] = np.asarray([k.decode('utf-8').strip().lower() for k in catalog['CLASS1'].value.filled('-')])

    # Select all rows that describe AGN
    agn_mask = np.isin(catalog['CLASS1'].data, np.array(['bcu', 'sey', 'ssrq', 'bll', 'fsrq', 'rdg', 'nlsy1', 'agn']))
    agns = catalog[agn_mask]

    energy_fluxes_4fgl = agns['Energy_Flux100'].value

    # Plotting

    # Set plot size
    plt.rcParams["figure.figsize"] = (6.4, 4.8)

    # Plot 4FGL data
    bin_edges = 10**np.linspace(-14, -9, 50)
    counts, bins = np.histogram(energy_fluxes_4fgl, bins=bin_edges)
    plt.stairs(counts, bins, label='4FGL')

    # Plot simulated data
    bin_edges = 10**np.linspace(-15, -8, 50)
    counts, bins = np.histogram(energy_fluxes, bins=bin_edges)
    plt.stairs(counts, bins, label='Simulated')

    # Formatting

    plt.title('AGN Luminosity Function')

    plt.xlabel('Energy Flux')
    plt.ylabel('No. Sources')

    plt.xscale('log')
    plt.yscale('log')

    plt.xlim(10**-14, 10**-8)
    plt.ylim(top=10**4)

    plt.legend()

    plt.savefig(directory + "/4fgl_agn_luminosity_function.png")


def plot_correlation_matrices(sources, source_type, directory):

    # CREATE PLOT

    plt.rcParams["figure.figsize"] = (27, 11)

    fig, axs = plt.subplots(1, 2)

    # Calculate both Pearson and Kendall Rank correlation coefficients
    corr_pearson = sources.corr()
    corr_kendall = sources.corr(method='kendall')

    # Used for labelling matrix
    matrix_labels = [mathematical_notation[k] for k in corr_pearson.columns.values]

    # Plot matrices - took the absolute value of coefficients to highlight suggestions of strong correlation, but
    # continued to label with + and - indicating positive or negative correlation
    sns.heatmap(np.abs(corr_pearson), xticklabels=matrix_labels, yticklabels=matrix_labels, annot=corr_pearson,
                cmap='Greens', ax=axs[0])
    sns.heatmap(np.abs(corr_kendall), xticklabels=matrix_labels, yticklabels=matrix_labels, annot=corr_kendall,
                cmap='Blues', ax=axs[1])

    # FORMATTING

    axs[0].set_title('Pearson Correlation Coefficient Matrix for {} Parameters'.format(source_type), fontsize=20)
    axs[1].set_title('Kendall Rank Correlation Coefficient Matrix for {} Parameters'.format(source_type), fontsize=20)

    plt.savefig(directory + "/{}_parameter_correlation_matrix.png".format(source_type.lower()))


def plot_parameter_distributions(sources, source_type: str, directory: str):

    # Used to label axes
    axis_labels = {"LP_Flux_Density": "Differential Flux Density, $F_0$ \n [ph cm$^{-2}$ MeV$^{-1}$ s$^{-1}$",
                   "Pivot_Energy": "Pivot Energy, $E_0$ [MeV]", "LP_Index": "Spectral Slope, $\\alpha$",
                   "LP_beta": "Spectral Curvature, $\\beta$", "PLEC_IndexS": "Spectral Slope, $\\Gamma$",
                   "PLEC_Flux_Density": "Differential Flux Density, $F_0$ \n [ph cm$^{-2}$ MeV$^{-1}$ s$^{-1}$",
                   "PLEC_Exp_Index": "Exponential Index, $b$", "PLEC_ExpfactorS": "Exponential Factor, $a$",
                   "GLAT": "Latitude [$\degree$]"}

    num_parameters = sources.shape[1]

    # CREATE PLOTS

    # Calculate appropriate number of parameters
    num_cols = (num_parameters // 2) + (num_parameters % 2)
    num_rows = 2

    plt.rcParams["figure.figsize"] = (num_rows * 4, num_cols * 4)
    fig, ax = plt.subplots(num_cols, num_rows)

    if (num_parameters % 2) == 1:
        fig.delaxes(ax[num_cols - 1, 1])

    # GENERATE DATA

    for pair in zip(ax.flatten(), [sources[col] for col in sources]):

        subplot = pair[0]

        # Remove NaN values
        values = pair[1][~np.isnan(pair[1])]

        # Plot actual 4FGL source distribution
        counts, bins = np.histogram(values, bins=30, density=True)
        subplot.stairs(counts, bins, label='4FGL Distribution')

        # Calculate distribution parameters of data
        mean, sigma = np.mean(values), np.std(values, ddof=1)
        scale, s = log_normal_parameter(values)

        # Plot Gaussian using mean and standard deviation of data
        x_values = np.linspace(np.min(values), np.max(values), 1000)
        subplot.plot(x_values, norm.pdf(x_values, loc=mean, scale=sigma), label='Gaussian', linestyle='-.')

        # Plot Log-Normal distribution using mean and standard deviation of data (recommended by ID8 for F_0, but not
        # any of the other AGN parameters)
        subplot.plot(x_values, lognorm.pdf(x_values, s=s, scale=scale), color='red', label='Log-Normal',
                     linestyle='--')

        subplot.set_xlabel(axis_labels[values.name])

    # FORMATTING

    fig.suptitle('Distributions of 4FGL {} Parameters'.format(source_type))

    # Label axes and enable legends
    for a in ax.flatten():
        a.set_ylabel('Source Density')
        a.legend()

    fig.tight_layout()

    plt.savefig(directory + "/{}_parameter_distributions.png".format(source_type.lower()))


def plot_parameter_relationships(sources, source_type: str, directory: str):

    # Find all possible combinations of parameters
    variable_combinations = list(combinations(list(sources.columns), 2))

    # CREATE PLOT AND LOG-LOG PLOT

    plt.rcParams["figure.figsize"] = (10, 14)

    if source_type == "AGN":
        num_plot_cols = 2
    else:
        num_plot_cols = 3

    # Find number of rows of subplots that will be in figure
    num_plot_rows = len(variable_combinations) // num_plot_cols

    # Plot parameters against one another in fig1 and the log of parameters against one another in fig2
    fig, ax = plt.subplots(num_plot_rows, num_plot_cols)
    fig2, ax2 = plt.subplots(num_plot_rows, num_plot_cols)

    # Used to index subplots
    plot_indices = list(product(range(num_plot_rows), range(num_plot_cols)))

    # PLOT DATA

    num_subplots = len(plot_indices)

    # Plot data for each subplot
    for sp in range(num_subplots):

        row, col = plot_indices[sp][0], plot_indices[sp][1]
        var1, var2 = variable_combinations[sp][0], variable_combinations[sp][1]

        parameter_1 = sources[var1]
        parameter_2 = sources[var2]

        # Calculate correlation coefficients - check for non-linear relationship using Kendall correlation coefficient,
        # as Pearson only determines if linear relationship.
        correlation_coefficient = str(round(parameter_1.corr(parameter_2), 3))
        kendall_coefficient = str(round(parameter_1.corr(parameter_2, method='kendall'), 3))

        # Plot data
        ax[row, col].scatter(parameter_2, parameter_1, color='red', label="Pearson: " + correlation_coefficient
                                                                          + "\nKendall: " + kendall_coefficient,
                             marker='+', s=8)

        # Plot log-log data data
        log_var1 = np.log(parameter_1)
        log_var2 = np.log(parameter_2)

        ax2[row, col].scatter(log_var2, log_var1, color='red', label="Pearson: " + correlation_coefficient +
                                                                     "\nKendall: " + kendall_coefficient, marker='+',
                              s=8)

        # Subplot formatting
        for a in [ax, ax2]:

            a[row, col].set_title(mathematical_notation[var1] + ' against ' + mathematical_notation[var2], fontsize=12)
            ax2[row, col].set_xlabel(mathematical_notation[var2])
            ax2[row, col].set_ylabel(mathematical_notation[var1])

            ax2[row, col].legend(fontsize=8, loc='upper left')

    # Figure formatting

    fig.suptitle("Plotting {} Parameters Against Each Other".format(source_type), fontsize=18, y=0.98)
    fig.tight_layout()

    fig2.suptitle("Log-Log Plotting {} Parameters Against Each Other".format(source_type), fontsize=18, y=0.98)
    fig2.tight_layout()

    fig.savefig(directory + "/{}_parameter_relationships.png".format(source_type.lower()))
    fig2.savefig(directory + "/{}_logarithmic_parameter_relationships.png".format(source_type.lower()))


# REFERENCES

# Astropy Documentation - https://docs.astropy.org/en/stable/index_user_docs.html
# Log-Normals - https://stackoverflow.com/questions/68361048/how-to-generate-lognormal-distribution-with-specific-mean-
# and-std-in-python
# Numpy Documentation - https://numpy.org/doc/stable/index.html
# Numpy Typing - https://stackoverflow.com/questions/35673895/type-hinting-annotation-pep-484-for-numpy-ndarray
# Pandas Documentation - https://pandas.pydata.org/docs/index.html
# Scipy Documentation - https://docs.scipy.org/doc/scipy/index.html
