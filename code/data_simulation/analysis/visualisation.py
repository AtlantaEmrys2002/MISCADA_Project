"""
Functions for visualising the analysis conducted on 4FGL sources' spectral and spatial parameters.
"""

from itertools import combinations, product
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import cauchy, lognorm, norm
from .utils import log_normal_parameter
import seaborn as sns
from sklearn.metrics import root_mean_squared_error

# Used to label axes
axis_labels = {"LP_Flux_Density": "Differential Flux Density",
               "Pivot_Energy": "Pivot Energy", "LP_Index": "Spectral Slope",
               "LP_beta": "Spectral Curvature", "PLEC_IndexS": "Spectral Slope",
               "PLEC_Flux_Density": "Differential Flux Density", "PLEC_Exp_Index": "Exponential Index",
               "PLEC_ExpfactorS": "Exponential Factor", "GLAT": "Latitude"}

# Used for mathematical descriptions - gives mathematical notation equivalent to variable
mathematical_notation = {"Pivot_Energy": "$E_0$", "LP_Flux_Density": "$F_0$", "LP_Index": "$\\alpha$",
                         "LP_beta": "$\\beta$", "PLEC_Flux_Density": "$F_0$", "PLEC_IndexS": "$\Gamma$",
                         "PLEC_Exp_Index": "$b$", "PLEC_ExpfactorS": "$a$", "GLAT": "Latitude"}

# Used for indicating units
units = {"LP_Flux_Density": "[ph cm$^{-2}$ MeV$^{-1}$ s$^{-1}$", "Pivot_Energy": "[MeV]", "LP_Index": "",
         "LP_beta": "",
         "PLEC_IndexS": "", "PLEC_Flux_Density": "[ph cm$^{-2}$ MeV$^{-1}$ s$^{-1}$", "PLEC_Exp_Index": "",
         "PLEC_ExpfactorS": "", "GLAT": "[rad]"}


def plot_correlation_matrices(sources, source_type: str, directory: str):
    """Plot all possible parameter combinations' correlations - both Pearson (for linear) and Kendall (for non-linear)
    coefficients should be shown in the same plot.

    Parameters
    ----------
    sources
        Columns of spectral and spatial parameters in frame
    source_type : str
        Type of gamma-ray source being analysed (used in figure title)
    directory : str
        File in which to store final plot
    """
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

    plt.close()


def plot_fitting_correlated_variable_dependency(var1, var2, source_type: str, directory: str, logarithmic_fit=True):
    """Function attempts to identify any correlations between any spectral and spatial parameter combinations - these
    correlations may take the form of polynomials relating the logarithm of both parameters (this polynomial may be
    linear, quadratic, cubic, quartic). It also attempts to identify any directly logarithmic relationships.

    Parameters
    ----------
    var1
        The values of a given spectral parameter for each of the sources
    var2
        The values of another given spectral parameter for each of the sources
    source_type : str
        Type of gamma-ray source being analysed (used in figure title)
    directory : str
        File in which to store final plot
    logarithmic_fit : bool, optional
        Indicates whether to attempt a logarithmic fit to the parameters

    """
    # N.B. when calculating the least squares fit - in case variables need to be normally distributed, we know that log
    # of E_0 and log of F_0 are both normally distributed (log-normal distribution)

    print("{} - {} Relationship".format(axis_labels[var1.name], axis_labels[var2.name]))
    print('-' * 60)

    plt.rcParams["figure.figsize"] = (10, 5)

    log_var1, log_var2 = np.log(var1), np.log(var2)

    # Create plot

    fig, ax = plt.subplots(1, 2)

    # Plot data

    ax[0].scatter(var1, var2, s=4, marker='+')
    ax[1].scatter(log_var1, log_var2, s=4, marker="+")

    # Fitting

    x_values = np.linspace(np.min(var1), np.max(var1), 1000)
    log_x_values = np.log(x_values)

    # Formatting information
    if logarithmic_fit is True:

        labels = ["Linear", "Quadratic", "Cubic", "Quartic", "Logarithmic"]

    else:

        labels = ["Linear", "Quadratic", "Cubic", "Quartic"]

    colours = ["red", "green", "orange", "purple"]

    linestyles = ['-.', ':', '--', (0, (3, 1, 1, 1, 1, 1))]

    residuals = []

    for degree in range(1, 5):
        idx = degree - 1

        polynomial = np.polynomial.Polynomial.fit(log_var1, log_var2, deg=degree)

        ax[0].plot(x_values, np.exp(polynomial(log_x_values)), color=colours[idx], label="Log {}".format(labels[idx]),
                   linestyle=linestyles[degree - 1])
        ax[1].plot(log_x_values, polynomial(log_x_values), color=colours[idx], label=labels[idx],
                   linestyle=linestyles[degree - 1])

        polynomial_log = polynomial(log_var1)

        # Prints the coefficient values of the best fitting polynomial to the parameters
        print(["{} : {}".format(chr((degree - x) + 97), polynomial.convert().coef[x]) for x in range(degree, -1, -1)])
        print("RMSE of {} Fit: {}".format(labels[idx], root_mean_squared_error(log_var2, polynomial_log)))

        residuals.append(log_var2 - polynomial_log)
        normalised_residual = (log_var2 - polynomial_log) / np.std(polynomial_log, ddof=1)

        # From Measurements and Their Uncertainties - aim for 96% of normalised residuals to lie between -2 and +2
        percentage = np.sum(np.abs(normalised_residual) > 2) / len(normalised_residual)

        print("Percentage of normalised residuals in [-2, 2]: {}".format(1 - percentage))
        print("Standard Deviation of Non-Normalised Residuals: {}".format(np.std(log_var2 - polynomial_log, ddof=1)))

        print('-' * 60)

    # Logarithmic Fit

    if logarithmic_fit is True:
        # This paper states that the spectral index depends linearly on ln E
        # - https://journals-aps-org.ezphost.dur.ac.uk/prd/abstract/10.1103/k5dp-5str

        polynomial = np.polynomial.Polynomial.fit(log_var1, var2, deg=1)
        ax[0].plot(x_values, polynomial(log_x_values), color="black", label="Logarithmic")
        ax[1].plot(log_x_values, np.log(polynomial(log_x_values)), color="black", label="Logarithmic")

        logarithmic_coefficients = polynomial.convert().coef

        print("a: {} b: {}".format(logarithmic_coefficients[1], logarithmic_coefficients[0]))

        log_polynomial = np.log(polynomial(log_var1))

        print("RMSE of Logarithmic Fit: {}".format(root_mean_squared_error(log_var2, log_polynomial)))

        residuals.append(log_var2 - log_polynomial)

        # From Measurements and Their Uncertainties - aim for 96% of normalised residuals to lie between -2 and +2
        normalised_residual = (log_var2 - log_polynomial) / np.std(log_polynomial, ddof=1)
        percentage = np.sum(np.abs(normalised_residual) > 2) / len(normalised_residual)
        print("Percentage of normalised residuals in [-2, 2]: {}".format(1 - percentage))
        print("Standard Deviation of Non-Normalised Residuals: {}".format(np.std(log_polynomial, ddof=1)))

        print('-' * 60)

    # Formatting

    ax[0].set_xlabel("{} {}".format(mathematical_notation[var1.name], units[var1.name]))
    ax[0].set_ylabel("{} {}".format(mathematical_notation[var2.name], units[var2.name]))

    title_0 = "{} vs {}".format(axis_labels[var1.name], axis_labels[var2.name])

    ax[0].set_title(title_0)
    ax[0].legend()

    if len(axis_labels[var1.name]) + len(axis_labels[var2.name]) > 20:
        new_line = '\n'
    else:
        new_line = ''

    title_1 = "Log-Log Plot of {} {} vs {}".format(new_line, axis_labels[var1.name], axis_labels[var2.name])

    ax[1].set_xlabel("log {}".format(mathematical_notation[var1.name]))
    ax[1].set_ylabel("log {}".format(mathematical_notation[var2.name]))
    ax[1].set_title(title_1)
    ax[1].legend()

    fig.suptitle("Fitting {} {} - {} Dependency".format(source_type, mathematical_notation[var1.name],
                                                        mathematical_notation[var2.name]))

    fig.tight_layout()

    fig.savefig(directory + "/fitted_{}_{}_{}_relationship.png".
                format(source_type.lower(), axis_labels[var1.name].lower().replace(" ", "_"),
                       axis_labels[var2.name].lower().replace(" ", "_")))

    plt.close()

    # Plot residuals

    plt.rcParams["figure.figsize"] = (25, 5)

    fig2, ax2 = plt.subplots(1, len(labels))

    for a in range(len(residuals)):
        residuals_std = np.std(residuals[a], ddof=1)

        normalised_residuals = residuals[a] / residuals_std

        ax2[a].scatter(log_var1, normalised_residuals, s=4, label="$\sigma =$ {0:.3f}".format(residuals_std))

        # Formatting
        ax2[a].set_title("{} Fit to Log-Log Plot".format(labels[a]))
        ax2[a].set_xlabel("log {}".format(mathematical_notation[var1.name]))
        ax2[a].set_ylabel("$y_i - \hat{y}_i$")
        ax2[a].legend()

    # FORMATTING

    fig2.suptitle('Residuals')

    fig2.tight_layout()

    (fig2.savefig
     (directory + "/fitted_{}_{}_{}_relationship_residuals.png"
      .format(source_type.lower(), axis_labels[var1.name].lower().replace(" ", "_"),
              axis_labels[var2.name].lower().replace(" ", "_"))))

    plt.close()


def plot_parameter_distributions(sources, source_type: str, directory: str):
    """Plots the histogram a parameter's binned values and then over-plots common PDFs to determine the best method for
    sampling a realistic population of sources.

    Parameters
    ----------
    sources
        Columns of spectral and spatial parameters in frame
    source_type : str
        Type of gamma-ray source being analysed (used in figure title)
    directory : str
        File in which to store final plot

    """
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

        cauchy_params = cauchy.fit(values, floc=0)

        subplot.plot(x_values, cauchy.pdf(x_values, *cauchy_params), label="Cauchy", color='green')

        # Long line
        long_line = "\n"

        if len(axis_labels[values.name]) < 20:
            long_line = ""

        subplot.set_xlabel("{} {}{}".format(axis_labels[values.name], long_line, units[values.name]))

    # FORMATTING

    fig.suptitle('Distributions of 4FGL {} Parameters'.format(source_type))

    # Label axes and enable legends
    for a in ax.flatten():
        a.set_ylabel('Source Density')
        a.legend()

    fig.tight_layout()

    plt.savefig(directory + "/{}_parameter_distributions.png".format(source_type.lower()))

    plt.close()


def plot_parameter_relationships(sources, source_type: str, directory: str):
    """Plots relationships of each possible parameter combination - this is to complement the correlation coefficients
    to ensure there is no mistake in determining two parameters' correlation (for example, the curvature of an AGN's
    spectrum is not correlated with other parameters - it can just only have certain values and 80% of those values are
    the same, indicating correlation where there isn't any).

    Parameters
    ----------
    sources
        Columns of spectral and spatial parameters in frame
    source_type : str
        Type of gamma-ray source being analysed (used in figure title)
    directory : str
        File in which to store final plot

    """
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

            a[row, col].set_xlabel("{} {}".format(mathematical_notation[var2], units[var2]))
            a[row, col].set_ylabel("{} {}".format(mathematical_notation[var1], units[var1]))

            a[row, col].legend(fontsize=8, loc='upper left')

    # Figure formatting

    fig.suptitle("Plotting {} Parameters Against Each Other".format(source_type), fontsize=18, y=0.98)
    fig.tight_layout()

    fig2.suptitle("Log-Log Plotting {} Parameters Against Each Other".format(source_type), fontsize=18, y=0.98)
    fig2.tight_layout()

    fig.savefig(directory + "/{}_parameter_relationships.png".format(source_type.lower()))
    fig2.savefig(directory + "/{}_logarithmic_parameter_relationships.png".format(source_type.lower()))

    plt.close()

# REFERENCES

# Alphabet-ASCII Relation - https://stackoverflow.com/questions/4528982/convert-alphabet-letters-to-number-in-python
# Astropy Documentation - https://docs.astropy.org/en/stable/index_user_docs.html
# Cauchy Distribution - https://en.wikipedia.org/wiki/Cauchy_distribution
# Closing Plot - https://stackoverflow.com/questions/741877/how-do-i-tell-matplotlib-that-i-am-done-with-a-plot
# Correlation Analysis - https://en.wikipedia.org/wiki/Covariance#Examples
# Covariance - https://en.wikipedia.org/wiki/Covariance_matrix
# Figure Sizing - https://stackoverflow.com/questions/332289/how-do-i-change-the-size-of-figures-drawn-with-matplotlib
# Fitting Log-normals - https://stackoverflow.com/questions/18534562/scipy-lognormal-fitting
# Fitting Noisy Data - https://stackoverflow.com/questions/49201515/fit-numpy-polynomials-to-noisy-data
# Fitting Recommendations - https://dataviz.shef.ac.uk/docs/18/03/2021/LearningPath-Statistical-Modeling-1
# Gaussian Fitting Sharp Peak - https://stackoverflow.com/questions/74146895/gaussian-fitting-of-a-sharply-peaked-curve
# Gaussian Mixture - https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
# Kendall Correlation - https://numiqo.com/tutorial/kendalls-tau
# Kendall Rank Correlation - https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
# Logistic Distribution - https://stackoverflow.com/questions/78113609/how-to-fit-a-logistic-distribution-use-a-fixed-
# location-parameter
# Logistic Distribution Fitting - https://stackoverflow.com/questions/78113609/how-to-fit-a-logistic-distribution-use-a-
# fixed-location-parameter
# Log-Normals - https://stackoverflow.com/questions/68361048/how-to-generate-lognormal-distribution-with-specific-mean-
# and-std-in-python
# Log-Normal Error - https://stackoverflow.com/questions/65302332/scipy-stats-lognorm-expect-returning-an-odd-result
# Log-Normal from Measurements - https://deltares.github.io/GEOLib-Plus/latest/community/probabilistic_macrostability/
# Tutorial_1_Fit_Lognormal_Distribution.html
# Log-Log Plot Fits - https://stackoverflow.com/questions/47226600/plot-straight-line-of-best-fit-on-log-log-plot
# Normalising Histograms - https://stackoverflow.com/questions/35482543/normalizing-histograms
# Normalised/Studentised Residual - https://stats.stackexchange.com/questions/22653/raw-residuals-versus-standardised-
# residuals-versus-studentised-residuals-what
# Numpy Documentation - https://numpy.org/doc/stable/index.html
# Numpy Typing - https://stackoverflow.com/questions/35673895/type-hinting-annotation-pep-484-for-numpy-ndarray
# Pandas Documentation - https://pandas.pydata.org/docs/index.html
# Pearson Correlation - https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
# Plotting Correlation Matrices - https://stackoverflow.com/questions/29432629/plot-correlation-matrix-using-pandas
# Polynomial Fit Coefficients - https://stackoverflow.com/questions/67371614/numpy-polynomial-polynomial-fit-gives-
# different-coefficients-than-polynomial-p
# PDFs - https://www.lesswrong.com/posts/jmq3mon8TSC99ittm/common-probability-distributions
# Scipy Documentation - https://docs.scipy.org/doc/scipy/index.html
# Seaborn Heatmaps - https://stackoverflow.com/questions/50947776/plot-two-seaborn-heatmap-graphs-side-by-side
# Shot Noise - https://en.wikipedia.org/wiki/Shot_noise
# Split-Normal Distributions - https://en.wikipedia.org/wiki/Split_normal_distribution
