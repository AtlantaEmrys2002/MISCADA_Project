from astropy import units as u
from astropy.table import QTable
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.stats import Normal, chi2, lognorm, norm, kstest
from scipy.special import erf


def agn_luminosity_function(catalog: str, energy_fluxes: npt.NDArray[np.float64]) -> None:

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

    plt.show()


def normal_func(x, mean, sigma):

    var = sigma**2

    # return (1/np.sqrt(2 * np.pi * var)) * np.e**-(((x - mean)**2) / (2 * var))

    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(((x - mean) ** 2) / (2 * var)))


def log_norm_pdf(x, mu, sigma):

    upper = (np.log(x) - mu)**2
    lower = 2 * (sigma ** 2)
    fraction = -1 * upper / lower

    f_x = (1/(x * sigma * np.sqrt(2 * np.pi))) * np.exp(fraction)

    return f_x


def log_norm_cdf(x, mu, sigma):

    upper = np.log(x) - mu
    lower = sigma * np.sqrt(2)

    phi_x = 0.5 * (1 + erf(upper / lower))

    return phi_x


def chi_squared_goodness_of_fit_gaussian(observed_counts, bin_intervals, mean, sigma, number_of_sources):

    # Add 0 count for all < than bin_intervals[0] and for all > bin_intervals[len(bin_intervals)]
    observed_counts = list(observed_counts)
    observed_counts.insert(0, 0)
    observed_counts.append(0)

    X = Normal(mu=mean, sigma=sigma)

    num_bins = len(bin_intervals) - 1

    # Probability of parameter value falling into this interval
    # probs = [X.cdf(bin_intervals[x], bin_intervals[x + 1]) for x in range(0, len(bin_intervals) - 1)]

    probs = [X.cdf(bin_intervals[x], bin_intervals[x + 1]) for x in range(0, len(bin_intervals) - 1)]

    # Probability of parameter value falling below bin_intervals[0]
    probs.insert(0, X.cdf(bin_intervals[0]))

    # Probability of parameter value falling above bin_intervals[-1]
    probs.append(1 - X.cdf(bin_intervals[-1]))

    expected_counts = number_of_sources * np.array(probs)

    # if outliers cause 1-5 divide by 0 errors with a few observations, but have expected count of 0 take out the values
    if len(np.argwhere(np.logical_and((expected_counts == 0), (observed_counts != 0)).flatten())) < 5:

        observed_counts = np.array([observed_counts[x] for x in range(len(observed_counts)) if expected_counts[x] > 0])
        expected_counts = np.array([x for x in expected_counts if x > 0])

        # N.B. Both normal and log-normal have two free parameters to be fitted - mean and sigma

        degrees_of_freedom = num_bins - 2 - 1

        test_statistic = np.sum(((observed_counts - expected_counts) ** 2) / expected_counts)

        reduced_chi_squared_min = test_statistic/degrees_of_freedom

        # print("\n")
        # print("Chi Squared Min Test Statistic: {}".format(test_statistic))
        # print("Degrees of Freedom: {}".format(degrees_of_freedom))
        # print("Reduced Chi-Squared Min Statistic: {}".format(reduced_chi_squared_min))

        chi_squared_distribution = chi2(df=degrees_of_freedom)

        # Ideally about 0.5
        cdf_probability = 1 - chi_squared_distribution.cdf(test_statistic)

        # print("P(X^2_min; {}) = {}".format(degrees_of_freedom, cdf_probability))

        return str(test_statistic), str(reduced_chi_squared_min), str(cdf_probability), str(degrees_of_freedom)

    else:

        # print("Automatic rejection of distribution as chi^2 = infinity due to divide by 0 error.")

        return "N/A", "N/A", "N/A", "N/A"


def chi_squared_goodness_of_fit_log_normal(observed_counts, bin_intervals, number_of_sources, loc, scale):

    # Add 0 count for all < than bin_intervals[0] and for all > bin_intervals[len(bin_intervals)]
    observed_counts = list(observed_counts)
    observed_counts.insert(0, 0)
    observed_counts.append(0)
    num_bins = len(bin_intervals) - 1


    # Probability of parameter value falling into this interval

    probs = [log_norm_cdf(bin_intervals[x + 1], mu=loc, sigma=scale) - log_norm_cdf(bin_intervals[x], mu=loc, sigma=scale) for x in range(0, len(bin_intervals) - 1)]

    # Probability of parameter value falling below bin_intervals[0]
    probs.insert(0, log_norm_cdf(bin_intervals[0], mu=loc, sigma=scale))

    # print(probs)

    # Probability of parameter value falling above bin_intervals[-1]
    probs.append(1 - log_norm_cdf(bin_intervals[-1], mu=loc, sigma=scale))

    expected_counts = number_of_sources * np.array(probs)

    # if outliers cause 1-5 divide by 0 errors with a few observations, but have expected count of 0 take out the values
    if len(np.argwhere(np.logical_and((expected_counts == 0), (observed_counts != 0)).flatten())) < 5:

        observed_counts = np.array([observed_counts[x] for x in range(len(observed_counts)) if expected_counts[x] > 0])
        expected_counts = np.array([x for x in expected_counts if x > 0])

        # N.B. Both normal and log-normal have two free parameters to be fitted - mean and sigma

        degrees_of_freedom = num_bins - 2 - 1

        test_statistic = np.sum(((observed_counts - expected_counts) ** 2) / expected_counts)

        reduced_chi_squared_min = test_statistic/degrees_of_freedom

        chi_squared_distribution = chi2(df=degrees_of_freedom)

        # Ideally about 0.5
        cdf_probability = 1 - chi_squared_distribution.cdf(test_statistic)

        # print("CRITICAL VALUE: {}".format(print(chi2.ppf(0.1, 97))))

        return str(test_statistic), str(reduced_chi_squared_min), str(cdf_probability), str(degrees_of_freedom)

    else:

        # print("Automatic rejection of distribution as chi^2 = infinity due to divide by 0 error.")

        return "N/A", "N/A", "N/A", "N/A"


def kolmogorov_smirnov(observed_values, gaussian_mean, gaussian_std, lognorm_mean, lognorm_std, alpha=0.05):

    # Default test at 5%.

    print("----------------------------")
    print("Kolmogorov-Smirnov Test")

    print("GAUSSIAN")

    # Followed this - https://www.geeksforgeeks.org/machine-learning/kolmogorov-smirnov-test-ks-test/

    ks_stat, p_val = kstest(observed_values, 'norm', args=(gaussian_mean, gaussian_std))

    print("Test Stat: {}".format(ks_stat))
    print("p-value: {}".format(p_val))

    if alpha == 0.05:
        # https://real-statistics.com/statistics-tables/kolmogorov-smirnov-table/
        critical_val = 1.35810 / np.sqrt(len(observed_values))

    else:
        # 10% significance
        critical_val = 1.22385 / np.sqrt(len(observed_values))

    print("Critical Value: {}".format(critical_val))

    if ks_stat > critical_val or p_val < alpha:
        print('Gaussian not suitable')
    else:
        print("Gaussian suitable")

    print("LOG NORMAL")

    x = lognorm(s=lognorm_mean, scale=lognorm_std)

    # ks_stat, p_val = kstest(observed_values, 'lognorm', args=(lognorm_mean, lognorm_std))
    ks_stat, p_val = kstest(observed_values, x.cdf)


    print("Test Stat: {}".format(ks_stat))
    print("p-value: {}".format(p_val))

    alpha = 0.05

    # https://real-statistics.com/statistics-tables/kolmogorov-smirnov-table/
    critical_val = 1.35810 / np.sqrt(len(observed_values))

    print("Critical Value: {}".format(critical_val))

    if ks_stat > critical_val or p_val < alpha:
        print('LogNorm not suitable')
    else:
        print("LogNorm suitable")

    print("----------------------------")
    print('\n')


def analysing_agn_parameters(agns):

    # ID8 assert that F_0 follows log normal distribution and other params in differential energy flux follow Gaussian
    # we check this

    # PIVOT ENERGY ANALYSIS

    plt.rcParams["figure.figsize"] = (10, 10)

    fig, ax = plt.subplots(2, 2)

    # READ IN DATA

    fds = agns['LP_Flux_Density'].to(u.ph / (u.cm * u.cm * u.MeV * u.s)).value.filled(np.nan)
    pivot_energies = agns['Pivot_Energy'].value
    alphas = agns['LP_Index'].data.filled(np.nan)
    betas = agns['LP_beta'].data.filled(np.nan)

    # PLOT DISTRIBUTIONS

    for pair in zip(ax.flatten(), [fds, pivot_energies, alphas, betas]):

        subplot = pair[0]

        # Remove NaN values
        values = pair[1][~np.isnan(pair[1])]

        # Plot actual 4FGL source distribution
        counts, bins = np.histogram(values, bins=100, density=True)
        subplot.stairs(counts, bins, label='4FGL Distribution')

        # Calculate mean and standard deviation of data
        mean, sigma = np.mean(values), np.std(values, ddof=1)

        counts_2, bins_2 = np.histogram(values, bins=100, density=False)

        chi_squared, reduced_chi_squared, chi2_cdf_prob, degree_freedom = (
            chi_squared_goodness_of_fit_gaussian(counts_2, bins_2, mean, sigma, number_of_sources=len(values)))

        if chi_squared != "N/A":

            print("Goodness of Fit of Gaussian Distribution: ")

            print("Chi Squared Min Test Statistic: {}".format(chi_squared))
            print("Reduced Chi-Squared Min Statistic: {}".format(reduced_chi_squared))
            print("P(X^2_min; {}) = {}".format(degree_freedom, chi2_cdf_prob))

            print('\n')

        else:

            print("Automatic Rejection of Gaussian Fit")
            print("\n")

        # Plot Gaussian using mean and standard deviation of data
        x_values = np.linspace(np.min(values), np.max(values), 1000)
        y_values = normal_func(x_values, mean, sigma)
        subplot.plot(x_values, y_values, label='Gaussian', linestyle='-.')

        # Plot Log-Normal distribution using mean and standard deviation of data (recommended by ID8 for F_0, but not
        # any of the other AGN parameters)

        mean_square = mean ** 2
        std_square = sigma ** 2

        mean_log = np.log(mean_square / (np.sqrt(mean_square + std_square)))
        std_log = np.sqrt(np.log(1 + (std_square / mean_square)))

        chi_squared, reduced_chi_squared, chi2_cdf_prob, degree_freedom = (
            chi_squared_goodness_of_fit_log_normal(counts_2, bins_2, number_of_sources=len(values), loc=mean_log,
                                                   scale=std_log))

        if chi_squared != "N/A":

            print("Goodness of Fit of Log-Normal Distribution: ")

            print("Chi Squared Min Test Statistic: {}".format(chi_squared))
            print("Reduced Chi-Squared Min Statistic: {}".format(reduced_chi_squared))
            print("P(X^2_min; {}) = {}".format(degree_freedom, chi2_cdf_prob))

            print('\n')

        else:

            print("Automatic Rejection of Log-Normal Fit")
            print("\n")

        subplot.plot(x_values, log_norm_pdf(x_values, mean_log, std_log), label='Log-Normal', color='red',
                     linestyle='--')

        subplot.plot(x_values, lognorm.pdf(x_values, s=std_log, scale=np.exp(mean_log)), color='black', label='TEST',
                     linestyle=':')

        kolmogorov_smirnov(observed_values=values, gaussian_mean=mean, gaussian_std=sigma, lognorm_mean=std_log,
                           lognorm_std=np.exp(mean_log))

    # FORMATTING

    fig.suptitle('Distributions of 4FGL AGN Parameters')

    ax[0, 0].set_title('Differential Flux Densities, $F_0$')
    ax[0, 1].set_title('Pivot Energies, $E_0$')
    ax[1, 0].set_title('Spectral Slopes, $\\alpha$')
    ax[1, 1].set_title('Spectral Curvature, $\\beta$')

    ax[0, 0].set_xlabel('$F_0$ [ph cm$^{-2}$ MeV$^{-1}$ s$^{-1}$]')
    ax[0, 1].set_xlabel('$E_0$ [MeV]')
    ax[1, 0].set_xlabel('$\\alpha$')
    ax[1, 1].set_xlabel('$\\beta$')

    # Label axes and enable legends
    for a in ax.flatten():
        a.set_ylabel('Source Density')
        a.legend()

    fig.tight_layout()

    plt.show()


def analysing_pulsar_parameters(pulsars):

    # Create plot
    plt.rcParams["figure.figsize"] = (8, 12)
    fig, ax = plt.subplots(3, 2)
    fig.delaxes(ax[2, 1])

    # pivot_energies, flux_densities, spectral_slopes, exponential_indices, exponential_factors

    fds = pulsars['PLEC_Flux_Density'].to(u.ph / (u.cm * u.cm * u.MeV * u.s)).value.filled(np.nan)
    pivot_energies = pulsars['Pivot_Energy'].value
    Gammas = pulsars['PLEC_IndexS'].data.filled(np.nan)
    b_values = pulsars['PLEC_Exp_Index'].data.filled(np.nan)

    print(len(np.abs(fds - np.mean(fds)) < np.std(fds, ddof=1)))

    # Select exponential factors - CHECK (SAYS IN UNITS OF MeV^-b BUT NEED with GeV)
    a_values = pulsars['PLEC_ExpfactorS'].data

    for pair in zip(ax.flatten(), [fds, pivot_energies, Gammas, b_values, a_values]):

        subplot = pair[0]

        # Remove NaN values
        values = pair[1][~np.isnan(pair[1])]

        # Plot actual 4FGL source distribution
        counts, bins = np.histogram(values, bins=30, density=True)
        subplot.stairs(counts, bins, label='4FGL Distribution')

        # Calculate mean and standard deviation of data
        mean, sigma = np.mean(values), np.std(values, ddof=1)

        counts_2, bins_2 = np.histogram(values, bins=30, density=False)

        chi_squared, reduced_chi_squared, chi2_cdf_prob, degree_freedom = (
            chi_squared_goodness_of_fit_gaussian(counts_2, bins_2, mean, sigma, number_of_sources=len(values)))

        if chi_squared != "N/A":

            print("Goodness of Fit of Gaussian Distribution: ")

            print("Chi Squared Min Test Statistic: {}".format(chi_squared))
            print("Reduced Chi-Squared Min Statistic: {}".format(reduced_chi_squared))
            print("P(X^2_min; {}) = {}".format(degree_freedom, chi2_cdf_prob))

            print('\n')

        else:

            print("Automatic Rejection of Gaussian Fit")
            print("\n")

        # Plot Gaussian using mean and standard deviation of data
        x_values = np.linspace(np.min(values), np.max(values), 1000)
        y_values = normal_func(x_values, mean, sigma)
        subplot.plot(x_values, y_values, label='Gaussian', linestyle='-.')

        # Plot Log-Normal distribution using mean and standard deviation of data (recommended by ID8 for F_0, but not
        # any of the other AGN parameters)

        mean_square = mean ** 2
        std_square = sigma ** 2

        mean_log = np.log(mean_square / (np.sqrt(mean_square + std_square)))
        std_log = np.sqrt(np.log(1 + (std_square / mean_square)))

        chi_squared, reduced_chi_squared, chi2_cdf_prob, degree_freedom = (
            chi_squared_goodness_of_fit_log_normal(counts_2, bins_2, number_of_sources=len(values), loc=mean_log,
                                                   scale=std_log))

        if chi_squared != "N/A":

            print("Goodness of Fit of Log-Normal Distribution: ")

            print("Chi Squared Min Test Statistic: {}".format(chi_squared))
            print("Reduced Chi-Squared Min Statistic: {}".format(reduced_chi_squared))
            print("P(X^2_min; {}) = {}".format(degree_freedom, chi2_cdf_prob))

            print('\n')

        else:

            print("Automatic Rejection of Log-Normal Fit")
            print("\n")

        subplot.plot(x_values, log_norm_pdf(x_values, mean_log, std_log), label='Log-Normal', color='red',
                     linestyle='--')

        # subplot.plot(x_values, log_norm_pdf(x_values, mean_log, std_log), label='Log-Normal', color='red',
        #              linestyle='--')

        # TESTING - TAKE OUT BELOW LINES - FOUND IN BUILT LOGNORM THAT WORKS AFTER READING https://medium.com/data-bistrot/log-normal-distribution-with-python-7b8e384e939e

        # test_mean = np.nanmean(np.log(values))
        # test_std = np.nanstd(np.log(values), ddof=1)
        #
        # subplot.plot(x_values, log_norm_pdf(x_values, mu=test_mean, sigma=test_std), label='try')
        #
        # subplot.plot(x_values, lognorm.pdf(x_values, s=test_std, scale=np.exp(test_mean)), color='black', label='TEST', linestyle=':')

        subplot.plot(x_values, lognorm.pdf(x_values, s=std_log, scale=np.exp(mean_log)), color='black', label='TEST',
                     linestyle=':')

        kolmogorov_smirnov(observed_values=values, gaussian_mean=mean, gaussian_std=sigma, lognorm_mean=std_log,
                           lognorm_std=np.exp(mean_log))

    # FORMATTING

    fig.suptitle('Distributions of 4FGL Pulsar Parameters')

    ax[0, 0].set_title('Differential Flux Densities, $F_0$')
    ax[0, 1].set_title('Pivot Energies, $E_0$')
    ax[1, 0].set_title('Spectral Slopes, $\\Gamma$')
    ax[1, 1].set_title('Exponential Indices, $b$')
    ax[2, 0].set_title('Exponential Factors, $a$')

    ax[0, 0].set_xlabel('$F_0$ [ph cm$^{-2}$ MeV$^{-1}$ s$^{-1}$]')
    ax[0, 1].set_xlabel('$E_0$ [MeV]')
    ax[1, 0].set_xlabel('$\\Gamma$')
    ax[1, 1].set_xlabel('$b$')
    ax[2, 0].set_xlabel('$a\ [MeV$^{-b}$]$')

    ax[0, 0].set_ylim(0, 1 * 10 ** 11)

    # Label axes and enable legends
    for a in ax.flatten():
        a.set_ylabel('Source Density')
        a.legend()

    fig.tight_layout()

    plt.show()


# REFERENCES

# Astropy Documentation - https://docs.astropy.org/en/stable/index_user_docs.html
# Numpy Documentation - https://numpy.org/doc/stable/index.html
# Numpy Typing - https://stackoverflow.com/questions/35673895/type-hinting-annotation-pep-484-for-numpy-ndarray
