import numpy as np
from scipy.stats import chi2, kstest, lognorm, Normal


def chi_squared_test(values, num_bins: int, distribution: str) -> None:

    parameter_name = values.name

    # bin data and record number of agns with values within each interval
    observed_counts, bin_intervals = np.histogram(values, bins=num_bins, density=False)

    # used for creating specific distribution
    mean, sigma = np.mean(values), np.std(values, ddof=1)

    # create distribution to test observed values against
    match distribution:

        case "normal":

            X = Normal(mu=mean, sigma=sigma)

        case "lognorm":

            mean_square = mean ** 2
            std_square = sigma ** 2

            mean_log = np.log(mean_square / (np.sqrt(mean_square + std_square)))
            std_log = np.sqrt(np.log(1 + (std_square / mean_square)))

            X = lognorm(s=std_log, scale=np.exp(mean_log))

        case _:

            raise NameError("That probability distribution is not built in to chi_squared_test().")

    # add 0 count for all < than bin_intervals[0] and for all > bin_intervals[len(bin_intervals)]
    observed_counts = list(observed_counts)
    observed_counts.insert(0, 0)
    observed_counts.append(0)

    # probability of parameter value falling into this interval
    probs = list(X.cdf(bin_intervals[1:]) - X.cdf(bin_intervals[:-1]))

    # probability of parameter value falling below bin_intervals[0]
    probs.insert(0, X.cdf(bin_intervals[0]))

    # probability of parameter value falling above bin_intervals[-1]
    probs.append(1 - X.cdf(bin_intervals[-1]))

    number_of_sources = len(values)

    expected_counts = number_of_sources * np.array(probs)

    num_outliers = len(np.argwhere(np.logical_and((expected_counts == 0), (observed_counts != 0)).flatten()))

    # if outliers cause 1-5 divide by 0 errors with a few observations, but have expected count of 0 take out the values
    if num_outliers < 5:

        observed_counts = np.array([observed_counts[x] for x in range(len(observed_counts)) if expected_counts[x] > 0])
        expected_counts = np.array([x for x in expected_counts if x > 0])

        # n.b. both normal and log-normal have two free parameters to be fitted
        degrees_of_freedom = num_bins - 2 - 1

        test_statistic = np.sum(((observed_counts - expected_counts) ** 2) / expected_counts)

        reduced_chi_squared_min = test_statistic/degrees_of_freedom

        chi_squared_distribution = chi2(df=degrees_of_freedom)

        # Ideally about 0.5
        cdf_probability = 1 - chi_squared_distribution.cdf(test_statistic)

        print("-" * 60)
        print("Chi-Squared Goodness of Fit of {} Distribution to {}: ".format(distribution, parameter_name))

        print("Chi Squared Min Test Statistic, X^2_min: {}".format(test_statistic))
        print("P(X^2_min; {}) = {}".format(degrees_of_freedom, cdf_probability))

        print("Reduced Chi-Squared Min Statistic: {}".format(reduced_chi_squared_min))

        if cdf_probability < 0.01 or reduced_chi_squared_min > 2:
            print("Reject H_0: The {} distribution is not a good fit.".format(distribution))
        else:
            print("There is not sufficient evidence to reject {} distribution as a good fit.".format(distribution))

        print("-" * 60)

    else:

        print("-" * 60)
        print("The {} distribution cannot be fit, as there are too many intervals in which the observed count is zero\n"
              "and the expected count is non-zero.".format(distribution))
        print("-" * 60)


def kolmogorov_smirnov_test(values, distribution: str, alpha=0.05) -> None:

    # N.B. This tutorial was consulted when creating this function:
    # https://www.geeksforgeeks.org/machine-learning/kolmogorov-smirnov-test-ks-test/

    parameter_name = values.name

    # used for creating specific distribution
    mean, sigma = np.mean(values), np.std(values, ddof=1)

    match distribution:

        case "normal":

            ks_stat, p_val = kstest(values, 'norm', args=(mean, sigma))

        case "lognorm":

            mean_square = mean ** 2
            std_square = sigma ** 2
            mean_log = np.log(mean_square / (np.sqrt(mean_square + std_square)))
            std_log = np.sqrt(np.log(1 + (std_square / mean_square)))

            x = lognorm(s=std_log, scale=np.exp(mean_log))
            ks_stat, p_val = kstest(values, x.cdf)

        case _:

            raise NameError("That probability distribution is not built in to kolmogorov_smirnov_test().")

    number_of_sources = len(values)

    if alpha == 0.05:
        # 5% significance
        critical_val = 1.35810 / np.sqrt(number_of_sources)

    else:
        # 10% significance
        critical_val = 1.22385 / np.sqrt(number_of_sources)

    print("-" * 60)
    print("K-S Goodness of Fit of {} Distribution to {}: ".format(distribution, parameter_name))
    print("Test Stat, K: {}".format(ks_stat))
    print("P(K < {}) = {}".format(ks_stat, p_val))

    if ks_stat > critical_val or p_val < alpha:
        print("Reject H_0: The {} distribution is not a good fit at the {} % significance level."
              .format(distribution, alpha * 100))
    else:
        print("There is not sufficient evidence to reject {} distribution as a good fit at the {} % significance level."
              .format(distribution, alpha * 100))

    print("-" * 60)


# REFERENCES

# Adding Noise to Simulated Data - https://medium.com/@ms_somanna/guide-to-adding-noise-to-your-data-using-python-
# and-numpy-c8be815df524
# Appending Integers - https://stackoverflow.com/questions/17911091/append-integer-to-beginning-of-list-in-python
# Astropy Documentation - https://docs.astropy.org/en/stable/index_user_docs.html
# Chi-Squared Critical Value - https://www.statology.org/chi-square-critical-value-python/
# Chi-Squared Goodness of Fit - http://www.stat.yale.edu/Courses/1997-98/101/chigf.htm
# Chi-Squared Test - https://en.wikipedia.org/wiki/Chi-squared_test
# Chi-Squred with Zero Expected Counts - https://stats.stackexchange.com/questions/78101/chi-squared-test-with-0-
# expected-values
# Goodness of Fit - https://www.lboro.ac.uk/media/media/schoolanddepartments/mlsc/downloads/HELM%20Workbook%2042%20
# Goodness%20of%20Fit%20and%20Contingency%20Tables.pdf
# Kolmogorov-Smirnov Statistic - https://en.wikipedia.org/wiki/Kolmogorov–Smirnov_test
# K-S Table - https://real-statistics.com/statistics-tables/kolmogorov-smirnov-table/
# K-S Tutorial - https://www.geeksforgeeks.org/machine-learning/kolmogorov-smirnov-test-ks-test/
# Match in Python - https://www.w3schools.com/python/python_match.asp
# Numpy Documentation - https://numpy.org/doc/stable/index.html
# Python Documentation - https://docs.python.org/3/index.html
# Reduced Chi-Squared - https://en.wikipedia.org/wiki/Reduced_chi-squared_statistic
# Scipy Documentation - https://docs.scipy.org/doc/scipy/index.html
