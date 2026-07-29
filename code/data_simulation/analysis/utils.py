"""
Utility functions for fitting probability density functions to 4FGL parameter distributions.
"""

import numpy as np


def log_normal_parameter(values):
    """Method for calculating distribution parameters that can be passed to scipy/numpy lognorm function to create
    distribution with mean and std of values passed to function.

    Parameters
    ----------
    values
        Series of values whose mean and standard deviation will be fed to lognorm function to create distribution
        resembling that of these values.

    """
    # This calculates the parameters for creating a log normal distribution based on parameter data

    mean, sigma = np.mean(values), np.std(values, ddof=1)

    mean_square = mean ** 2
    std_square = sigma ** 2

    mean_log = np.log(mean_square / (np.sqrt(mean_square + std_square)))
    std_log = np.sqrt(np.log(1 + (std_square / mean_square)))

    return np.exp(mean_log), std_log
