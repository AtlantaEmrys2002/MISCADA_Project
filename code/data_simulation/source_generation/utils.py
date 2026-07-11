import numpy as np


def split_normal(x, sigma_1, sigma_2):

    # I believe this is what ID8 was suggesting - but they were too vague, so I implemented this myself.

    mu = 0

    upper = -1 * ((x - mu) ** 2)

    # ID8 found different values A_1 and A_2 - normalizing factor is the same for both distributions here (as indicated
    # in split normal distribution wiki - even though this is not a traditional split normal, but a mixture normal)
    A = np.sqrt(2/np.pi) * 1/(sigma_1 + sigma_2)

    # Defined cutoff between distributions as within 10 degree of galactic plane (lat = 0 degrees)
    mask = np.abs(x - mu) < 10

    return np.where(mask, A * np.exp(upper / (2 * (sigma_1 ** 2))), A * np.exp(upper / 2 * (sigma_2 ** 2)))


# REFERENCES

# Trapezoidal Rule - https://en.wikipedia.org/wiki/Trapezoidal_rule
