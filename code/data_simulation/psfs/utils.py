"""
Utility functions when fitting and applying point spread functions (PSFs).
"""

import numpy as np
import numpy.typing as npt

import warnings


def dual_function(x: npt.ArrayLike, sigma_core: np.float64, gamma_core: np.float64, sigma_tail: np.float64,
                  gamma_tail: np.float64, f_core: np.float64) -> npt.ArrayLike:
    """Dual King function - one more prominent at small separations and the other at large separations.

    Parameters
    ----------
    x
        Separation from photons' true origin.
    sigma_core : np.float64
        Coefficient in central King function.
    gamma_core : np.float64
        Coefficient in central King function.
    sigma_tail : np.float64
        Coefficient in tail King function.
    gamma_tail : np.float64
        Coefficient in trail King function.
    f_core : np.float64
        Normalising coefficient to reconcile dual functions.

    """
    first_distribution = king_function(x, sigma=sigma_core, gamma=gamma_core)

    second_distribution = king_function(x, sigma=sigma_tail, gamma=gamma_tail)

    return (f_core * first_distribution) + ((1 - f_core) * second_distribution)


def king_function(x: npt.ArrayLike, sigma: np.float64, gamma: np.float64) -> npt.ArrayLike:
    """King Function (or Moffat Distribution) is the standard function for "reconstructing" PSFs from observed data.
    Recommended in this paper: https://iopscience.iop.org/article/10.1088/0004-637X/765/1/54/pdf

    Parameters
    ----------
    x
        Separation from photons' true origin
    sigma : np.float64
        Coefficient in function
    gamma : np.float64
        Coefficient in function

    """
    factor = 1 / (2 * np.pi * (sigma ** 2))

    first_term = (1 - (1 / gamma))

    second_term = 1 + ((1 / (2 * gamma)) * (x ** 2 / sigma ** 2))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = factor * first_term * (second_term ** (- gamma))

        result = np.nan_to_num(result)

    return result


def monte_carlo_sampler(parameters: npt.NDArray[np.float64], num_samples: int) -> npt.NDArray[np.float64]:
    """Used to sample random values from PDF described by dual King Function. Took inspiration from this code
    (different method, but still useful) when implementing final Ratio of Uniforms code) -
    https://github.com/scipy/scipy/blob/v1.18.0/scipy/stats/_sampling.py#L1130-L1319

    Parameters
    ----------
    parameters : ndarray
        Parameters of dual King function.
    num_samples : int
        Number of random values to sample.

    Returns
    -------
    ndarray
        List of random values sampled from PDF.

    """
    # Used to sample random values directly from PDF

    # https://en.wikipedia.org/wiki/Ratio_of_uniforms

    # Find the upper bound of the interval from which we sample initial x - take initial maximum to be 30 degrees (as
    # that is our specified radius for diffuse sources - much greater than for this for our point sources)

    intervals = np.linspace(0, 30, num=10000)

    func_values = dual_function(intervals, sigma_core=np.float64(parameters[0]), gamma_core=np.float64(parameters[1]),
                                sigma_tail=np.float64(parameters[2]), gamma_tail=np.float64(parameters[3]),
                                f_core=np.float64(parameters[4]))

    # Bounding box
    y_min, y_max = 0, func_values[np.argmax(func_values)]

    # Uniformly sample this bounding box - if under the curve, include

    samples = np.zeros(num_samples)

    num_samples_generated = 0

    while num_samples_generated < num_samples:

        # "Throw dart" into bounding box
        # if not energy_bin_vals:
        # candidate_x = np.random.uniform(low=0, high=30)
        # else:
        #     if energy_bin == 0:
        #         candidate_x = np.random.uniform(low=0, high=30 / np.sqrt(((3.5) ** 2) + (0.15 ** 2)))
        #     else:
        #         candidate_x = np.random.uniform(low=0, high=30 / np.sqrt(((3.5 * ((energy_bin / 100) ** (-0.8))) ** 2) +
        #                                                              (0.15 ** 2)))

            # if not energy_bin_vals:
            #     samples.append(candidate_x)
            # else:
            #     samples.append(candidate_x * np.sqrt(((3.5 * ((energy_bin / 100) ** (-0.8))) ** 2) + (0.15 ** 2)))

            # samples.append(candidate_x)

            # samples.append(np.sin(np.deg2rad(candidate_x / 2)) * 2)

        candidate_xs = np.random.uniform(low=0, high=30, size=num_samples - num_samples_generated)
        candidate_ys = np.random.uniform(low=0, high=y_max, size=num_samples - num_samples_generated)

        mask = (candidate_ys < dual_function(candidate_xs, sigma_core=np.float64(parameters[0]),
                                             gamma_core=np.float64(parameters[1]), sigma_tail=np.float64(parameters[2]),
                                             gamma_tail=np.float64(parameters[3]), f_core=np.float64(parameters[4])))

        num_new_vals = np.sum(mask)

        samples[num_samples_generated: num_samples_generated + num_new_vals] = candidate_xs[mask]

        num_samples_generated += num_new_vals

    return samples.astype(np.float64)

    # return np.array(samples).astype(np.float64)


def normalise_psf(thetas: npt.NDArray[np.float64], psf_values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Normalises a function over solid angle. Recommendations taken from
    https://math.stackexchange.com/questions/4806473/forcing-a-function-to-integrate-to-1 and from
    https://gamma-astro-data-formats.readthedocs.io/en/v0.1/irfs/psf/index.html#psf-pdf

    N.B. apply after energy scaling.

    Parameters
    ----------
    thetas
        Range over which to normalise.
    psf_values
        Separation values to normalise.

    Returns
    -------
    ndarray
        Normalised function values

    """
    #
    # probs = ((2 * np.pi * thetas) ** 2) * psf_values

    probs = (2 * np.pi * thetas) * psf_values

    # Integrate over probs
    approx_integral = np.sum(np.array(
        [((probs[k + 1] + probs[k]) / 2) * (thetas[k + 1] - thetas[k]) for k in range(len(psf_values) - 1)]))

    # Normalise such that the integral is 1
    probs /= approx_integral

    return probs


def scale_psf(psf_values: npt.NDArray[np.float64], energy_bin: npt.NDArray[np.float64], c_0: float = 3.5,
              c_1: float = 0.15, beta: float = 0.8):
    """Scales out the energy dependence of a PSF found using gtpsf fermitools function (constants originate from here
    - https://iopscience.iop.org/article/10.1088/0004-637X/765/1/54/pdf).

    Parameters
    ----------
    psf_values : ndarray
        Values of PSF function ordered according to radial separation from spatial origin of point source.
    energy_bin : ndarray
        Energy values for the photon energy bin boundaries
    c_0 : float
        Constant in scaling function.
    c_1 : float
        Constant in scaling function.
    beta : float
        Constant in scaling function.

    """

    # Calculate energy scale factor
    scale_factor = np.sqrt(((c_0 * ((energy_bin / 100) ** (-beta))) ** 2) + (c_1 ** 2))

    # Scale PSF values
    psf_values /= scale_factor

    return psf_values

# REFERENCES

# Moffat Distribution - https://en.wikipedia.org/wiki/Moffat_distribution
# Optimised Sampling - https://github.com/scipy/scipy/blob/v1.18.0/scipy/stats/_sampling.py#L1130-L1319
# Ratio of Uniforms - https://en.wikipedia.org/wiki/Ratio_of_uniforms
