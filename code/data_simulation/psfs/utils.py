import numpy as np


def dual_function(x, sigma_core, gamma_core, sigma_tail, gamma_tail, f_core):

    first_distribution = king_function(x, sigma=sigma_core, gamma=gamma_core)

    second_distribution = king_function(x, sigma=sigma_tail, gamma=gamma_tail)

    return (f_core * first_distribution) + ((1 - f_core) * second_distribution)


def king_function(x, sigma, gamma):

    # Called King function by this paper - https://iopscience.iop.org/article/10.1088/0004-637X/765/1/54/pdf
    # However, it is often referred to as the Moffat distribution - note in final paper

    factor = 1 / (2 * np.pi * (sigma ** 2))

    first_term = (1 - (1 / gamma))

    second_term = 1 + ((1 / (2 * gamma)) * (x ** 2 / sigma ** 2))

    return factor * first_term * (second_term ** (- gamma))


def monte_carlo_sampler(func, parameters, num_samples):

    # Used to sample random values directly from PDF

    # https://en.wikipedia.org/wiki/Ratio_of_uniforms

    # Find the upper bound of the interval from which we sample initial x - take initial maximum to be 30 degrees (as
    # that is our specified radius for diffuse sources - much greater than for this for our point sources)

    intervals = np.linspace(0, 30, num=100000)

    func_values = func(intervals, sigma_core=parameters[0], gamma_core=parameters[1], sigma_tail=parameters[2],
                       gamma_tail=parameters[3], f_core=parameters[4])

    # Bounding box
    y_min, y_max = 0, func_values[np.argmax(func_values)]
    x_min, x_max = 0, 30

    # Uniformly sample this bounding box - if under the curve, include

    samples = []

    while len(samples) < num_samples:

        # "Throw dart"
        candidate_x = np.random.uniform(low=x_min, high=x_max)
        candidate_y = np.random.uniform(low=y_min, high=y_max)

        # If it is under the curve
        if candidate_y <= func(candidate_x, sigma_core=parameters[0], gamma_core=parameters[1],
                               sigma_tail=parameters[2], gamma_tail=parameters[3], f_core=parameters[4]):
            samples.append(candidate_x)

    return samples


# REFERENCES

# Moffat Distribution - https://en.wikipedia.org/wiki/Moffat_distribution
# Ratio of Uniforms - https://en.wikipedia.org/wiki/Ratio_of_uniforms
