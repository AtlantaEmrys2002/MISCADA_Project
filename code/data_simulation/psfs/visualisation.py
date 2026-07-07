from astropy.io import fits
from . fit_psf import normalise_psf, scale_psf
import matplotlib.pyplot as plt
import numpy as np
from .utils import dual_function, monte_carlo_sampler


def format_scientific_notation_label(numbers):
    # Formats numbers into scientific notation for inclusion on graphs

    labels = []

    for number in numbers:

        scientific = str(np.format_float_scientific(number, precision=2, trim='0'))

        base, exponent = scientific.split('e')

        if exponent[0] == "+":
            sign = ""
        else:
            sign = "-"

        label = base + "$\\times 10^{" + sign + str(int(exponent[1:])) + "}$"

        labels.append(label)

    return labels


def plot_fitted_point_source_psf(psf_file: str, function_parameters, directory: str):

    empirical = []
    energy_bins = []

    # Plot empirical data from gtpsf
    with fits.open(psf_file) as hdul:

        thetas = np.array([k[0] for k in hdul["THETA"].data])

        psf_data = hdul["PSF"].data

        num_bins = len(psf_data)

        for b in range(num_bins):

            # Lowest energy (MeV) of this bin
            energy_value = psf_data[b][0]

            # PSF values dP/dOmega - probability to find event in solid angle dOmega at offset r from point source
            psf_values = np.array(psf_data[b][2])

            psf_values = scale_psf(psf_values, energy_value)

            probs = normalise_psf(thetas, psf_values)

            empirical.append(probs)
            energy_bins.append(energy_value)

    # Create figure
    plt.rcParams["figure.figsize"] = (10, 14)

    fig, ax = plt.subplots((num_bins // 2) + (num_bins % 2), 2)

    # Plotting
    flattened_axes = ax.flatten()

    for x in range(num_bins):

        flattened_axes[x].scatter(thetas, empirical[x], marker='+', label="Empirical", color="blue")

        popt = function_parameters[x]

        flattened_axes[x].plot(thetas, dual_function(thetas, sigma_core=popt[0], gamma_core=popt[1], sigma_tail=popt[2],
                                                     gamma_tail=popt[3], f_core=popt[4]), label="Fitted King Function",
                               color="orange", linestyle='--')

        # Sample random values
        random_values = monte_carlo_sampler(dual_function, popt, 5000)

        # Plot sampled values
        counts, bins = np.histogram(random_values, bins=250, density=True)
        flattened_axes[x].stairs(counts, bins, color="green", label="Random Samples")


    # Formatting

    if num_bins % 2 == 1:
        fig.delaxes(ax[-1, -1])

    labels = format_scientific_notation_label(energy_bins)

    for x in range(len(energy_bins)):

        a = flattened_axes[x]

        a.set_xlim(0, 5)
        a.set_ylim(0, )

        a.set_xlabel("Energy Scaled Angular Deviation of $\gamma$-Ray, $x$ [$\degree$]")
        a.set_ylabel("PSF($x, E$)")

        if x == len(energy_bins) - 1:
            a.set_title("PDF of Angular Deviation for $\gamma$-Rays \nwith Energy {}+ MeV".format(labels[x]))
        else:
            a.set_title("PDF of Angular Deviation for $\gamma$-Rays \nwith Energy {}-{} MeV".format(labels[x],
                                                                                                  labels[x+1]))

        a.legend()

    fig.suptitle("Point Spread Function Fitting")

    fig.tight_layout()

    fig.savefig(directory + "/fitted_point_source_psf.png")

    plt.close()
