from healpy.newvisufunc import projview
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


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


def plot_all_sky_map(healpix_maps, energy_bins, title: str, directory: str, logarithmic=False):

    # Create directory to store results
    Path(directory).mkdir(parents=True, exist_ok=True)

    # Format energy bins
    energy_bin_labels = format_scientific_notation_label(energy_bins)

    # Plot either raw or log of data
    if logarithmic is True:
        data = np.log(healpix_maps)
    else:
        data = healpix_maps

    # Formatting
    if len(title) > 10:
        new_line = "\n"
    else:
        new_line = ""

    for b in range(len(healpix_maps)):

        label = "{} Map for {}{} - {} MeV".format(title, new_line, energy_bin_labels[b], energy_bin_labels[b + 1])

        projview(
            data[b],
            coord=["G"],
            graticule=True,
            graticule_labels=True,
            xlabel="Galactic Longitude [$\degree$]",
            ylabel="Galactic Latitude [$\degree$]",
            cb_orientation="vertical",
            projection_type="aitoff",
            title=label,
            cbar=False,
            sub=(3, 2, b + 1),
        )

    plt.tight_layout()

    plt.savefig(directory + title.lower().replace(" ", "_") + "_map.png")

    plt.close()
