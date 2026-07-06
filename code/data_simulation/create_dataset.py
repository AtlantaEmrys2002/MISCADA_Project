from astropy.io import fits
import healpy as hp
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


def plot_all_sky_map(healpix_maps, energy_bins, title: str, directory: str):

    # Create directory to store results
    Path(directory).mkdir(parents=True, exist_ok=True)

    # Format energy bins
    energy_bin_labels = format_scientific_notation_label(energy_bins)

    for b in range(len(healpix_maps)):

        projview(
            healpix_maps[b],
            coord=["G"],
            graticule=True,
            graticule_labels=True,
            xlabel="Galactic Longitude [$\degree$]",
            ylabel="Galactic Latitude [$\degree$]",
            cb_orientation="vertical",
            projection_type="aitoff",
            title="{} Map for {} - {} MeV".format(title, energy_bin_labels[b], energy_bin_labels[b + 1]),
            cbar=False,
            sub=(3, 2, b + 1),
        )

    plt.tight_layout()

    plt.savefig(directory + title.lower() + "_map.png")

    plt.close()


def create_exposure_map(file_name: str):

    # Read and plot binned exposure files

    exposure_file = "/Volumes/T7/project_data/real_data/fermi_filtered_gti_exposure_map.fits"

    with fits.open(exposure_file) as hdul:

        num_bins = hdul[1].header["TFIELDS"]

        energy_bins = np.array([k[0] for k in hdul[2].data])

        exposure_maps = [hp.read_map(file_name, hdu="HPXEXPOSURES", field=b) for b in range(num_bins - 1)]

    return exposure_maps, energy_bins


exposure_maps, energy_bins = create_exposure_map(file_name="/Volumes/T7/project_data/real_data/fermi_filtered_gti_exposure_map.fits")

plot_all_sky_map(healpix_maps=exposure_maps, energy_bins=energy_bins, title="Exposure", directory="./plots/all_sky_maps/")

