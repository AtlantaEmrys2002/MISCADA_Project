"""
This code is used to create extra plots for the report (not all plots in the report can be generated with this file
- see any file named visualisation.py in data_simulation and evaluation for other plots.
"""

from astropy.io import fits
import copy
import healpy as hp
import matplotlib.colorizer as mcolorizer
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def plot_fermi_lat_count_map(binned_count_map_fermi: str="/Volumes/T7/project_data/real_data/fermi_filtered_gti_binned_for_evaluation.fits"):

    plt.cla()
    plt.clf()
    plt.close()

    # NOTE THAT THIS IS JUST TAKING A LOW ENERGY bin AND NO TITLE INCLUDED

    Path("./plots").mkdir(parents=True, exist_ok=True)

    count_map = np.sum(np.array([hp.read_map(binned_count_map_fermi, hdu="SKYMAP", field=b) for b in range(5)]), axis=0)

    hp.newvisufunc.projview(
        count_map, coord=["G"], graticule=True, graticule_labels=True, xlabel="Galactic Longitude, $l$ [$\\degree$]",
        ylabel="Galactic Latitude, $b$ [$\\degree$]", unit="Photons", cb_orientation="vertical",
        projection_type="aitoff", norm='log',
    )

    plt.tight_layout()

    plt.savefig("./plots/fermi_count_map.png")

    plt.close()

    hp.newvisufunc.projview(
        count_map, coord=["G"], graticule=True, graticule_labels=True, xlabel="Galactic Longitude, $l$ [$\\degree$]",
        ylabel="Galactic Latitude, $b$ [$\\degree$]", unit="Photons", cb_orientation="vertical",
        projection_type="aitoff", norm='log', title="Aitoff Projection of Fermi-LAT Count Map",
    )

    plt.tight_layout()

    plt.savefig("./plots/fermi_count_map_titled.png")

    plt.close()


def plot_fermi_lat_exposure(binned_count_map_fermi: str="/Volumes/T7/project_data/real_data/fermi_filtered_gti_exposure_map.fits"):

    plt.cla()
    plt.clf()
    plt.close()

    # NOTE THAT THIS IS JUST TAKING A LOW ENERGY bin AND NO TITLE INCLUDED

    Path("./plots").mkdir(parents=True, exist_ok=True)

    count_map = np.sum(np.array([hp.read_map(binned_count_map_fermi, hdu="SKYMAP", field=b) for b in range(5)]), axis=0)

    hp.newvisufunc.projview(
        count_map, coord=["G"], graticule=True, graticule_labels=True, xlabel="Galactic Longitude, $l$ [$\\degree$]",
        ylabel="Galactic Latitude, $b$ [$\\degree$]", unit="$\\text{cm}^2\\text{s}$", cb_orientation="vertical",
        projection_type="aitoff", norm='log',
    )

    plt.tight_layout()

    plt.savefig("./plots/fermi_count_map.png")

    plt.close()

    hp.newvisufunc.projview(
        count_map, coord=["G"], graticule=True, graticule_labels=True, xlabel="Galactic Longitude, $l$ [$\\degree$]",
        ylabel="Galactic Latitude, $b$ [$\\degree$]", unit="Photons", cb_orientation="vertical",
        projection_type="aitoff", norm='log', title="Aitoff Projection of $Fermi$ Exposure Maps",
    )

    plt.tight_layout()

    plt.savefig("./plots/exposure_map.png")

    plt.close()



def plot_419(patch_directory="./../source_extractors/real_data/real_patches/patches/patch_419"):

    plt.cla()
    plt.clf()
    plt.close()

    plt.rcParams["figure.figsize"] = (4, 10)

    binned_patch = np.load(patch_directory + "/patch.npy")[:4]

    fig, axs = plt.subplots(4, 1, gridspec_kw={"height_ratios":[1, 1, 1, 0.05]})
    # fig.suptitle("Energy-Binned Patch Centred \n on $l = -146.25\\degree, b=-4.78\\degree$")

    # create a colorizer with a predefined norm to be shared across all images
    norm = mcolors.Normalize(vmin=np.min(binned_patch), vmax=np.max(binned_patch))
    colorizer = mcolorizer.Colorizer(norm=norm)

    images = []

    count = 0

    for ax, data in zip(axs.flat, binned_patch):

        if count == 3:
            fig.colorbar(images[0], orientation='horizontal', cax=ax, fraction=0.1, label="Photons")
            continue

        images.append(ax.imshow(data, colorizer=colorizer))
        ax.set_axis_off()

        count += 1

    fig.tight_layout()

    plt.savefig("./plots/binned_patch.png")

    plt.cla()
    plt.clf()
    plt.close()

    plt.rcParams["figure.figsize"] = (4, 5)

    mask = np.load(patch_directory + "/mask.npy")

    noise = np.random.choice(2, size=(64 * 64), p=[0.99, 0.01]).reshape((64, 64))

    plt.imshow(np.logical_or(mask, noise))

    plt.savefig("./plots/mask_with_noise.png")

    plt.cla()
    plt.clf()
    plt.close()

    plt.rcParams["figure.figsize"] = (4, 5)

    plt.imshow(mask)

    plt.savefig("./plots/mask.png")

    plt.cla()
    plt.clf()
    plt.close()

    plt.rcParams["figure.figsize"] = (8, 4)

    metadata = pd.read_csv(patch_directory + "/metadata.csv")

    regions = []

    for k in zip(metadata["cartesian_y"].to_numpy(), metadata["cartesian_x"].to_numpy()):

        coord = [int(k[0]) // 2, int(k[1]) // 2]

        regions.append(copy.deepcopy(binned_patch[:, coord[0] - 5 : coord[0] + 5, coord[1] - 5 : coord[1] + 5]))

        print(regions[0][0].shape)

    regions = np.array(regions)

    num_region = 1

    for r in regions:

        fig, axs = plt.subplots(1, 3,  gridspec_kw={"width_ratios":[1, 1, 1]})
        # fig.suptitle("Boxes Around Detected Sources")

        images = []

        count = 0

        # create a colorizer with a predefined norm to be shared across all images
        norm = mcolors.Normalize(vmin=np.min(binned_patch), vmax=np.max(binned_patch))
        colorizer = mcolorizer.Colorizer(norm=norm)

        for ax, data in zip(axs.flat, r):

            if count == 3:
                # fig.colorbar(images[0], orientation='vertical', cax=ax, label="Photons", fraction=0.046, pad=0.04)
                continue

            images.append(ax.imshow(data, colorizer=colorizer))
            ax.set_axis_off()

            count += 1

        # cax = fig.add_axes([axs.flat[3].get_position().x1 - 0.25, axs.flat[2].get_position().y0, 0.02,
        #                     axs.flat[2].get_position().y1 - axs.flat[2].get_position().y0])

        cax = axs[2].inset_axes((1.05, 0, 0.08, 1.0))

        fig.colorbar(images[0], orientation='vertical', cax=cax, label="Photons")

        fig.tight_layout()

        plt.savefig("./plots/box_{}.png".format(num_region))

        plt.cla()
        plt.clf()
        plt.close()

        num_region += 1









if __name__ == "__main__":

    # plot_fermi_lat_count_map()
    #
    # plot_419()

    plot_fermi_lat_exposure()


# REFERENCES

# Colour Bar - https://stackoverflow.com/questions/13310594/positioning-the-colorbar
# Colour Bar - https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
# Remove Axes Labels - https://stackoverflow.com/questions/9295026/how-to-remove-axis-legends-and-white-padding
