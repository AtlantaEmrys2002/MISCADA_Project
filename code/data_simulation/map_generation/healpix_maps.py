from astropy.io import fits
import healpy as hp
import numpy as np


def create_expected_counts_map(infinite_counts_map):

    binned_count_maps = []

    num_bins = len(infinite_counts_map)

    for b in range(num_bins):

        # Poisson sample each pixel in the infinite statistics count map to get a count map c
        sampled_counts = np.random.poisson(lam=infinite_counts_map[b])

        binned_count_maps.append(sampled_counts)

    return binned_count_maps


def create_exposure_map(exposure_file: str):

    # Read and plot binned exposure files

    with fits.open(exposure_file) as hdul:

        num_bins = hdul[1].header["TFIELDS"] - 1

        energy_bins = np.array([k[0] for k in hdul[2].data])

        exposure_maps = [hp.read_map(exposure_file, hdu="HPXEXPOSURES", field=b) for b in range(num_bins)]

    return exposure_maps, energy_bins


def create_infinite_statistics_map(exposure_maps, fluxes, pixels):

    # Pixel x corresponds to location of source x in the sky with flux x

    binned_infinite_statistics = []

    num_bins = len(exposure_maps)

    for b in range(num_bins):

        healpix_exposure_map = exposure_maps[b]

        # Copy for infinite statistics
        infinite_statistics_counts = np.zeros_like(healpix_exposure_map)

        # Calculate infinite statistics realisation of sky (c bar)
        for x in range(len(pixels)):
            pixel = pixels[x]

            # Important to add - Poisson value is additive
            infinite_statistics_counts[pixel] += healpix_exposure_map[pixel] * fluxes[x, b]

        # Add infinite statistics bin to list
        binned_infinite_statistics.append(infinite_statistics_counts)

    return binned_infinite_statistics
