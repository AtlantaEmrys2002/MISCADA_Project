from astropy.io import fits
import healpy as hp
import numpy as np


def create_exposure_map(exposure_file: str):

    # Read and plot binned exposure files

    with fits.open(exposure_file) as hdul:

        num_bins = hdul[1].header["TFIELDS"] - 1

        energy_bins = np.array([k[0] for k in hdul[2].data])

        exposure_maps = [hp.read_map(exposure_file, hdu="HPXEXPOSURES", field=b) for b in range(num_bins)]

    return exposure_maps, energy_bins
