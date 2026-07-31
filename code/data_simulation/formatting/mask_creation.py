"""
Functions used when creating masks of patches (indicating where sources are in a patch). This performs the same
functionality as that in https://github.com/bapanes/AutoSourceID/blob/main/codes/from-cats-to-locnet-input.py. However,
I rewrote the code from scratch to improve efficiency and readability, making use of several new libraries. The original
code had the following author note:

# author: Boris Panes, February 4, 2021
# this code contains several contributions from Christopher Eckner, Gulli and Roberto
# specially concerning patch generation and photon flux

"""

import numpy as np
from scipy.spatial.distance import cdist


def psf_bck_mask(y0, x0, radius, psf_mask):
    """Calculates new mask of a patch given previous mask and the centre of a new source to be masked within the patch.

    Parameters
    ----------
    y0
        y index into patch indicating the centre of the new source - remembering that images are indexed y, x rather x,
        y in this context.
    x0
        x index into patch indicating the centre of the new source - remembering that images are indexed y, x rather x,
        y in this context.
    radius
        Assumed radius of circle introduced in mask with centre on y0, x0 around source.
    psf_mask
        Previous mask of patch to be added to and returned.

    """
    nrow = psf_mask.shape[0]
    ncol = psf_mask.shape[1]

    grid2D_psf = psf_mask.copy()

    # Calculate all possible indices into image
    coords = np.array([(y, x) for y in range(nrow) for x in range(ncol)])

    centre = np.array([[y0, x0]])

    # Calculate distance between centre of source and all positions within image
    distances = cdist(coords, centre).flatten()

    # Find where less than radius
    mask = coords[distances < radius]

    # Mask array
    grid2D_psf[mask[:, 0], mask[:, 1]] = 1.0

    return grid2D_psf

# REFERENCES

# Modifying Array Locations - https://stackoverflow.com/questions/7761393/how-to-modify-a-2d-numpy-array-at-specific-loc
# ations-without-a-loop
# Selecting Certain Indices - https://stackoverflow.com/questions/30917753/subsetting-a-2d-numpy-array
