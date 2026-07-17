# THE CODE IN THIS FILE WAS ADAPTED FROM ID8's - SEE REFERENCE INFORMATION BELOW
# I REWROTE THE MASKING FUNCTION TO BE OPTIMAL IN TERMS OF COMPUTATION TIME AND MADE USE OF SCIPY

# Code can be found here: https://github.com/bapanes/AutoSourceID/blob/main/codes/from-cats-to-locnet-input.py

# author: Boris Panes, February 4, 2021
# this code contains several contributions from Christopher Eckner, Gulli and Roberto
# specially concerning patch generation and photon flux

import numpy as np
from scipy.spatial.distance import cdist


def psf_bck_mask(y0, x0, radius, psf_mask):
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
