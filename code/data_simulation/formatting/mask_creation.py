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
from skimage.draw import disk


# def psf_bck_mask(y0, x0, radius, psf_mask):
#     """Calculates new mask of a patch given previous mask and the centre of a new source to be masked within the patch.
#
#     Parameters
#     ----------
#     y0
#         y index into patch indicating the centre of the new source - remembering that images are indexed y, x rather x,
#         y in this context.
#     x0
#         x index into patch indicating the centre of the new source - remembering that images are indexed y, x rather x,
#         y in this context.
#     radius
#         Assumed radius of circle introduced in mask with centre on y0, x0 around source.
#     psf_mask
#         Previous mask of patch to be added to and returned.
#
#     """
#     nrow = psf_mask.shape[0]
#     ncol = psf_mask.shape[1]
#
#     grid2D_psf = psf_mask.copy()
#
#     # Calculate all possible indices into image
#     coords = np.array([(y, x) for y in range(nrow) for x in range(ncol)])
#
#     centre = np.array([[y0, x0]])
#
#     # Calculate distance between centre of source and all positions within image
#     distances = cdist(coords, centre).flatten()
#
#     # Find where less than radius
#     mask = coords[distances < radius]
#
#     # Mask array
#     grid2D_psf[mask[:, 0], mask[:, 1]] = 1.0
#
#     return grid2D_psf


def create_mask(agn_pos_list, psr_pos_list, xsize_location: int, radius: float | int = 2.5):
    max_pixel = xsize_location - 1

    # Find the minimum of the position in the image and 127 (the maximum index of the image in the y or
    # x-axis)
    psr_ys = np.array([]) if psr_pos_list.shape[0] == 0 else np.where(psr_pos_list[:, 0] < max_pixel,
                                                                      psr_pos_list[:, 0], max_pixel)
    psr_xs = np.array([]) if psr_pos_list.shape[0] == 0 else np.where(psr_pos_list[:, 1] < max_pixel,
                                                                      psr_pos_list[:, 1], max_pixel)

    agn_ys = np.array([]) if agn_pos_list.shape[0] == 0 else np.where(agn_pos_list[:, 0] < max_pixel,
                                                                      agn_pos_list[:, 0], max_pixel)
    agn_xs = np.array([]) if agn_pos_list.shape[0] == 0 else np.where(agn_pos_list[:, 1] < max_pixel,
                                                                      agn_pos_list[:, 1], max_pixel)

    ys = np.concatenate((agn_ys, psr_ys)) // 2
    xs = np.concatenate((agn_xs, psr_xs)) // 2

    # Remove any sources that are too close to the edge of the image
    ys_in_range = np.logical_and(ys >= radius, ys <= 63 - radius)
    xs_in_range = np.logical_and(xs >= radius, xs <= 63 - radius)

    in_range = np.logical_and(ys_in_range, xs_in_range)

    centres = np.stack((ys[in_range], xs[in_range]), axis=1)

    # Create mask for each source

    disks = [disk(centre, radius=radius, shape=(64, 64)) for centre in centres]

    # Create mask

    # Create blank mask which can be added to
    mask = np.zeros((64, 64))

    for d in disks:
        mask[d[0], d[1]] = 1

    return mask, ys, xs

# REFERENCES

# Array of Tuples - https://stackoverflow.com/questions/26634579/convert-array-of-lists-to-array-of-tuples-triple
# Concatenate 1D Arrays - https://stackoverflow.com/questions/9236926/concatenating-two-one-dimensional-numpy-arrays
# Modifying Array Locations - https://stackoverflow.com/questions/7761393/how-to-modify-a-2d-numpy-array-at-specific-loc
# ations-without-a-loop
# Selecting Certain Indices - https://stackoverflow.com/questions/30917753/subsetting-a-2d-numpy-array
# Skimage Recommend - https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
