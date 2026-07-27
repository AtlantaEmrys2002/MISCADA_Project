


# THE CODE IN THIS FILE WAS TAKEN (AND ADAPTED FROM ID8 PAPER'S CODE) - SEE BELOW FOR REFERENCING INFORMATION.

# I HAVE OPTIMISED/MODIFIED SEVERAL FUNCTIONS

# The link to this code can be found here: https://github.com/bapanes/AutoSourceID/blob/main/codes/from-cats-to-locnet-
# input.py

# author: Boris Panes, February 4, 2021
# this code contains several contributions from Christopher Eckner, Gulli and Roberto
# specially concerning patch generation and photon flux

# This code helps project the all-sky maps into Cartesian coordinates and slice them into 64 x 64 images.

import numpy as np


# geometric utilities
def RotMatrixY(psi):

    return np.array([[np.cos(np.radians(psi)), 0.0, -np.sin(np.radians(psi))], [0.0, 1.0, 0.0],
                     [np.sin(np.radians(psi)), 0.0, np.cos(np.radians(psi))]])


def RotMatrixZ(psi):

    return np.array([[np.cos(np.radians(psi)), np.sin(np.radians(psi)), 0.0],
                     [-np.sin(np.radians(psi)), np.cos(np.radians(psi)), 0.0], [0.0, 0.0, 1.0]])


def sph2xyz(r, theta, phi):

    return np.array([r * np.sin(np.radians(theta)) * np.cos(np.radians(phi)),
                     r * np.sin(np.radians(theta)) * np.sin(np.radians(phi)), r * np.cos(np.radians(theta))])


def xyz2sph(x, y, z, is_lat=False):
    r = np.sqrt(x * x + y * y + z * z)

    phi = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arctan2(z, np.sqrt(x * x + y * y)))
    if is_lat:
        return np.array([r, lat, phi])
    else:
        return np.array([r, 90. - lat, phi])


def get_lb_from_pixel(pixel_id, lb_centre, xsize=128, is_lat=True):

    # Ensures function works with arrays and scalars
    if type(pixel_id) is int or type(pixel_id) is np.int64:

        pixel_id = np.array([pixel_id])

    ##if input angles are in degree use 'isdeg = True'
    ######### Generate (l,b) coordinate map of 10x10deg patch ######

    # DON'T FORGET TO PASS lb_centre with ranges l = [-180, 180] and b = [-90, 90] as assumed HEALPIX format!!!!

    coord_range = np.linspace(-4.9609375, 4.9609375, xsize)

    X, Y = np.meshgrid(coord_range, coord_range)
    lonlat_patch = np.array(list(zip(np.flip(X.flatten()), Y.flatten())))

    ######### Get rotation matrix used to rotate the original centre to (0., 0.) #########
    l_centre, b_centre = lb_centre
    r = np.dot(RotMatrixY(-b_centre), RotMatrixZ(l_centre))
    #########

    rotated_coordinates = lonlat_patch[pixel_id]

    lon_PS_rotated, lat_PS_rotated = rotated_coordinates[:, 0], rotated_coordinates[:, 1]

    xyz_PS_rotated = sph2xyz(1., 90. - lat_PS_rotated, lon_PS_rotated)
    x_PS, y_PS, z_PS = np.array(np.dot(r.T, xyz_PS_rotated), dtype='float32')
    r, b_PS, l_PS = xyz2sph(x_PS, y_PS, z_PS, is_lat=is_lat)

    # N.B. FROM MHR - Fixed this bit - stored my coordinates for l_PS as -180 to 180 instead of 0 to 360 (cause of
    # HEALPIX SYSTEM)
    l_PS += 180  # to get to 0 - 360

    out_of_range = np.logical_or(l_PS > 360, l_PS < 0)

    l_PS = np.where(out_of_range, l_PS % 360, l_PS)

    # When returning individual values
    if type(pixel_id) is int or type(pixel_id) is np.int64:

        l_PS = l_PS[0]
        b_PS = b_PS[0]

    return l_PS, b_PS


def pixel_id(row, col, xsize_patch):
    return (xsize_patch * row + col).astype(int)
