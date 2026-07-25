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
def RotMatrixY(psi, isdeg=True):
    if isdeg:
        return np.array([[np.cos(np.radians(psi)), 0.0, -np.sin(np.radians(psi))], [0.0, 1.0, 0.0],
                         [np.sin(np.radians(psi)), 0.0, np.cos(np.radians(psi))]])
    else:
        return np.array([[np.cos(psi), 0.0, -np.sin(psi)], [0.0, 1.0, 0.0], [np.sin(psi), 0.0, np.cos(psi)]])


def RotMatrixZ(psi, isdeg=True):
    if isdeg:
        return np.array([[np.cos(np.radians(psi)), np.sin(np.radians(psi)), 0.0],
                         [-np.sin(np.radians(psi)), np.cos(np.radians(psi)), 0.0], [0.0, 0.0, 1.0]])
    else:
        return np.array([[np.cos(psi), np.sin(psi), 0.0], [-np.sin(psi), np.cos(psi), 0.0], [0.0, 0.0, 1.0]])


def sph2xyz(r, theta, phi, isdeg=True):
    if isdeg:
        return np.array([r * np.sin(np.radians(theta)) * np.cos(np.radians(phi)),
                         r * np.sin(np.radians(theta)) * np.sin(np.radians(phi)), r * np.cos(np.radians(theta))])
    else:
        return np.array([r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)])


def xyz2sph(x, y, z, isdeg=True, is_lat=False):
    r = np.sqrt(x * x + y * y + z * z)
    if isdeg:
        phi = np.degrees(np.arctan2(y, x))
        lat = np.degrees(np.arctan2(z, np.sqrt(x * x + y * y)))
        if is_lat:
            return np.array([r, lat, phi])
        else:
            return np.array([r, 90. - lat, phi])
    else:
        phi = np.arctan2(y, x)
        lat = np.arctan2(z, np.sqrt(x * x + y * y))
        if is_lat:
            return np.array([r, lat, phi])
        else:
            return np.array([r, np.pi / 2.0 - lat, phi])


def get_lb_from_pixel(pixel_id, lb_centre, xsize=128, isdeg=True, is_lat=True):
    ##if input angles are in degree use 'isdeg = True'
    ######### Generate (l,b) coordinate map of 10x10deg patch ######

    # DON'T FORGET TO PASS lb_centre with ranges l = [-180, 180] and b = [-90, 90] as assumed HEALPIX format!!!!


    coord_range = np.linspace(-4.9609375, 4.9609375, xsize)

    X, Y = np.meshgrid(coord_range, coord_range)
    lonlat_patch = list(zip(np.flip(X.flatten()), Y.flatten()))

    ######### Get rotation matrix used to rotate the original centre to (0., 0.) #########
    l_centre, b_centre = lb_centre
    r = np.dot(RotMatrixY(-b_centre), RotMatrixZ(l_centre))
    #########

    lon_PS_rotated, lat_PS_rotated = lonlat_patch[pixel_id]

    xyz_PS_rotated = sph2xyz(1., 90. - lat_PS_rotated, lon_PS_rotated)
    x_PS, y_PS, z_PS = np.array(np.dot(r.T, xyz_PS_rotated), dtype='float32')
    r, b_PS, l_PS = xyz2sph(x_PS, y_PS, z_PS, isdeg=isdeg, is_lat=is_lat)

    # N.B. FROM MHR - Fixed this bit - stored my coordinates for l_PS as -180 to 180 instead of 0 to 360 (cause of
    # HEALPIX SYSTEM)
    l_PS += 180  # to get to 0 - 360

    if l_PS > 360 or l_PS < 0:

        l_PS = l_PS % 360

    # if l_PS < 0:
    #
    #     # l_PS = 360 + l_PS

    return l_PS, b_PS


def pixel_id(row, col, xsize_patch):
    return int(xsize_patch * row + col)
