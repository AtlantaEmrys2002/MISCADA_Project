# THE CODE IN THIS FILE WAS NOT WRITTEN BY ME - SEE BELOW FOR REFERENCING INFORMATION

# The link to this code can be found here: https://github.com/bapanes/AutoSourceID/blob/main/codes/from-cats-to-locnet-
# input.py

# author: Boris Panes, February 4, 2021
# this code contains several contributions from Christopher Eckner, Gulli and Roberto
# specially concerning patch generation and photon flux

# This code helps project the all-sky maps into Cartesian coordinates and slice them into 64 x 64 images.

from bisect import bisect
import numpy as np


def RotMatrixY(psi, isdeg=True):
    if isdeg:
        return np.array([[np.cos(np.radians(psi)), 0.0, -np.sin(np.radians(psi))], [0.0, 1.0, 0.0],
                         [np.sin(np.radians(psi)), 0.0, np.cos(np.radians(psi))]])
    else:
        return np.array([[np.cos(psi), 0.0, -np.sin(psi)], [0.0, 1.0, 0.0], [np.sin(psi), 0.0, np.cos(psi)]])


def RotMatrixZ(psi, isdeg=True):
    if isdeg:
        return np.array([[np.cos(np.radians(psi)), np.sin(np.radians(psi)), 0.0], [-np.sin(np.radians(psi)),
                                                                                   np.cos(np.radians(psi)), 0.0],
                         [0.0, 0.0, 1.0]])
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


# AGAIN BELOW USED TO GET L, B IN PATCH COORDINATES
# Function to implement inverse rotation
# to add in the predictions?
def get_lb_ps_centered(lb_ps, lb_centre, isdeg=True, is_lat=True):  ##if input angles are in degree use 'isdeg = True'

    l_centre, b_centre = lb_centre
    r = np.dot(RotMatrixY(-b_centre), RotMatrixZ(l_centre))

    lon_PS_rotated, lat_PS_rotated = lb_ps

    xyz_PS_rotated = sph2xyz(1., 90. - lat_PS_rotated, lon_PS_rotated)
    x_PS, y_PS, z_PS = np.array(np.dot(r, xyz_PS_rotated), dtype='float32')
    r, b_PS, l_PS = xyz2sph(x_PS, y_PS, z_PS, isdeg=isdeg, is_lat=is_lat)
    return l_PS, b_PS


def get_lb_from_pixel(pixel_id, lb_centre, xsize=128, isdeg=True,
                      is_lat=True):  ##if input angles are in degree use 'isdeg = True'
    ######### Generate (l,b) coordinate map of 10x10deg patch ######

    # Following the suggestions of CA mail
    if (xsize == 100):
        coord_range = np.linspace(-4.95, 4.95, xsize)

    if (xsize == 128):
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
    return l_PS, b_PS


def id_pixel(row, col, xsize_patch):
    return xsize_patch * row + col


def get_xml_lb_list_in_std_patch_coord(xsize_patch, patch_centre, list_of_pos_xml):

    # AGAIN USED TO GET COORDINATES OF SOURCE WITH L, B COORD INTO PATCH COORD SYS

    # corners around the patch_centre_position

    l_a, b_a = get_lb_from_pixel(id_pixel(0, 0, xsize_patch), patch_centre, xsize=xsize_patch)
    l_b, b_b = get_lb_from_pixel(id_pixel(xsize_patch - 1, 0, xsize_patch), patch_centre, xsize=xsize_patch)

    l_c, b_c = get_lb_from_pixel(id_pixel(0, xsize_patch - 1, xsize_patch), patch_centre, xsize=xsize_patch)
    l_d, b_d = get_lb_from_pixel(id_pixel(xsize_patch - 1, xsize_patch - 1, xsize_patch), patch_centre,
                                 xsize=xsize_patch)

    # corners around the center position

    l_c_a, b_c_a = get_lb_ps_centered((l_a, b_a), patch_centre)
    l_c_b, b_c_b = get_lb_ps_centered((l_b, b_b), patch_centre)
    l_c_c, b_c_c = get_lb_ps_centered((l_c, b_c), patch_centre)
    l_c_d, b_c_d = get_lb_ps_centered((l_d, b_d), patch_centre)

    l_arr = np.array([l_c_a, l_c_b, l_c_c, l_c_d])
    b_arr = np.array([b_c_a, b_c_b, b_c_c, b_c_d])

    l_c_min = np.amin(l_arr)
    l_c_max = np.amax(l_arr)
    b_c_min = np.amin(b_arr)
    b_c_max = np.amax(b_arr)

    list_of_lb_in_std_patch_coord = []

    for pos_con in range(len(list_of_pos_xml)):

        # positions in global lon lat coordinates
        l_pos = list_of_pos_xml[pos_con][0]
        b_pos = list_of_pos_xml[pos_con][1]

        # positions in standard patch coordinates
        l_c, b_c = get_lb_ps_centered((l_pos, b_pos), patch_centre)

        if ((l_c >= l_c_min) & (l_c <= l_c_max) & (b_c >= b_c_min) & (b_c <= b_c_max)):

            list_of_lb_in_std_patch_coord.append((l_c, b_c, l_pos, b_pos))

    return list_of_lb_in_std_patch_coord


def get_pixel_rc_list_from_xml_lb_list(xsize, lb_centre, list_of_pos_xml):

    # FUNCTION CONVERTS LON, LAT POSITION OF SOURCE TO POSITION IN IMAGE (IF THERE)

    lb_std_list = get_xml_lb_list_in_std_patch_coord(xsize, lb_centre, list_of_pos_xml)

    coord_range_x = np.linspace(-4.9609375, 4.9609375, xsize)
    coord_range_y = np.linspace(-4.9609375, 4.9609375, xsize)

    list_of_pixel_row = []
    list_of_pixel_col = []

    list_of_pixel_ltrue = []
    list_of_pixel_btrue = []
    # l = longitude
    # b = latitude

    for i in range(len(lb_std_list)):
        l, b, ltrue, btrue = lb_std_list[i][0], lb_std_list[i][1], lb_std_list[i][2], lb_std_list[i][3]

        pixel_l = xsize - bisect(list(coord_range_x), l)
        pixel_b = bisect(list(coord_range_y), b)

        list_of_pixel_row.append(pixel_b)
        list_of_pixel_col.append(pixel_l)

        list_of_pixel_ltrue.append(ltrue)
        list_of_pixel_btrue.append(btrue)

    return (list_of_pixel_row, list_of_pixel_col, list_of_pixel_ltrue, list_of_pixel_btrue)


def get_ps_info_128(patch_centre, list_agn_xml, list_psr_xml, xsize_patch):

    # THIS FUNCTION RETURNS THE NO. AGN AND PULSARS IN A PATCH AND THEIR CARTESIAN COORDINATES WITHIN PATCH

    list_row_agn, list_col_agn, list_ltrue_agn, list_btrue_agn = get_pixel_rc_list_from_xml_lb_list(
        xsize_patch, patch_centre, list_agn_xml)

    list_row_psr, list_col_psr, list_ltrue_psr, list_btrue_psr = get_pixel_rc_list_from_xml_lb_list(
        xsize_patch, patch_centre, list_psr_xml)

    agn_pos_list = np.zeros((len(list_row_agn), 6))
    psr_pos_list = np.zeros((len(list_row_psr), 6))

    # agn sources
    # notice that we are using the 0 position for y0=list_row_agn and the first position for x0=list_col_agn
    for i in range(len(list_row_agn)):
        y0 = list_row_agn[i]
        x0 = list_col_agn[i]

        ltrue = list_ltrue_agn[i]
        btrue = list_btrue_agn[i]

        agn_pos_list[i][0] = y0
        agn_pos_list[i][1] = x0
        agn_pos_list[i][2] = ltrue
        agn_pos_list[i][3] = btrue

    # psr sources
    for i in range(len(list_row_psr)):
        y0 = list_row_psr[i]
        x0 = list_col_psr[i]

        ltrue = list_ltrue_psr[i]
        btrue = list_btrue_psr[i]

        psr_pos_list[i][0] = y0
        psr_pos_list[i][1] = x0
        psr_pos_list[i][2] = ltrue
        psr_pos_list[i][3] = btrue

    return len(list_row_agn), len(list_row_psr), agn_pos_list, psr_pos_list
