from bisect import bisect
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

# THE CODE BELOW WAS TAKEN (AND ADAPTED FROM ID8 PAPER'S CODE) - SEE BELOW FOR REFERENCING INFORMATION

# The link to this code can be found here: https://github.com/bapanes/AutoSourceID/blob/main/codes/from-cats-to-locnet-
# input.py

# author: Boris Panes, February 4, 2021
# this code contains several contributions from Christopher Eckner, Gulli and Roberto
# specially concerning patch generation and photon flux

# This code helps project the all-sky maps into Cartesian coordinates and slice them into 64 x 64 images.

# CONSTANTS

coord_range_x = np.linspace(-4.9609375, 4.9609375, 128)
coord_range_y = np.linspace(-4.9609375, 4.9609375, 128)


def RotMatrixY(psi):

    # Convert to radians
    psi_rad = np.radians(psi)

    sin_psi = np.sin(psi_rad)
    cos_psi = np.cos(psi_rad)

    return np.array([[cos_psi, 0.0, -sin_psi],
                     [0.0, 1.0, 0.0],
                     [sin_psi, 0.0, cos_psi]])


def RotMatrixZ(psi):

    # Convert to radians
    psi_rad = np.radians(psi)

    sin = np.sin(psi_rad)
    cos = np.cos(psi_rad)

    return np.array([[cos, sin, 0.0], [-sin, cos, 0.0], [0.0, 0.0, 1.0]])


def sph2xyz(r, theta, phi):

    # Convert theta and phi to radians
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)

    new = np.array([r * np.sin(theta_rad) * np.cos(phi_rad),
                    r * np.sin(theta_rad) * np.sin(phi_rad),
                    r * np.cos(theta_rad)])

    return new


def xyz2sph(x, y, z):

    phi = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arctan2(z, np.sqrt(x * x + y * y)))

    return np.array([lat, phi])


# AGAIN BELOW USED TO GET L, B IN PATCH COORDINATES
# Function to implement inverse rotation
# to add in the predictions?
def get_lb_ps_centered(lb_ps, lb_centre):

    l_centre, b_centre = lb_centre
    r = np.dot(RotMatrixY(-b_centre), RotMatrixZ(l_centre))

    lon_PS_rotated, lat_PS_rotated = lb_ps

    xyz_PS_rotated = sph2xyz(1., 90. - lat_PS_rotated, lon_PS_rotated)
    x_PS, y_PS, z_PS = np.array(np.dot(r, xyz_PS_rotated), dtype='float32')

    b_PS, l_PS = xyz2sph(x_PS, y_PS, z_PS)

    return l_PS, b_PS


def get_lb_from_pixel(pixel_id, lb_centre):
    ######### Generate (l,b) coordinate map of 10x10deg patch ######

    # Following the suggestions of CA mail when generating template - THIS ASSUMES xsize=128 - otherwise would have to
    # introduce conditional statement

    lonlat_patch = np.load("template.npy")

    ######### Get rotation matrix used to rotate the original centre to (0., 0.) #########
    l_centre, b_centre = lb_centre
    r = np.dot(RotMatrixY(-b_centre), RotMatrixZ(l_centre))
    #########

    lon_PS_rotated, lat_PS_rotated = lonlat_patch[pixel_id]

    xyz_PS_rotated = sph2xyz(1., 90. - lat_PS_rotated, lon_PS_rotated)
    x_PS, y_PS, z_PS = np.array(np.dot(r.T, xyz_PS_rotated), dtype='float32')

    b_PS, l_PS = xyz2sph(x_PS, y_PS, z_PS)

    return l_PS, b_PS


def get_xml_lb_list_in_std_patch_coord(patch_centre, list_of_pos_xml, source_ids):

    # AGAIN USED TO GET COORDINATES OF SOURCE WITH L, B COORD INTO PATCH COORD SYS

    # corners around the patch_centre_position

    # GET THE LAT AND LON OF THE FOUR CORNERS OF THE 128 x 128 PATCH
    l_a, b_a = get_lb_from_pixel(0, patch_centre)
    l_b, b_b = get_lb_from_pixel(16256, patch_centre)
    l_c, b_c = get_lb_from_pixel(127, patch_centre)
    l_d, b_d = get_lb_from_pixel(16383, patch_centre)

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

    list_of_ids = []

    for pos_con in range(len(list_of_pos_xml)):

        # positions in global lon lat coordinates
        l_pos = list_of_pos_xml[pos_con][0]
        b_pos = list_of_pos_xml[pos_con][1]

        # positions in standard patch coordinates
        l_c, b_c = get_lb_ps_centered((l_pos, b_pos), patch_centre)

        if ((l_c >= l_c_min) & (l_c <= l_c_max) & (b_c >= b_c_min) & (b_c <= b_c_max)):

            list_of_lb_in_std_patch_coord.append((l_c, b_c, l_pos, b_pos))

            list_of_ids.append(source_ids[pos_con])

    return list_of_lb_in_std_patch_coord, list_of_ids


def get_pixel_rc_list_from_xml_lb_list(lb_centre, list_of_pos_xml, source_ids):

    # FUNCTION CONVERTS LON, LAT POSITION OF SOURCE TO POSITION IN IMAGE (IF THERE)

    lb_std_list, list_of_ids = get_xml_lb_list_in_std_patch_coord(lb_centre, list_of_pos_xml, source_ids)

    list_of_pixel_row = []
    list_of_pixel_col = []

    list_of_pixel_ltrue = []
    list_of_pixel_btrue = []
    # l = longitude
    # b = latitude

    num_source_type = len(lb_std_list)

    for i in range(num_source_type):

        l, b, ltrue, btrue = lb_std_list[i][0], lb_std_list[i][1], lb_std_list[i][2], lb_std_list[i][3]

        # Again assuming that xsize will always be 128
        pixel_l = 128 - bisect(list(coord_range_x), l)
        pixel_b = bisect(list(coord_range_y), b)

        list_of_pixel_row.append(pixel_b)
        list_of_pixel_col.append(pixel_l)

        list_of_pixel_ltrue.append(ltrue)
        list_of_pixel_btrue.append(btrue)

    return list_of_pixel_row, list_of_pixel_col, list_of_pixel_ltrue, list_of_pixel_btrue, list_of_ids


def get_ps_info_128(patch_centre, list_agn_xml, list_psr_xml, agn_ids, pulsar_ids):

    # Assume size of patch (when deriving coordinates) is equal to 128 x 128 (not 64 x 64) to increase precision

    # Always pass AGN and pulsar longitudes and latitudes in degrees and not radians

    # THIS FUNCTION RETURNS THE NO. AGN AND PULSARS IN A PATCH AND THEIR CARTESIAN COORDINATES WITHIN PATCH

    list_row_agn, list_col_agn, list_ltrue_agn, list_btrue_agn, list_agn_ids = get_pixel_rc_list_from_xml_lb_list(
        patch_centre, list_agn_xml, agn_ids)

    list_row_psr, list_col_psr, list_ltrue_psr, list_btrue_psr, list_psr_ids = get_pixel_rc_list_from_xml_lb_list(
        patch_centre, list_psr_xml, pulsar_ids)

    num_agn_in_patch = len(list_row_agn)
    num_pulsar_in_patch = len(list_row_psr)

    # agn sources

    # y0 and x0 in pos 0 and 1 respectively because of image indexing in python - image has coordinates formatted as y,
    # x instead of x, y
    # Per row, pos 0 holds y0 in image, pos 1 holds x0 in image, pos 2 holds true l and pos 3 holds true b
    agn_pos_list = np.array([[list_row_agn[i], list_col_agn[i], list_ltrue_agn[i], list_btrue_agn[i]] for i in
                             range(num_agn_in_patch)])

    # psr sources
    psr_pos_list = np.array([[list_row_psr[i], list_col_psr[i], list_ltrue_psr[i], list_btrue_psr[i]] for i in
                             range(num_pulsar_in_patch)])

    return num_agn_in_patch, num_pulsar_in_patch, agn_pos_list, psr_pos_list, list_agn_ids, list_psr_ids
