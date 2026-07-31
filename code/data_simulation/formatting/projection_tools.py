"""
This code was adapted from code presented in ID8's https://github.com/bapanes/AutoSourceID/blob/main/codes/from-cats-to
-locnet-input.py. I have optimised and rewritten sections of it to improve performance and tailor it to my use case.

The following note was included with the original code:

author: Boris Panes, February 4, 2021
this code contains several contributions from Christopher Eckner, Gulli and Roberto
specially concerning patch generation and photon flux

This code is the mechanism for projecting the all-sky maps into Cartesian coordinates and slicing them into 64 x 64
images.
"""

import numpy as np

# CONSTANTS

coord_range_x = np.linspace(-4.9609375, 4.9609375, 128)
coord_range_y = np.linspace(-4.9609375, 4.9609375, 128)


def RotMatrixY(psi):
    """Calculates matrix to rotates point about the Y-axis by given angle psi.

    Parameters
    ----------
    psi: np.float64
        Angle to rotate coordinate about the y-axis.

    Returns
    -------
    ndarray
        Rotational matrix.

    """
    # Convert to radians
    psi_rad = np.radians(psi)

    sin_psi = np.sin(psi_rad)
    cos_psi = np.cos(psi_rad)

    return np.array([[cos_psi, 0.0, -sin_psi],
                     [0.0, 1.0, 0.0],
                     [sin_psi, 0.0, cos_psi]])


def RotMatrixZ(psi):
    """Calculates matrix to rotates point about the Z-axis by given angle psi.

    Parameters
    ----------
    psi: np.float64
        Angle to rotate coordinate about the z-axis.

    Returns
    -------
    ndarray
        Rotational matrix.

    """
    # Convert to radians
    psi_rad = np.radians(psi)

    sin = np.sin(psi_rad)
    cos = np.cos(psi_rad)

    return np.array([[cos, sin, 0.0], [-sin, cos, 0.0], [0.0, 0.0, 1.0]])


def sph2xyz(r, theta, phi):
    """Calculates Cartesian x, y, z coordinates given spherical coordinates.

    See https://en.wikipedia.org/wiki/Spherical_coordinate_system

    Parameters
    ----------
    r
        radial distance
    theta
        polar angle
    phi
        azimuthal angle

    Returns
    -------
    ndarray
        x, y, z coordinates of same point in space.

    """
    # Convert theta and phi to radians
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)

    sin_theta = np.sin(theta_rad)

    new = np.array([r * sin_theta * np.cos(phi_rad),
                    r * sin_theta * np.sin(phi_rad),
                    r * np.cos(theta_rad)])

    return new


def xyz2sph(x, y, z):
    """Calculates spherical coordinates given Cartesian x, y, z coordinates.

    Parameters
    ----------
    x
        Coordinate along the x-axis.
    y
        Coordinate along the y-axis.
    z
        Coordinate along the z-axis.

    Returns
    -------
    ndarray
        Returns corresponding parts of the spherical coordinates that are relevant.

    """
    phi = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arctan2(z, np.sqrt(x * x + y * y)))

    return np.array([lat, phi])


def get_lb_ps_centered(lb_ps, lb_centre):
    """ Centres central lon, lat of patch centred in 64 x 64 image.

    Parameters
    ----------
    lb_ps
        Centre of patch in galactic coordinates.
    lb_centre
        Desired patch centre.

    """
    l_centre, b_centre = lb_centre
    r = np.dot(RotMatrixY(-b_centre), RotMatrixZ(l_centre))

    lon_PS_rotated, lat_PS_rotated = lb_ps

    xyz_PS_rotated = sph2xyz(1., 90. - lat_PS_rotated, lon_PS_rotated)
    x_PS, y_PS, z_PS = np.array(np.dot(r, xyz_PS_rotated), dtype='float32')

    b_PS, l_PS = xyz2sph(x_PS, y_PS, z_PS)

    return l_PS, b_PS


def get_lb_from_pixel(pixel_id, lb_centre):
    """Get galactic coordinate map of 10 x 10 degree patch.

    Params
    ------
    pixel_id
        Pixel to calculate longitude and latitude for.
    lb_centre
        Centre of patch in longitude and latitude.

    Returns
    ------
    ndarray
        Longitude and latitude of pixel specified with pixel_id.

    """

    lonlat_patch = np.load("template.npy")

    # Rotation matrix around (0, 0)
    l_centre, b_centre = lb_centre
    r = np.dot(RotMatrixY(-b_centre), RotMatrixZ(l_centre))

    lon_PS_rotated, lat_PS_rotated = lonlat_patch[pixel_id]

    xyz_PS_rotated = sph2xyz(1., 90. - lat_PS_rotated, lon_PS_rotated)
    x_PS, y_PS, z_PS = np.array(np.dot(r.T, xyz_PS_rotated), dtype='float32')

    b_PS, l_PS = xyz2sph(x_PS, y_PS, z_PS)

    return l_PS, b_PS


def get_xml_lb_list_in_std_patch_coord(patch_centre, list_of_pos_xml, source_ids):
    """Get coordinates of source in patch coordinate system from galactic coordinates of source (returning the positions
    of sources present within the given patch and ignoring all others).

    Parameters
    ----------
    patch_centre
        Centre of patch in galactic coordinates.
    list_of_pos_xml
        Galactic coordinates of each source in sky.
    source_ids
        Unique identifiers of each source in sky.

    """

    # corners around the patch_centre_position

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

        if (l_c >= l_c_min) & (l_c <= l_c_max) & (b_c >= b_c_min) & (b_c <= b_c_max):
            list_of_lb_in_std_patch_coord.append((l_c, b_c, l_pos, b_pos))

            list_of_ids.append(source_ids[pos_con])

    return list_of_lb_in_std_patch_coord, list_of_ids


def get_pixel_rc_list_from_xml_lb_list(lb_centre, list_of_pos_xml, source_ids):
    """Completely rewrote this function, especially, as it did not make use of numpy. It converts of the galactic
    coordinates of a source to a position in the 64 x 64 image.

    Parameters
    ----------
    lb_centre
        Centre of patch.
    list_of_pos_xml
        Galactic coordinates of each source.
    source_ids
        Unique identifier of each source.

    """
    # FUNCTION CONVERTS LON, LAT POSITION OF SOURCE TO POSITION IN IMAGE (IF THERE)

    lb_std_list, list_of_ids = get_xml_lb_list_in_std_patch_coord(lb_centre, list_of_pos_xml, source_ids)

    lb_std_list = np.array(lb_std_list)

    if lb_std_list.shape[0] == 0:

        coordinate_information = np.array([])

    else:

        # y0 and x0 in pos 0 and 1 respectively because of image indexing in python - image has coordinates formatted as
        # y, x instead of x, y
        # Per row, pos 0 holds y0 in image, pos 1 holds x0 in image, pos 2 holds true l and pos 3 holds true b
        list_of_pixel_ltrue = lb_std_list[:, 2]
        list_of_pixel_btrue = lb_std_list[:, 3]
        list_of_pixel_col = 128 - np.searchsorted(coord_range_x, lb_std_list[:, 0], side='right')
        list_of_pixel_row = np.searchsorted(coord_range_y, lb_std_list[:, 1], side='right')

        coordinate_information = np.stack((list_of_pixel_row, list_of_pixel_col, list_of_pixel_ltrue,
                                           list_of_pixel_btrue), axis=1)

    return coordinate_information, list_of_ids


def get_ps_info_128(patch_centre, list_agn_xml, list_psr_xml, agn_ids, pulsar_ids):
    """Assuming size of desired patch is equal to 128 x 128 pixels to increase precision, this function returns the
    number of AGN and pulsars within a patch, as well as their Cartesian coordinates within the patch, and their true
    galactic coordinates.

    Parameters
    ----------
    patch_centre
        Centre of patch in galactic coordinates.
    list_agn_xml
        List of AGNs' galactic coordinates.
    list_psr_xml
        List of pulsars' galactic coordinates.
    agn_ids
        Unique identifiers of each AGN in sky.
    pulsar_ids
        Unique identifiers of each pulsar in sky.


    """
    # Assume size of patch (when deriving coordinates) is equal to 128 x 128 (not 64 x 64) to increase precision

    # Always pass AGN and pulsar longitudes and latitudes in degrees and not radians

    # THIS FUNCTION RETURNS THE NO. AGN AND PULSARS IN A PATCH AND THEIR CARTESIAN COORDINATES WITHIN PATCH

    agn_coordinate_information, list_agn_ids = get_pixel_rc_list_from_xml_lb_list(patch_centre, list_agn_xml, agn_ids)

    psr_coordinate_information, list_psr_ids = get_pixel_rc_list_from_xml_lb_list(patch_centre, list_psr_xml,
                                                                                  pulsar_ids)

    num_agn_in_patch = agn_coordinate_information.shape[0]
    num_pulsar_in_patch = psr_coordinate_information.shape[0]

    return (num_agn_in_patch, num_pulsar_in_patch, agn_coordinate_information, psr_coordinate_information, list_agn_ids,
            list_psr_ids)
