import argparse
from bisect import bisect
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from read_write_functions import xml_parser


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



if __name__ == "__main__":

    # INPUT PARAMETERS AND DATA

    parser = argparse.ArgumentParser(description="Read in simulated all-sky maps and catalogs to create a dataset of "
                                                 "ROIs to be fed to source dectection algorithms.")

    parser.add_argument("--num_catalogs", required=True, type=int, help="The number of simulated catalogs "
                                                                        "and skymaps that have been generated and can be"
                                                                        "split into patches.")

    parser.add_argument("--num_energy_bins", required=True, type=int, help="The number of energy bins that "
                                                                           "photons were binned into using fermitools.")

    parser.add_argument("--num_patches_per_catalog", required=False, type=int, help="The number of patches"
                                                                                    "to generate per sky-map provided.")

    args = parser.parse_args()

    num_catalogs = args.num_catalogs
    num_bins = args.num_energy_bins
    # n = args.num_patches_per_catalog

    # FILE CREATION AND ORGANISATION

    # Create directory where patches stored
    Path("./simulated_data/patches").mkdir(parents=True, exist_ok=True)

    # list for the csv files - stores information about each patch (including true source locations)
    header_line = "filename,xmin,xmax,ymin,ymax,class,lon_c,lat_c,flux_1000,lon_p,lat_p,catalog,flux_10000\n"

    f1 = open(os.path.join("./simulated_data/patches/", "metadata.csv"), "w+")
    f1.writelines(header_line)
    f1.close()


    energy_bins = np.logspace(np.log(300), np.log(200000), num=num_bins + 1, base=np.e)
    xsize_location = 128

    ##################################################
    # global 64x64 correction
    ##################################################
    xsize_patch_generation = 64
    # NSIDE = 256

    # Npix = 12 * NSIDE * NSIDE
    # pix_sr = 4.0 * np.pi / Npix

    lb_range = [-5, 5]
    proj = hp.projector.CartesianProj(lonra=lb_range, latra=lb_range, xsize=xsize_patch_generation)

    I, J = np.meshgrid(np.arange(xsize_patch_generation), np.arange(xsize_patch_generation))
    x, y = proj.ij2xy(I, J)

    # Used to correct the rotation smear
    dl = np.radians((lb_range[1] - lb_range[0])/ (xsize_patch_generation - 1))
    solid_area_ratio = dl * dl * np.cos(np.radians(y))

    # Gulli's approach to generate a more uniform coverage of the sky - specifies the longitudes at which to draw out
    # the slice of the sky to take
    longitude, latitude = hp.pix2ang(8, np.arange(hp.nside2npix(8)), lonlat=True)

    # The number of patches to generate per catalog
    max_patches_per_catalog = len(longitude)

    for c in range(num_catalogs):

        # READ IN MAPS AND SPATIAL/SPECTRAL PARAMETERS

        # Num patches to generate
        patches = max_patches_per_catalog

        skymaps_directory = "./simulated_data/count_maps/skymap_{}".format(c + 1)
        catalogs_directory = "./simulated_data/catalogs/catalog_{}".format(c + 1)

        # Read in the coordinates and associated integral photon fluxes of each source in AGN and pulsar maps
        agn_coordinates, agn_photon_fluxes = xml_parser(energy_bins=energy_bins,
                                                        xml_file=catalogs_directory + "/agns.xml")
        pulsar_coordinates, pulsar_photon_fluxes = xml_parser(energy_bins=energy_bins,
                                                              xml_file=catalogs_directory + "/pulsars.xml")

        binned_agn_map = []
        binned_pulsar_map = []
        binned_background_map = []

        for b in range(num_bins):

            # Read in binned all-sky count maps for AGN, pulsars, and background

            agns = hp.fitsfunc.read_map(filename=skymaps_directory + "/agns_{}.fits".format(b))
            pulsars = hp.fitsfunc.read_map(filename=skymaps_directory + "/pulsars_{}.fits".format(b))
            background = hp.fitsfunc.read_map(filename=skymaps_directory + "/background_{}.fits".format(b))

            binned_agn_map.append(agns)
            binned_pulsar_map.append(pulsars)
            binned_background_map.append(background)

        # Convert to numpy array
        binned_agn_map = np.array(binned_agn_map)
        binned_pulsar_map = np.array(binned_pulsar_map)
        binned_background_map = np.array(binned_background_map)

        # Loop to generate each patch

        for p in range(patches):

            # CENTRE OF PATCH p

            # transformation from 0-360 to -180-180 - healpy uses this coordinate system
            lon = (longitude[p] + 180) % 360 - 180

            lat = latitude[p]

            patch_centre = np.array([lon, lat])

            # PROJECT ROI OF COUNT MAP INTO CARTESIAN
            # N.B. in previous code, they performed a solid angle ratio correction here (assume not needed, as already
            # accounted for this in a previous step? - CHECK UNDERSTANDING)

            binned_agn_patch = []
            binned_pulsar_patch = []
            binned_background_patch = []

            for b in range(num_bins):

                agn_patch_bin = hp.visufunc.cartview(binned_agn_map[b], rot=(lon, lat, 0.), coord='G',
                                                       xsize=xsize_patch_generation, lonra=lb_range, latra=lb_range,
                                                       return_projected_map=True)

                pulsar_patch_bin = hp.visufunc.cartview(binned_pulsar_map[b], rot=(lon, lat, 0.), coord='G',
                                                       xsize=xsize_patch_generation, lonra=lb_range, latra=lb_range,
                                                       return_projected_map=True)

                background_patch_bin = hp.visufunc.cartview(binned_background_map[b], rot=(lon, lat, 0.), coord='G',
                                                       xsize=xsize_patch_generation, lonra=lb_range, latra=lb_range,
                                                       return_projected_map=True)

                binned_agn_patch.append(agn_patch_bin)
                binned_pulsar_patch.append(pulsar_patch_bin)
                binned_background_patch.append(background_patch_bin)

                # Need these here (even though we are not showing the plots - this is because visufunc creates a plot - we
                # only need the 2D array)
                plt.cla()
                plt.clf()
                plt.close("all")

            # Convert to numpy
            binned_agn_patch = np.array(binned_agn_patch)
            binned_pulsar_patch = np.array(binned_pulsar_patch)
            binned_background_patch = np.array(binned_background_patch)

            # GET NO. AGN AND PULSARS WITHIN PATCH, AS WELL AS THE CARTESIAN COORDINATES OF AGN AND PULSARS IN THE PATCH

            # we recover the info using 128x128 patch dimensions
            # this remains from our initial approach. it does not affect at all the image and mask generation
            nagn, npsr, agn_pos_list, psr_pos_list = get_ps_info_128(patch_centre, agn_coordinates,
                                                                     pulsar_coordinates, xsize_location)





