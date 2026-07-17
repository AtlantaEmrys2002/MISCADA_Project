# PLEASE NOTE THAT THE CODE IN THIS FILE WAS HEAVILY INFLUENCED AND ADAPTED FROM ID8's GitHub Code
# WHICH CAN BE FOUND HERE: https://github.com/bapanes/AutoSourceID/blob/main/codes/from-cats-to-locnet-input.py


# author: Boris Panes, February 4, 2021
# this code contains several contributions from Christopher Eckner, Gulli and Roberto
# specially concerning patch generation and photon flux

import os

from astropy.coordinates import SkyCoord

import numpy as np
import healpy as hp
import math as mt

from bisect import bisect

np.random.seed()

# new approach
def get_ps_info_128(patch_centre, list_agn_xml, list_psr_xml, xsize_patch):
    list_row_agn, list_col_agn, list_flux_agn_1000, list_flux_agn_10000, list_ltrue_agn, list_btrue_agn = get_pixel_rc_list_from_xml_lb_list(
        xsize_patch, patch_centre, list_agn_xml)

    list_row_psr, list_col_psr, list_flux_psr_1000, list_flux_psr_10000, list_ltrue_psr, list_btrue_psr = get_pixel_rc_list_from_xml_lb_list(
        xsize_patch, patch_centre, list_psr_xml)

    agn_pos_list = np.zeros((len(list_row_agn), 6))
    psr_pos_list = np.zeros((len(list_row_psr), 6))

    # agn sources
    # notice that we are using the 0 position for y0=list_row_agn and the first position for  x0=list_col_agn
    for i in range(len(list_row_agn)):
        y0 = list_row_agn[i]
        x0 = list_col_agn[i]

        flux_1000 = list_flux_agn_1000[i]
        flux_10000 = list_flux_agn_10000[i]

        ltrue = list_ltrue_agn[i]
        btrue = list_btrue_agn[i]

        agn_pos_list[i][0] = y0
        agn_pos_list[i][1] = x0
        agn_pos_list[i][2] = ltrue
        agn_pos_list[i][3] = btrue
        agn_pos_list[i][4] = flux_1000
        agn_pos_list[i][5] = flux_10000

    # psr sources
    for i in range(len(list_row_psr)):
        y0 = list_row_psr[i]
        x0 = list_col_psr[i]

        flux_1000 = list_flux_psr_1000[i]
        flux_10000 = list_flux_psr_10000[i]

        ltrue = list_ltrue_psr[i]
        btrue = list_btrue_psr[i]

        psr_pos_list[i][0] = y0
        psr_pos_list[i][1] = x0
        psr_pos_list[i][2] = ltrue
        psr_pos_list[i][3] = btrue
        psr_pos_list[i][4] = flux_1000
        psr_pos_list[i][5] = flux_10000

    return len(list_row_agn), len(list_row_psr), agn_pos_list, psr_pos_list


def distance(y0, x0, y1, x1):
    return mt.sqrt((mt.pow(y1 - y0, 2) + mt.pow(x1 - x0, 2)))


def psf_bck_mask(y0, x0, radius, psf_mask):
    nrow = psf_mask.shape[0]
    ncol = psf_mask.shape[1]

    grid2D_psf = psf_mask.copy()
    grid2D_bck = np.ones((nrow, ncol)) - grid2D_psf

    for y in range(nrow):
        for x in range(ncol):

            # distance to the center
            s1 = distance(y0, x0, y, x)

            if (s1 < radius):
                grid2D_psf[y, x] = 1.0
                grid2D_bck[y, x] = 0.0

    return grid2D_psf, grid2D_bck


def get_lb_from_rd(ra, dec):
    # From galactic to equatorial coordinates
    # https://docs.astropy.org/en/stable/coordinates/transforming.html

    sc = SkyCoord(ra=ra, dec=dec, unit='deg', frame='fk5')
    lon, lat = sc.galactic.l.degree, sc.galactic.b.degree

    if (lon > 180):
        lon = lon - 360
    if (lon < -180):
        lon = 360 + lon

    return lon, lat


def id_pixel(row, col, xsize_patch):
    return xsize_patch * row + col


# def create_dataset(folder, file="training.csv", n=50, height=128, width=128, prefix="test",
#                    n_classes=2, faint="F0", init_con=0):

def create_dataset(folder, file="training.csv", n=50, prefix="test",

    # for cat_number in range(int(n / max_patches_per_catalog)):
    #
    #     source_lines = []
    #
    #     if (n < (cat_number + 1) * max_patches_per_catalog):
    #         patches = n - cat_number * max_patches_per_catalog
    #         if patches == 0:
    #             break

        # part of the angle area correction
        # iem /= pix_sr
        # agn /= pix_sr
        # psr /= pix_sr
        ##############################################################

        # for k in range(patches):
        #
        #     patch_iem = []
        #     patch_agn = []
        #     patch_psr = []
        #
        #     # transformation from 0-360 to -180-180
        #     lon = (longitude[k] + 180) % 360 - 180
        #
        #     lat = latitude[k]

            # for i in range(Nbins):
            #     # notice that now we multiply each bin array by solid_area_ratio
            #
            #     patch_agn_tmp = hp.visufunc.cartview(agn[i], rot=(lon, lat, 0.), coord='G',
            #                                          xsize=xsize_patch_generation,
            #                                          lonra=lb_range, latra=lb_range, return_projected_map=True)
            #     patch_agn.append(np.array(patch_agn_tmp) * solid_area_ratio)
            #
            #     patch_psr_tmp = hp.visufunc.cartview(psr[i], rot=(lon, lat, 0.), coord='G',
            #                                          xsize=xsize_patch_generation,
            #                                          lonra=lb_range, latra=lb_range, return_projected_map=True)
            #     patch_psr.append(np.array(patch_psr_tmp) * solid_area_ratio)
            #
            #     patch_iem_tmp = hp.visufunc.cartview(iem[i], rot=(lon, lat, 0.), coord='G',
            #                                          xsize=xsize_patch_generation,
            #                                          lonra=lb_range, latra=lb_range, return_projected_map=True)
            #     patch_iem.append(np.array(patch_iem_tmp) * solid_area_ratio)

            # IEM_element = np.array(patch_iem)
            # AGN_element = np.array(patch_agn)
            # PSR_element = np.array(patch_psr)
            # centre_coordinate = np.array((lon, lat))

            # we recover the info using 128x128 patch dimensions
            # # this remains from our initial approach. it does not affect at all the image and mask generation
            # nagn, npsr, agn_pos_list, psr_pos_list = get_ps_info_128(centre_coordinate, list_of_agn_lb_from_xml,
            #                                                          list_of_psr_lb_from_xml, xsize_location)
            #
            # print('patch number: %d,  coordinate center: (%.2f, %.2f) ' % (k, lon, lat))

            # output generation
            # IEM_64 = np.zeros((xsize_patch_generation, xsize_patch_generation, Nbins), dtype=np.float32)
            # AGN_64 = np.zeros((xsize_patch_generation, xsize_patch_generation, Nbins), dtype=np.float32)
            # PSR_64 = np.zeros((xsize_patch_generation, xsize_patch_generation, Nbins), dtype=np.float32)
            #
            # for bin_k in range(Nbins):
            #     IEM_64[:, :, bin_k] = IEM_element[bin_k, :, :]
            #     AGN_64[:, :, bin_k] = AGN_element[bin_k, :, :]
            #     PSR_64[:, :, bin_k] = PSR_element[bin_k, :, :]
            #
            # # generation of the input image
            # # we save IEM, AGN and PSR info separetely because it is useful for evaluations
            # X_64 = np.zeros((3, 64, 64, 5))
            #
            # X_64[0, :, :, :] = IEM_64
            # X_64[1, :, :, :] = AGN_64
            # X_64[2, :, :, :] = PSR_64
            #
            # # generation of patch total (sum of components) in image (tensor) like files
            # out_fn = f"{prefix}_image_{init_con + (cat_number * max_patches_per_catalog + k)}.npy"
            # np.save(os.path.join(folder, out_fn), X_64)

            # generation of csv file content
            grid2D_psf = np.zeros((xsize_patch_generation, xsize_patch_generation))
            grid2D_bck = np.ones((xsize_patch_generation, xsize_patch_generation))
            radius_64 = 2.5

            # agns
            id = 0
            for i in range(nagn):
                y = min(agn_pos_list[i][0], xsize_location - 1)
                x = min(agn_pos_list[i][1], xsize_location - 1)

                ltrue = agn_pos_list[i][2]
                btrue = agn_pos_list[i][3]

                # true photon flux from xml
                flux_1000 = agn_pos_list[i][4]
                flux_10000 = agn_pos_list[i][5]

                xmin, xmax, ymin, ymax = max(0, x - r), min(xsize_location - 1, x + r), max(0, y - r), min(
                    xsize_location - 1, y + r)

                xmin = xmin // 2
                xmax = xmax // 2
                ymin = ymin // 2
                ymax = ymax // 2

                # generation of the masks
                # print("agn: ", nagn, xmin,xmax,ymin,ymax)
                grid2D_psf, grid2D_bck = psf_bck_mask(y // 2, x // 2, radius_64, grid2D_psf)

                source_lines.append(f"{out_fn},{int(xmin)},{int(xmax)},{int(ymin)},{int(ymax)}," +
                                    f"{int(id)},{float(lon)},{float(lat)},{float(flux_1000)}," +
                                    f"{float(ltrue)},{float(btrue)},{int(catalog_id)},{float(flux_10000)}\n")

            # pulsars
            id = 1
            for i in range(npsr):
                y = min(psr_pos_list[i][0], xsize_location - 1)
                x = min(psr_pos_list[i][1], xsize_location - 1)

                ltrue = psr_pos_list[i][2]
                btrue = psr_pos_list[i][3]

                # true photon flux from xml
                flux_1000 = psr_pos_list[i][4]
                flux_10000 = psr_pos_list[i][5]

                xmin, xmax, ymin, ymax = max(0, x - r), min(xsize_location - 1, x + r), max(0, y - r), min(
                    xsize_location - 1, y + r)

                xmin = xmin // 2
                xmax = xmax // 2
                ymin = ymin // 2
                ymax = ymax // 2

                grid2D_psf, grid2D_bck = psf_bck_mask(y // 2, x // 2, radius_64, grid2D_psf)

                source_lines.append(f"{out_fn},{int(xmin)},{int(xmax)},{int(ymin)},{int(ymax)}," +
                                    f"{int(id)},{float(lon)},{float(lat)},{float(flux_1000)}," +
                                    f"{float(ltrue)},{float(btrue)},{int(catalog_id)},{float(flux_10000)}\n")

            Y_p = np.zeros((xsize_patch_generation, xsize_patch_generation, 2))
            Y_p[:, :, 0] = grid2D_psf
            Y_p[:, :, 1] = grid2D_bck

            out_mk = f"{prefix}_masks_{init_con + cat_number * max_patches_per_catalog + k}.npy"
            np.save(os.path.join(folder, out_mk), Y_p)

            # we write just at the end of the process to avoid the repeated opening of the file

        f1 = open(os.path.join(folder, file), "a")
        f1.writelines(source_lines)
        f1.close()

    return 0


if __name__ == "__main__":

    # Make directory in which patches are stored
    Path("./simulated_data/patches/").mkdir(parents=True, exist_ok=True)



# February 4, 2020
# given AGN, PSR and Background fits file, generate 768 patches per sky instance

# Gulli's approach to generate a more uniform coverage of the sky
longitude, latitude = hp.pix2ang(8, np.arange(hp.nside2npix(8)), lonlat=True)

# patches per catalog
max_patches_per_catalog = len(longitude)

# catalog list
catalog_list = []
catalog_list.append(400)

# generate patch catalog
# previous_plot_backend = matplotlib.get_backend()
# matplotlib.use('Agg')
# cats_test = create_dataset(test_folder, file="test.csv", prefix="test", n=len(longitude), faint="F0", init_con=0)
# matplotlib.use(previous_plot_backend)