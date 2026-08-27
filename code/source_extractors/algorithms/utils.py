# THE CODE IN THIS FILE WAS TAKEN (AND ADAPTED FROM ID8 PAPER'S CODE) - SEE BELOW FOR REFERENCING INFORMATION.

# I HAVE OPTIMISED/MODIFIED SEVERAL FUNCTIONS

# The link to this code can be found here: https://github.com/bapanes/AutoSourceID/blob/main/codes/from-cats-to-locnet-
# input.py

# author: Boris Panes, February 4, 2021
# this code contains several contributions from Christopher Eckner, Gulli and Roberto
# specially concerning patch generation and photon flux

# This code helps project the all-sky maps into Cartesian coordinates and slice them into 64 x 64 images.

from astropy.table import QTable
import numpy as np


def get_catalog_data(catalog_file):
    catalog = QTable.read(catalog_file, format='fits', hdu=1)

    columns = ("Source_Name", "CLASS1", "RAJ2000", "DEJ2000")

    catalog = catalog[columns]

    # Reformat CLASS1 column - remove empty spaces and make all lower case
    catalog["CLASS1"] = np.asarray([k.decode('utf-8').strip().lower() for k in catalog["CLASS1"].value.filled('-')])

    # Reformat source name column - remove empty spaces and make all lower case
    catalog["Source_Name"] = np.asarray(
        [k.decode('utf-8').strip().lower()[5:] for k in catalog["Source_Name"].value])

    # Select all rows that describe pulsars
    pulsar_mask = (catalog["CLASS1"] == "psr")

    # Select all rows that describe AGN
    agn_mask = np.isin(catalog["CLASS1"].data,
                       np.array(["bcu", "sey", "ssrq", "bll", "fsrq", "rdg", "nlsy1", "agn"]))

    agns = catalog[agn_mask]
    psrs = catalog[pulsar_mask]

    # Convert to pandas dataframes for covariance and correlation calculations, as well as plotting
    agns = agns.to_pandas()
    pulsars = psrs.to_pandas()

    # Remove sources with NaN values
    pulsars.dropna(inplace=True)
    agns.dropna(inplace=True)

    agn_coordinates_per_catalog = [dict(zip(agns["Source_Name"], agns[["RAJ2000", "DEJ2000"]].to_numpy()))]
    pulsar_coordinates_per_catalog = [dict(zip(pulsars["Source_Name"], pulsars[["RAJ2000", "DEJ2000"]].to_numpy()))]

    return agn_coordinates_per_catalog, pulsar_coordinates_per_catalog


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


def xyz2sph(x, y, z):
    r = np.sqrt(x * x + y * y + z * z)

    phi = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arctan2(z, np.sqrt(x * x + y * y)))

    return np.array([r, lat, phi])


def get_lb_from_pixel(pixel_id, lb_centre, xsize=128):

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
    r, b_PS, l_PS = xyz2sph(x_PS, y_PS, z_PS)

    # N.B. FROM MHR - Fixed this bit - stored my coordinates for l_PS as -180 to 180 instead of 0 to 360 (cause of
    # HEALPIX SYSTEM)
    # l_PS += 180  # to get to 0 - 360

    out_of_range = np.logical_or(l_PS > 360, l_PS < 0)

    l_PS = np.where(out_of_range, l_PS % 360, l_PS)

    # When returning individual values
    if type(pixel_id) is int or type(pixel_id) is np.int64:

        l_PS = l_PS[0]
        b_PS = b_PS[0]

    return l_PS, b_PS


def loss_values(epochs, losses, directory: str, method: str):

    loss_variation = np.vstack((np.array(epochs), np.array(losses))).T

    np.save(directory + f"{method}_loss_per_epoch.npy", loss_variation)


def pixel_id(row, col, xsize_patch):
    return (xsize_patch * row + col).astype(int)
