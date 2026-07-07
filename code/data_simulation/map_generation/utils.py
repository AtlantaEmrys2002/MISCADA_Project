import healpy as hp


def angle_to_healpix_pixels(coordinates, nside: int):

    # Transform shape from num_sources x 2 to 2 x num_sources
    coordinates = coordinates.T

    # Extract two lists - one of galactic latitudes and one of galactic longitudes
    latitudes = coordinates[0]
    longitudes = coordinates[1]

    pixels = hp.pixelfunc.ang2pix(nside=nside, theta=latitudes, phi=longitudes, lonlat=True)

    return pixels


def get_nside(healpix_exposure_map):

    # Healpix parameter nside can be calculated with this function

    nside = hp.pixelfunc.get_nside(healpix_exposure_map)

    return nside

