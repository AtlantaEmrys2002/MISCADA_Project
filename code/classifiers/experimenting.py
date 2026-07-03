from astropy.table import QTable
from astropy.io import fits

# THERE EXAMPLE

# file_name = "/Volumes/T7/MOCK_4FGL_psr_400_healpix.fits"

# hdul = fits.open(file_name)

# print(hdul.info())

# print(hdul[1].info())

# file_name = "/Volumes/T7/data/sky_map_creation_data/fermi_filtered_exposure_map.fits"

file_name = "/Volumes/T7/project_data/real_data/fermi_filtered_gti_exposure_map.fits"

# file_name = "/Volumes/T7/data/background_models/gll_iem_v07.fits"


hdul = fits.open(file_name)

print(hdul.info())

print(hdul[0].shape)


