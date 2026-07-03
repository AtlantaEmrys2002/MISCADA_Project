# THIS IS TO CREATE AN EMPTY FITS FILE WHEN DOING FERMITOOLS CUTS

from astropy.io import fits
from astropy.table import QTable
import numpy as np

with fits.open("/Volumes/T7/data/empty_evfile.fits") as hdul:
   hdul.info()

   hdul[1].data = hdul[1].data[1: 2]

   hdul.info()

   # print(hdul[1].data.shape)
   #
   # bad = np.logical_or.reduce([np.isnan(col) for col in hdul.itercols()])
   #
   # print(hdul[bad])
   #
   # header_copy = hdul[0].header.copy()

   # print(header_copy)
   #
   # header_copy.writeto("./test_header.fits", None, header_copy)

