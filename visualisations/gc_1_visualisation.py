from astropy.io import fits
from astropy.table import QTable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1 degree bins for RA, 0.5 degree bins for Dec
# x = np.zeros((362, 362))

# ra_bins = np.asarray([k for k in range(0, 361)])
#
# dec_bins = np.linspace(-90, 90, 360)

# Large file - use memmap
# with fits.open("./gc_1.fits", memmap=True) as hdul:
#
#     # Select relevant columns
#     events = hdul[1].data['DEC']  # , 'RA', 'DEC', 'ENERGY', 'TIME']
#
#     dec_binned = np.digitize(events, dec_bins)
#
#     events = hdul[1].data['RA']
#
#     ra_binned = np.digitize(events, ra_bins)
#
#     coordinates = np.stack((ra_binned, dec_binned), axis=1)
#
#     np.savetxt('pixels.txt', coordinates)

# pix = np.loadtxt("pixels.txt")
#
# for k in pix:
#     x[tuple(k.astype(np.int64))] += 1

# np.savetxt('count_map.txt', x)

binned_image = np.loadtxt("count_map.txt")


plt.imshow(binned_image, interpolation=None)

plt.show()
