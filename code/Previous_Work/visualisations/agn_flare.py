from astropy.io import fits
from astropy.table import QTable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation

# READ DATA

events = QTable.read('./agn_flare_2/agn_gti.fits', format='fits', hdu=1)

events = events['RA', 'DEC', 'TIME']

# BIN DATA

# Bin data into "days"

days = 25

months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

ranges = np.linspace(min(events['TIME'].value), max(events['TIME'].value), days + 1)

# 0.1 degree x 0.1 degree represented by 1 pixel
xedges_bins = np.linspace(325, 335, 100)
yedges_bins = np.linspace(37, 47, 100)

# PLOT DATA

plt.rcParams["figure.figsize"] = [10.00, 10.00]
plt.rcParams["figure.autolayout"] = True

fig, ax = plt.subplots()

# First day
photons_that_day_mask = (events['TIME'].value >= ranges[0]) & (events['TIME'].value < ranges[1])
photons_that_day = events[photons_that_day_mask]

H, xedges, yedges = np.histogram2d(photons_that_day['RA'].value, photons_that_day['DEC'].value, bins=(xedges_bins, yedges_bins))

H = H.T

X, Y = np.meshgrid(xedges, yedges)

fax = ax.pcolormesh(X, Y, H)

frames = []

im = plt.imshow(H, interpolation=None, origin='lower')

plt.title('JAN')

# Rest of the month
for d in range(0, days - 1):

    photons_that_day_mask = (events['TIME'].value >= ranges[d]) & (events['TIME'].value < ranges[d + 1])
    photons_that_day = events[photons_that_day_mask]

    H, xedges, yedges = np.histogram2d(photons_that_day['RA'].value, photons_that_day['DEC'].value, bins=(xedges_bins, yedges_bins))

    H = H.T

    frames.append(H)

# frames = np.asarray([np.sqrt(f) for f in frames])


def animate(frame):

    im.set_data(frames[frame])

    plt.title(months[frame // 2] + str(frame))

    print(frame)

    return fax,


# Animate
anim = animation.FuncAnimation(fig, animate, frames=range(0, days - 1), blit=False, repeat=False)

writer = animation.PillowWriter(fps=2)

anim.save("./agn_flare_fps2_sqrt.gif", writer=writer,)

plt.show()

# REFERENCES

# AGN Flare Event - https://astrobites.org/2024/11/13/super-fast-bllac-flare/
# Animating Histograms - https://stackoverflow.com/questions/48395209/how-to-animate-an-image-derived-from-a-2d-
# histogram
# Fermi Query - https://fermi.gsfc.nasa.gov/cgi-bin/ssc/LAT/QueryResults.cgi?id=L2605221114164D7EA5AE60
# Matplotlib Documentation - https://matplotlib.org/stable/gallery/animation/simple_anim.html
# Numpy Documentation - https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html
# Original BL Lac - https://en.wikipedia.org/wiki/BL_Lacertae
# Printing Title - https://stackoverflow.com/questions/67532491/how-to-print-image-with-title-in-matplotlib
# Saving Animation - https://stackoverflow.com/questions/64529921/the-saved-animated-plot-keeps-looping-although-
# matplotlib-funcanimation-repe


