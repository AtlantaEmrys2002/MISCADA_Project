from astropy.table import QTable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# READ DATA

events = QTable.read('./agn_flare_2/agn_gti.fits', format='fits', hdu=1)

events = events['RA', 'DEC', 'TIME', 'ENERGY']

# BIN DATA

# Bin data into "days" (half months here)
days = 25

months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

ranges = np.linspace(min(events['TIME'].value), max(events['TIME'].value), days + 1)

# 0.1 degree x 0.1 degree represented by 1 pixel
# xedges_bins = np.linspace(325, 335, 100)
# yedges_bins = np.linspace(37, 47, 100)

xedges_bins = np.linspace(310, 360, 400)
yedges_bins = np.linspace(22, 62, 400)

# PLOT DATA

plt.rcParams["figure.figsize"] = [10.00, 10.00]
plt.rcParams["figure.autolayout"] = True

fig, ax = plt.subplots()

plt.title('October 2024 AGN Flare')

# N.B. IMPORTANT TO KEEP xedges and yedges the same (not adapting for each month) to ensure event visbility
photons_that_day_mask = (events['TIME'].value >= ranges[0]) & (events['TIME'].value < ranges[1])
photons_that_day = events[photons_that_day_mask]
H, xedges, yedges = np.histogram2d(photons_that_day['RA'].value, photons_that_day['DEC'].value,
                                   bins=(xedges_bins, yedges_bins))

# Spatial correction
H = H.T

# Recommended in tutorials to "accentuate faint maxima"
H = np.sqrt(H)

# Do not need pcolormesh for uniform bins (we are doing 0.1 x 0.1 degree squares)
im = plt.imshow(H, interpolation=None, origin='lower')

# SELECT OCTOBER

photons_that_day_mask = (events['TIME'].value >= ranges[19]) & (events['TIME'].value < ranges[20])
photons_that_day = events[photons_that_day_mask]

# Fermi energy bins
energy_bins = [50, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000]

frames = []

# Use 4FGL energy bins to classify photons emitted during October flaring conditions
for e in range(len(energy_bins) - 1):

    mask = (photons_that_day['ENERGY'].value >= energy_bins[e]) & (photons_that_day['ENERGY'].value < energy_bins[e + 1])
    energy_binned_photons = photons_that_day[mask]

    H, _, _ = np.histogram2d(energy_binned_photons['RA'].value, energy_binned_photons['DEC'].value,
                             bins=(xedges_bins, yedges_bins))

    H = H.T

    frames.append(H)


def animate(frame):

    im.set_data(frames[frame])

    plt.title(str(energy_bins[frame]) + "-" + str(energy_bins[frame + 1]) + " MeV")

    return frame


# Animate
anim = animation.FuncAnimation(fig, animate, frames=range(len(frames)), blit=False, repeat=False)

writer = animation.PillowWriter(fps=1)

anim.save("./agn_flare_energy_bins.gif", writer=writer)

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
