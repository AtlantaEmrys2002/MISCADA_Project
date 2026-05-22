from astropy.io import fits
from astropy.table import QTable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation


# 0.1 degree bins for RA, 0.5 degree bins for Dec
x = np.zeros((51, 51))


def count_plot(ras, decs):

    new_map, xedges, yedges = np.histogram2d(ras, decs, bins=(500, 500))

    ax = fig.add_subplot(111, title='Photon Count', aspect='equal', ylim=yedges[[0, -1]], xlim=xedges[[0, -1]])

    mesh = ax.pcolormesh(xedges, yedges, new_map, cmap=mpl.cm.hot)

    mesh.set_clim(0, 8)

    plt.xlabel('RA (deg)')
    plt.ylabel('Dec (deg')

    plt.show()


def photons_in_timeframe(ras, decs, t_1, t_2):

    mask = np.argwhere(np.logical_and(x > t_1, x < t_2))

    return ras[mask], decs[mask]


def animate(k):

    masked_ra, masked_dec = photons_in_timeframe(ras, decs, t_1=k * 2628000, t_2=(k * 2628000) + 2628000)

    new_map, xedges, yedges = np.histogram2d(masked_ra, masked_dec, bins=(500, 500))

    mesh = ax.pcolormesh(xedges, yedges, new_map, cmap=mpl.cm.hot)

    mesh.set_clim(0, 8)

    # artists.append(mesh)

    # return mesh


with fits.open("./test_data_5/roi_filtered_gti.fits", memmap=True) as hdul:
    print(hdul[1].columns)

    # print(min(hdul[1].data['TIME'])) # 239566097.30712715
    # print(max(hdul[1].data['TIME']))# 801035979.6212697

    n_timebins = (801035979.6212697 - 239566097.30712715) / 2628000

    ras = (hdul[1].data['RA'])
    ras = np.asarray([k % 10 for k in ras])

    decs = hdul[1].data['DEC']

    times = hdul[1].data['TIME']

    # Create plot

    fig = plt.figure()

    # mpl.pylab.plt.hold(True)

    masked_ra, masked_dec = photons_in_timeframe(ras, decs, t_1=239566097, t_2=239566097 + 2628000)

    new_map, xedges, yedges = np.histogram2d(masked_ra, masked_dec, bins=(500, 500))

    ax = fig.add_subplot(111, title='Photon Count', aspect='equal', ylim=yedges[[0, -1]], xlim=xedges[[0, -1]])

    mesh = ax.pcolormesh(xedges, yedges, new_map, cmap=mpl.cm.hot)

    mesh.set_clim(0, 8)

    anim = animation.FuncAnimation(fig, animate, frames=range(1, 213), blit=False)

    # mpl.pylab.plt.hold(False)

    # for k in range(239566097 + 2628000, 801035979, 2628000):
    #
    #     masked_ra, masked_dec = photons_in_timeframe(ras, decs, t_1=k, t_2=k+2628000)
    #
    #     # count_plot(masked_ra, masked_dec)
    #
    #     new_map, xedges, yedges = np.histogram2d(masked_ra, masked_dec, bins=(500, 500))
    #
    #     mesh = ax.pcolormesh(xedges, yedges, new_map, cmap=mpl.cm.hot)
    #
    #     mesh.set_clim(0, 8)
    #
    #     # artists.append(mesh)
    #
    #     return mesh

# ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=400)

plt.show()








# REFERENCES

# https://indico.cern.ch/event/860079/contributions/3731903/attachments/1979469/3295709/Tirana2020.pdf
