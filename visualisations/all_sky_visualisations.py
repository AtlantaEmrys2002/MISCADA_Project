from astropy.io import fits
from astropy.table import QTable
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt

# Look for all globular clusters in 4FGL (baseline)


catalog = QTable.read('/Volumes/T7/data/catalog/4FGL_DR4.fit', format='fits', hdu=1)

# Clean CLASS1 column (i.e. remove empty spaces and then ensure letter case is all the same - lower)
catalog['CLASS1'] = np.array([str(k).strip().lower() for k in catalog['CLASS1']])

# Select all globular clusters
globular_clusters = catalog[catalog['CLASS1'] == 'glc']

# Find all positions of globular clusters
globular_clusters = globular_clusters['Source_Name', 'RAJ2000', 'DEJ2000', 'GLON', 'GLAT']

# Format coordinates of each globular cluster into format compatible with astropy
coords = list(zip(globular_clusters['RAJ2000'].value, globular_clusters['DEJ2000'].value))

sky_coords = []

for k in coords:

    # Convert tuples to astropy SkyCoords
    sky_coords.append(SkyCoord(ra=k[0]*u.degree, dec=k[1]*u.degree, frame='icrs'))

# Convert coordinates to radians - different methods for RA and DEC
ra_rad = [k.ra.wrap_at(180 * u.deg).radian for k in sky_coords]
dec_rad = [k.dec.radian for k in sky_coords]

# Plot coordinates
fig, ax = plt.subplots(figsize=(8, 4.2), subplot_kw=dict(projection='aitoff'))
plt.title('Distribution of Gamma-Ray Emitting Globular Clusters', pad=20)
ax.grid(True)
ax.scatter(ra_rad, dec_rad, marker='o', s=3)
fig.subplots_adjust(top=0.95, bottom=0.0)
plt.show()

# N.B. if this method for selecting globular clusters does not work then try the extended sources ROI_num

# REFERENCES

# Astropy Documentation - https://docs.astropy.org/en/stable/coordinates/index.html
# Matplotlib Documentation - 
