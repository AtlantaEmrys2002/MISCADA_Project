from astropy.table import QTable, MaskedColumn, Column, join
# from astropy.utils.masked import Masked
from astropy import units as u
import numpy as np
# import numpy.ma as ma
# import pandas as pd
import os

# RANDOM SEED
np.random.seed(42)

# FUNCTIONS FOR DERIVED FEATURES


def sed_flux(photon_flux, alpha, low_energy, high_energy):
    r"""

    Calculates the energy flux at the geometric mean of the energy band.

    Parameters
    ----------
    photon_flux :
    alpha :
    low_energy :
    high_energy :

    Returns
    -------

    """

    # Ratio
    r = low_energy / high_energy

    # Conversion factor
    GeV2erg = 0.00160217657

    # Energy flux at geometric mean of the energy band (i.e. think of SED plot)
    sed_point = (GeV2erg * low_energy) * (alpha - 1) * photon_flux * pow(r, (alpha / 2) - 1) / (1 - pow(r, alpha - 1))

    return sed_point


def correct_skew(column_data, non_zero_boolean):

    logarithm = np.log(column_data)

    masked_logarithm = np.where(non_zero_boolean, logarithm, np.zeros_like(column_data))

    return masked_logarithm


# READ DATA FILES

# Access 4FGL catalog - note, file originally called gll_psc_v35.fit
# Read catalog data and format as astropy QTable (allowing for units to be associated with each column, if necessary
catalog = QTable.read('/Volumes/T7/data/catalog/4FGL_DR4.fit', format='fits', hdu=1)

# Access 3PC Pulsar catalog - used this instead of the fermicatsR version, as this is the official catalog of pulsars
# presented by the LAT collaboration (was not available at time of ID4 paper publishing) and is up to date with current
# detections

# Take parameters present in this pulsar data format - https://cran.r-project.org/web/packages/fermicatsR/fermicatsR.pdf
# - (no need for codes - they vary between two datasets and are not used - or edot, rajd, decjd which are not used)

pulsar_catalog = QTable.read('/Volumes/T7/data/catalog/3PC_Catalog.fits', format='fits', hdu=1)

pulsar_catalog = pulsar_catalog['PSRJ', 'P0']

pulsar_catalog['PSRJ'].name = 'PSR_coords'

# DATA CLEAN

relevant_columns = ['Source_Name', 'RAJ2000', 'DEJ2000', 'GLON', 'GLAT', 'PL_Index', 'PLEC_IndexS', 'LP_Index',
                    'Variability_Index', 'Conf_68_SemiMajor', 'Conf_68_SemiMinor', 'Conf_68_PosAng', 'SpectrumType',
                    'Conf_95_SemiMajor', 'Conf_95_SemiMinor', 'Conf_95_PosAng', 'Signif_Avg', 'Pivot_Energy',
                    'LP_Flux_Density', 'PLEC_Flux_Density', 'PL_Flux_Density', 'Unc_PL_Flux_Density',
                    'Unc_LP_Flux_Density', 'Unc_PLEC_Flux_Density', 'Flux1000', 'Unc_Flux1000', 'Energy_Flux100',
                    'Unc_Energy_Flux100', 'LP_SigCurv', 'PLEC_SigCurv', 'Flux_Band', 'CLASS1', 'ASSOC1']

# Select relevant columns
catalog = catalog[relevant_columns]

# Restructure 4FGL as if it were 3FGL data

# Determine which rows have log parabolic, PL super exponential cut-off, or power law spectral shapes.
# N.B. char will be deprecated and replaced with strings
spectrum_types = np.char.strip(catalog['SpectrumType'].data.astype(str))

# Calculate spectral index column - N.B. 3FGL had Spectral_Index parameter which has since been replaced by more
# specific PL_Index, PLEC_Index, and LP_Index. Combine these three columns into Spectral_Index column
pl_index = catalog['PL_Index'].data * (spectrum_types == 'PowerLaw')
lp_index = catalog['LP_Index'].data * (spectrum_types == 'LogParabola')
plec_index = catalog['PLEC_IndexS'].data * (spectrum_types == 'PLSuperExpCutoff')

spectral_index = MaskedColumn(data=pl_index + lp_index + plec_index, name='Spectral_Index', format='{:8.4f}')
catalog.add_column(spectral_index)

# Remove specific spectral index columns (i.e. PL_Index, PLEC_IndexS, LP_Index)
catalog.remove_columns(['PL_Index', 'PLEC_IndexS', 'LP_Index'])

# Calculate flux density column - N.B. 3FGL had Flux_Density parameter which has since been replaced by more specific
# PL_Flux_Density, PLEC_Flux_Density, and LP_Flux_Density. Combine these three columns into Flux_Density column.
pl_flux_density = catalog['PL_Flux_Density'].data * (spectrum_types == 'PowerLaw')
lp_flux_density = catalog['LP_Flux_Density'].data * (spectrum_types == 'LogParabola')
plec_flux_density = catalog['PLEC_Flux_Density'].data * (spectrum_types == 'PLSuperExpCutoff')

flux_density = MaskedColumn(data=pl_flux_density + lp_flux_density + plec_flux_density, name='Flux_Density',
                            format='{:10.4e}', unit=u.ph / (u.cm * u.cm * u.MeV * u.s))
catalog.add_column(flux_density)

# Remove specific flux density columns
catalog.remove_columns(['PL_Flux_Density', 'PLEC_Flux_Density', 'LP_Flux_Density'])

# Calculate uncertainty in flux density column - N.B. 3FGL had Unc_Flux_Density parameter which has since been replaced
# by more specific Unc_PL_Flux_Density, Unc_PLEC_Flux_Density, and Unc_LP_Flux_Density. Combine these three columns into
# Unc_Flux_Density column.

unc_pl_flux_density = catalog['Unc_PL_Flux_Density'].data * (spectrum_types == 'PowerLaw')
unc_lp_flux_density = catalog['Unc_LP_Flux_Density'].data * (spectrum_types == 'LogParabola')
unc_plec_flux_density = catalog['Unc_PLEC_Flux_Density'].data * (spectrum_types == 'PLSuperExpCutoff')

unc_flux_density = MaskedColumn(data=unc_pl_flux_density + unc_lp_flux_density + unc_plec_flux_density,
                                name='Unc_Flux_Density', format='{:10.4e}', unit=u.ph / (u.cm * u.cm * u.MeV * u.s))
catalog.add_column(unc_flux_density)

# Remove specific flux density columns
catalog.remove_columns(['Unc_PL_Flux_Density', 'Unc_PLEC_Flux_Density', 'Unc_LP_Flux_Density'])

# Calculate Signif_Curv (merging separate columns for LP_SigCurv and PLEC_SigCurv to form 3FGL column) -
# see https://heasarc.gsfc.nasa.gov/w3browse/fermi/fermi3fgl.html for when to use each column.

pulsars = np.logical_or(catalog['CLASS1'] == 'PSR', catalog['CLASS1'] == 'psr')
sig_curv = Column(data=(np.logical_not(pulsars) * catalog['LP_SigCurv']) + (pulsars * catalog['PLEC_SigCurv']),
                  name='Signif_Curv', format='{:8.3f}')
catalog.add_column(sig_curv)

# Remove specific significant curve columns
catalog.remove_columns(['LP_SigCurv', 'PLEC_SigCurv'])

# Separate each energy band for integral photon flux and rename (calculating 10 - 100 GeV band using 'Flux1000')
transpose_flux_bands = catalog['Flux_Band'].value.T

catalog['Flux100_300'] = transpose_flux_bands[1] * u.ph / (u.cm * u.cm * u.s)
catalog['Flux300_1000'] = transpose_flux_bands[2] * u.ph / (u.cm * u.cm * u.s)
catalog['Flux1000_3000'] = transpose_flux_bands[3] * u.ph / (u.cm * u.cm * u.s)
catalog['Flux3000_10000'] = transpose_flux_bands[4] * u.ph / (u.cm * u.cm * u.s)

# Flux1000 is integral photon flux from 1 to 100 GeV - we want 10 to 100 GeV. Subtract integral photon flux from 1 to 10
# GeV
catalog['Flux10000_100000'] = catalog['Flux1000'] - catalog['Flux1000_3000'] - catalog['Flux3000_10000']

# Remove flux band columns
catalog.remove_columns(['Flux1000', 'Flux_Band'])

# Remove spectrum type column
catalog.remove_columns(['SpectrumType'])

# Format data columns correctly

# Rename columns
catalog.rename_columns(['Signif_Avg', 'Flux_Density', 'RAJ2000', 'DEJ2000'], ['Signif', 'Flux', 'RA', 'DEC'])

# Keep only the unique parts of each source name (i.e. we know that all sources are in the 4FGL)
catalog['Source_Name'] = [k[6:18] for k in catalog['Source_Name']]

# Round spatial coordinates to 2 decimal places
catalog['RA'] = catalog['RA'].round(2)
catalog['DEC'] = catalog['DEC'].round(2)
catalog['GLON'] = catalog['GLON'].round(2)
catalog['GLAT'] = catalog['GLAT'].round(2)

# Round signif avg to 3 decimal places
catalog['Signif'] = catalog['Signif'].round(3)

# NEW ATTRIBUTES

# Spectral Energy Distribution Points

catalog['SED100_300'] = (sed_flux(catalog['Flux100_300'], catalog['Spectral_Index'], 0.1, 0.3)
                         * u.ph / (u.cm * u.cm * u.s))
catalog['SED300_1000'] = (sed_flux(catalog['Flux300_1000'], catalog['Spectral_Index'], 0.3, 1.0)
                          * u.ph / (u.cm * u.cm * u.s))
catalog['SED1000_3000'] = (sed_flux(catalog['Flux1000_3000'], catalog['Spectral_Index'], 1.0, 3.0)
                           * u.ph / (u.cm * u.cm * u.s))
catalog['SED3000_10000'] = (sed_flux(catalog['Flux3000_10000'], catalog['Spectral_Index'], 3.0, 10.0)
                            * u.ph / (u.cm * u.cm * u.s))
catalog['SED10000_100000'] = (sed_flux(catalog['Flux10000_100000'], catalog['Spectral_Index'], 10.0, 100.0)
                              * u.ph / (u.cm * u.cm * u.s))

# Hardness Ratios
catalog['hr12'] = (catalog['SED300_1000'] - catalog['SED100_300']) / (catalog['SED300_1000'] + catalog['SED100_300'])
catalog['hr23'] = (catalog['SED1000_3000'] - catalog['SED300_1000']) / (catalog['SED1000_3000'] +
                                                                        catalog['SED300_1000'])
catalog['hr34'] = (catalog['SED3000_10000'] - catalog['SED1000_3000']) / (catalog['SED3000_10000'] +
                                                                          catalog['SED1000_3000'])
catalog['hr45'] = (catalog['SED10000_100000'] - catalog['SED3000_10000']) / (catalog['SED10000_100000'] +
                                                                             catalog['SED3000_10000'])

# Remove any empty spaces from CLASS1
catalog['CLASS1'][:] = [k.strip() for k in catalog['CLASS1']]

# Drop correlated and unused variables

catalog.remove_columns(['Unc_Flux1000', 'Energy_Flux100', 'Conf_95_SemiMajor', 'Conf_95_SemiMinor', 'Flux100_300',
                        'Conf_68_SemiMajor', 'Conf_68_SemiMinor', 'Conf_68_PosAng', 'SED100_300', 'SED300_1000',
                        'SED1000_3000', 'SED3000_10000', 'SED10000_100000', 'Flux300_1000', 'Flux1000_3000',
                        'Flux3000_10000'])

# Take logarithm for highly skewed distributions to prevent this from effecting classifiers
not_close_to_zero = np.logical_not(np.isclose(catalog['Signif_Curv'], np.zeros(np.shape(catalog['Signif_Curv']))))

catalog['Variability_Index'] = correct_skew(catalog['Variability_Index'], not_close_to_zero)
catalog['Pivot_Energy'] = correct_skew(catalog['Pivot_Energy'].data, not_close_to_zero)
catalog['Flux'] = correct_skew(catalog['Flux'].data, not_close_to_zero)
catalog['Unc_Flux_Density'] = correct_skew(catalog['Unc_Flux_Density'].data, not_close_to_zero)
catalog['Unc_Energy_Flux100'] = correct_skew(catalog['Unc_Energy_Flux100'].data, not_close_to_zero)
catalog['Flux10000_100000'] = correct_skew(catalog['Flux10000_100000'].data, not_close_to_zero)
catalog['Signif_Curv'] = correct_skew(catalog['Signif_Curv'], not_close_to_zero)

catalog.remove_columns(['Pivot_Energy', 'Unc_Flux_Density', 'Flux10000_100000'])

# Reformat flux
catalog['Flux'] = (np.e**catalog['Flux'])
catalog['Flux'] = catalog['Flux'].round(3)

# Create column indicating whether source is pulsar type or AGN type
pulsars = np.logical_or(catalog['CLASS1'] == 'psr', catalog['CLASS1'] == 'PSR')

agn_types = ['BCU', 'bcu', 'BLL', 'bll', 'FSRQ', 'fsrq', 'rdg', 'RDG', 'nlsy1', 'NLSY1', 'agn', 'AGN', 'ssrq', 'SSRQ',
             'sey', 'SEY']

agns = np.zeros(catalog['CLASS1'].shape)

for k in agn_types:
    agns += (catalog['CLASS1'] == k)

agns = agns > 0

catalog['agnness'] = ['AGN' if k > 0 else 'Non-AGN' for k in agns]
catalog['pulsarness'] = ['Pulsar' if k > 0 else 'Non-Pulsar' for k in pulsars]

# CREATE TWO NEW DATASETS - AGN/PULSARS AND PULSARS

# DATASET 1 - AGN vs PULSARS

mask = (catalog['pulsarness'] == 'Pulsar') | (catalog['agnness'] == 'AGN')

# Convert to pandas dataframe

# Select only rows that indicate source is either an AGN or pulsar
agn_and_pulsars = catalog[mask].to_pandas()

# Drop rows that have missing values (only 4 for the current version of 4FGL)
indices = agn_and_pulsars[agn_and_pulsars.isna().any(axis=1)].index.tolist()
agn_and_pulsars.drop(indices, inplace=True)

# Drop more unnecessary columns (agnness can be removed as now 'Non-Pulsar' indicates AGN)
agn_and_pulsars.drop(columns=["CLASS1", "ASSOC1", "Source_Name", "GLAT", "Conf_95_PosAng", "agnness"], inplace=True)

# Save dataset
os.makedirs("./datasets", exist_ok=True)
agn_and_pulsars.to_csv("./datasets/agn_and_pulsars.csv", index=False)

# DATASET 2 - PULSARS

pulsars = catalog[catalog['pulsarness'] == 'Pulsar']
pulsars.remove_column('pulsarness')

# Convert type of ASSOC1 column and remove the "PSR J" at the front and strip the empty space at the end
# Rename ASSOC1 as PSR_coors
pulsars['ASSOC1'].name = 'ASSOC1_Bytes'
new_assoc_1 = Column(data=[k.decode('utf-8')[4:].strip() for k in pulsars['ASSOC1_Bytes'].data], name='PSR_coords')
pulsars.add_column(new_assoc_1)
pulsars.remove_column('ASSOC1_Bytes')

# Convert barycentric period to milliseconds (from seconds)
pulsar_catalog['P0'] *= 1000

# Join so can determine if pulsars YNG or MSP based on period P0
pulsars = join(pulsars, pulsar_catalog, keys='PSR_coords', join_type='left')

# We use the definition that a millisecond pulsar has <~10 ms period (unlike the 3PC catalog which defines a MSP as
# having a pulsar as < 30 ms) to ensure consistency with ID4.
pulsars['pulsarness'] = ['YNG' if k == 1 else 'MSP' for k in pulsars['P0'] > 10]

# Remove redundant columns
pulsars.remove_columns(['P0', 'CLASS1', 'Source_Name', 'PSR_coords', 'agnness'])

# Convert pulsars to pandas dataframe (as before)
pulsars = pulsars.to_pandas()

# Save dataset
os.makedirs("./datasets", exist_ok=True)
pulsars.to_csv("./datasets/pulsars.csv", index=False)

# REFERENCES

# Key Sources

# Astropy Documentation - https://docs.astropy.org/en/stable/index.html
# Energy Band Range Definitions - https://academic.oup.com/mnras/article/527/2/1794/7277574
# Justification for QTable not Dataframe - https://docs.astropy.org/en/latest/table/table_and_dataframes.html
# Millisecond Pulsar Definitions - https://fermi.gsfc.nasa.gov/science/mtgs/symposia/eleventh/program/posters/Ray_3PC_
# Poster.pdf
# Numpy Documentation - https://numpy.org/doc/stable/index.html
# Pandas Documentation - https://pandas.pydata.org/docs/index.html
# R Documentation - https://www.rdocumentation.org/
# Saz Parkinson et al. Paper - https://arxiv.org/abs/1602.00385
# Saz Parkinson et al.'s R Scripts - https://scipp-legacy.pbsci.ucsc.edu/~pablo/pulsarness.html
# Spectral Energy Distribution - https://en.wikipedia.org/wiki/Spectral_energy_distribution
# 4FGL Catalog Dataset - https://fermi.gsfc.nasa.gov/ssc/data/access/lat/14yr_catalog/
# 4FGL Paper - https://arxiv.org/pdf/1902.10045

# Bug Fixing
# Bytestrings to Strings - https://stackoverflow.com/questions/23618218/numpy-bytes-to-plain-string
# numpy char vs string - https://github.com/numpy/numpy/issues/28559
# Numpy Random Seed -
# https://stackoverflow.com/questions/31057197/should-i-use-random-seed-or-numpy-random-seed-to-control-random-number-gener
# OS Directories - https://www.geeksforgeeks.org/python/python-os-makedirs-method/
# Rows with Empty Values (Pandas) -
# https://stackoverflow.com/questions/43424199/display-rows-with-one-or-more-nan-values-in-pandas-dataframe
# Saving without Index - https://stackoverflow.com/questions/20845213/how-to-avoid-pandas-creating-an-index-in-a-saved-
# csv
