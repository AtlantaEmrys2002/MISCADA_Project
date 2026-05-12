from astropy.table import QTable

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
    sed_point = (GeV2erg * low_energy) * (alpha - 1) * photon_flux * pow(r, (alpha/2) - 1) / (1 - pow(r, alpha - 1))

    return sed_point


# READ DATA FILE

# Access 4FGL catalog - note, file originally called gll_psc_v35.fit
# Read catalog data and format as astropy QTable (allowing for units to be associated with each column, if necessary
catalog = QTable.read('/Volumes/T7/data/catalog/4FGL_DR4.fit', format='fits', hdu=1)


# DATA CLEAN

# USE Spectrum Type and then remove that column

# N.B. 3FGL had Spectral_Index parameter which has since been replaced by more specific PL_Index, PLEC_Index,
# LP_Index. Therefore, we take those columns here and will combine into Spectral_Index (for consistency in reproducing
# paper). Think that PLEC_Index_S is the correct choice for replacing Spectral_Index (CHECK NUMBERS)
# Similarly, Flux_Density is not specific to spectral shape and has been split into columns (WILL COMBINE LATER TOO)
# AND Unc_Flux_Density (changed to Unc_SHAPE_Flux_Density I THINK - CHECK - COMBINE TOO)
# AND SignifCuve for LP and PL (Merge spectral types)
# FLUX FOR EACH BAND ARE STORED IN VETOR COLUMNS NOW - PICK OUT BAND 2, 3, 4, 5 and rename to Flux100_300, etc. (p17 in 4FGL paper)

# DON'T FORGET UNITS IN EACH COLUMN IF NEEDED

# Select relevant columns

catalog = catalog['Source_Name', 'RAJ2000', 'DEJ2000', 'GLON', 'GLAT', 'PL_Index', 'PLEC_IndexS', 'LP_Index',
                  'Variability_Index', 'Conf_68_SemiMajor', 'Conf_68_SemiMinor', 'Conf_68_PosAng', 'SpectrumType',
                  'Conf_95_SemiMajor', 'Conf_95_SemiMinor', 'Conf_95_PosAng', 'Signif_Avg', 'Pivot_Energy',
                  'LP_Flux_Density', 'PLEC_Flux_Density', 'PL_Flux_Density', 'Unc_PL_Flux_Density',
                  'Unc_LP_Flux_Density', 'Unc_PLEC_Flux_Density', 'Flux1000', 'Unc_Flux1000', 'Energy_Flux100',
                  'Unc_Energy_Flux100', 'LP_SigCurv', 'PLEC_SigCurv', 'Flux_Band', 'CLASS1', 'ASSOC1']

print(catalog.info)








# Select all sources that are considerec AGN and pulsars
# sources =


# print(catalog.info)


# REFERENCES

# Astropy Documentation - https://docs.astropy.org/en/stable/index.html
# Justification for QTable not Dataframe - https://docs.astropy.org/en/latest/table/table_and_dataframes.html
# Saz Parkinson et al. Paper - https://arxiv.org/abs/1602.00385
# Saz Parkinson et al.'s R Scripts - https://scipp-legacy.pbsci.ucsc.edu/~pablo/pulsarness.html
# Spectral Energy Distribution - https://en.wikipedia.org/wiki/Spectral_energy_distribution
# 4FGL Catalog Dataset - https://fermi.gsfc.nasa.gov/ssc/data/access/lat/14yr_catalog/
# 4FGL Paper - https://arxiv.org/pdf/1902.10045
