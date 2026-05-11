from astropy.io import fits
from astropy.table import QTable
import pandas


def read_fits_catalog(file_name, table_name):

    # Read catalog data and format as astropy QTable (allowing for units to be associated with each column, if necessary
    catalog = QTable.read(file_name, format='fits', hdu=1)

    # # Access 4FGL catalog - note, file originally called gll_psc_v35.fit
    # with fits.open(file_name) as catalog:
    #
    #     # get catalog data
    #     catalog_data = catalog[table_name].data
    #
    #     # get names of each table attributes (i.e. column names)
    #     column_names = catalog_data.columns.names
    #
    #     # get units of each column - there are empty sections
    #     column_units = catalog_data.columns.units
    #
    #     # print(type(catalog_data))



    return


# References
# Astropy Documentation - https://docs.astropy.org/en/stable/index.html
# Justification for Table not Dataframe - https://docs.astropy.org/en/latest/table/table_and_dataframes.html
