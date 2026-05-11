from astropy.io import fits
import reading_fits_catalog_data as read_fits

# Access 4FGL catalog - note, file originally called gll_psc_v35.fit
read_fits.read_fits_catalog(file_name='/Volumes/T7/data/catalog/4FGL_DR4.fit', table_name='LAT_Point_Source_Catalog')





# References
# Astropy Documentation - https://docs.astropy.org/en/stable/index.html
# 4FGL Catalog Dataset - https://fermi.gsfc.nasa.gov/ssc/data/access/lat/14yr_catalog/
