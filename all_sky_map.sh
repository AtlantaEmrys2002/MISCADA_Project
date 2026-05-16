# GENERATE ALL-SKY MAP

# Set up fermitools environment
conda activate fermi

# List all weekly photon files
mkdir ./data
ls ./weekly/photon/lat_photon_weekly* > ./data/filelist.txt

# Delete file which will not allow for combination - no time intervals available
sed -i.bak '/lat_photon_weekly_w539_p305_v001.fits/d' ./data/filelist.txt
rm ./weekly/photon/lat_photon_weekly_w539_p305_v001.fits

punlearn gtselect

# Combine all fits files into one FITS file - not filtering at this point
gtselect evclass=INDEF evtype=INDEF
  @./data/filelist.txt
  ./data/lat_alldata.fits
  0
  0
  180
  INDEF
  INDEF
  30
  1000000
  180

# Clear gtselect defaults
punlearn gtselect

# Filter photon source types AND remove Earth limb using recommended zenith cut of 90 degrees

# Take the SOURCE class (evclass=128) photons and front and back converting events (evtype=3)
gtselect evclass=128 evtype=3
  ./data/lat_alldata.fits
  ./data/lat_source_zenith_90_100_to_1000000.fits
  0
  0
  180
  INDEF
  INDEF
  100
  1000000
  90

# Correct exposure and GTIs
# N.B. do not correct for the zenith cut as otherwise all data will vanish (this
# time selection method is recommended here)
gtmktime
  ./mission/spacecraft/lat_spacecraft_merged.fits
  (DATA_QUAL>0) && (LAT_CONFIG==1)
  no
  ./data/lat_source_zenith_90_100_to_1000000.fits
  ./data/lat_source_zenith_90_100_to_1000000_gti.fits

# Spatially bin data into single energy bin - N.B. this is what you might change later on
gtbin
  CCUBE
  ./data/lat_source_zenith_90_100_to_1000000_gti.fits
  ./data/lat_source_zenith_90_100_to_1000000_ccube.fits
  ./mission/spacecraft/lat_spacecraft_merged.fits
  3600
  1800





# Leave fermitools environment
conda deactivate

# REFERENCES
# Recommended Parameters - https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/lat_data_selection.html
