# Activate fermitools

# Filenames
EVENTS=/Volumes/T7/fermi_data_backup/data/lat_alldata.fits
OUTFILE=./gc_1.fits
OUTFILEGTI=./gc_1_gti.fits
SPACECRAFTFILE=/Volumes/T7/data/lat_spacecraft_merged.fits

conda init

conda activate fermi

# Select source class and front and back events
#gtselect evclass=128 evtype=3 infile=$EVENTS outfile=$OUTFILE ra=6.0082 dec=-72.0788 rad = 20 tmin=INDEF tmax=INDEF \
#  emin=30 emax=300000 zmax=180

# Select correct good time intervals and correct exposure
gtmktime scfile=$SPACECRAFTFILE filter="(DATA_QUAL>0)&&(LAT_CONFIG==1)" roicut=no evfile=$OUTFILE outfile=$OUTFILEGTI

conda deactivate