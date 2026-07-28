#!/usr/bin/env bash -l

# THIS SCRIPT PREPARES ACTUAL FERMI DATA IN A FORMAT CONSISTENT WITH THE SOURCE DETECTION ALGORITHMS
# AND SAVES METADATA IN CSV FILES. THIS ALLOWS FOR NEW SOURCES TO BE FOUND

# Stages of data preparation
LIST=/Volumes/T7/data/weekly/photon/events.txt
RAWDATAFILES=/Volumes/T7/data/weekly/photon
SPACECRAFTFILE=/Volumes/T7/data/lat_spacecraft_merged.fits

FERMIDATACUT=/Volumes/T7/project_data/real_data/fermi_filtered.fits
FERMIDATACUTGTI=/Volumes/T7/project_data/real_data/fermi_filtered_gti.fits
FERMIBINNED=/Volumes/T7/project_data/real_data/fermi_filtered_gti_binned_for_evaluation.fits

source /Users/milli/miniconda3/bin/activate

conda init

# Enter Fermi environment
conda activate fermi

# Store the names of all relevant .fits files in events.txt
ls $RAWDATAFILES/*.fits > $LIST

# N.B. Only run gtselect and gtmktime if you have not done so before hand

# Select front and back events (ID8) with energy between 300 MeV and 200 GeV (recommended by ID8) and zenith cut is 100
# degrees (ID25)
# gtselect evclass=128 evtype=3 infile=@/Volumes/T7/data/weekly/photon/events.txt outfile=$FERMIDATACUT ra=INDEF dec=INDEF rad=INDEF tmin=INDEF tmax=INDEF emin=300 emax=200000 zmax=100

# Select corresponding good time intervals (GTI) - we do not apply a zenith cut here, as we will loose all our data
# - therefore, we apply the zenith cut when running the gtltcube function
# gtmktime scfile=$SPACECRAFTFILE evfile=$FERMIDATACUT outfile=$FERMIDATACUTGTI filter="(DATA_QUAL == 1) && (LAT_CONFIG == 1) && (IN_SAA != T)" roicut=no

# Reset all parameters

punlearn gtbin

# Bin data and create count maps - in HEALPIX format
gtbin evfile=$FERMIDATACUTGTI scfile=NONE outfile=$FERMIBINNED algorithm=HEALPIX coordsys=GAL ebinalg=LOG emin=300 emax=200000 enumbins=5 hpx_ordering_scheme=RING hpx_order=8

conda deactivate
