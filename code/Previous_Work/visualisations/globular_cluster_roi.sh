#!/usr/bin/env bash -l

# Activate fermitools

# Filenames
EVENTS=@./test_data/events.txt
OUTFILE=./test_data/globular_cluster_1.fits
SPACECRAFTFILE=./test_data/L2605210835054D7EA5AE06_SC00.fits
OUTFILEGTI=./globular_cluster_gti.fits

conda init

conda activate fermi

# Combine photon files
ls ./test_data/*_PH* > ./test_data/events.txt

# Select source class and front and back events
gtselect evclass=128 evtype=3 infile=$EVENTS outfile=$OUTFILE ra=6.0082 dec=-72.0788 rad = 20 tmin=INDEF tmax=INDEF \
  emin=100 emax=300000 zmax=180

# Select correct good time intervals and correct exposure
gtmktime scfile=$SPACECRAFTFILE filter="(DATA_QUAL>0)&&(LAT_CONFIG==1)" roicut=no evfile=$OUTFILE outfile=$OUTFILEGTI





conda deactivate