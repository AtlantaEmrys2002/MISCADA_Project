#!/usr/bin/env bash -l

# File names
#EVENTS=./agn_flare/L2605220919104D7EA5AE90_PH00.fits
#OUTFILE=./agn_flare/agn.fits
#OUTFILEGTI=./agn_flare/agn_gti.fits
#SPACECRAFTFILE=./agn_flare/L2605220919104D7EA5AE90_SC00.fits

EVENTS=@./agn_flare_2/events.txt
OUTFILE=./agn_flare_2/agn.fits
OUTFILEGTI=./agn_flare_2/agn_gti.fits
SPACECRAFTFILE=./agn_flare_2/L2605221114164D7EA5AE60_SC00.fits

conda init

conda activate fermi

## Select source class and front and back events
#gtselect evclass=128 evtype=3 infile=$EVENTS outfile=$OUTFILE ra=330.68 dec=42 rad=5 tmin=749433605 tmax=752025605 \
#  emin=50 emax=500000 zmax=180
#
## Select correct good time intervals and correct exposure
#gtmktime scfile=$SPACECRAFTFILE filter="(DATA_QUAL>0)&&(LAT_CONFIG==1)" roicut=no evfile=$OUTFILE outfile=$OUTFILEGTI

ls ./agn_flare_2/*_PH* > ./agn_flare_2/events.txt

# Select source class and front and back events - TRIED ZENitH CUT OF 90 TO GET RID OF EARTH LIMB (zmax = 90 instead of 180)
gtselect evclass=128 evtype=3 infile=$EVENTS outfile=$OUTFILE ra=330.68 dec=42 rad=10 tmin=72576000 tmax=757209605 \
  emin=50 emax=500000 zmax=90

# Select correct good time intervals and correct exposure
gtmktime scfile=$SPACECRAFTFILE filter="(DATA_QUAL>0)&&(LAT_CONFIG==1)" roicut=no evfile=$OUTFILE outfile=$OUTFILEGTI


conda deactivate
