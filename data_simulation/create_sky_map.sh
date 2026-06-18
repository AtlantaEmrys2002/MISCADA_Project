#!/usr/bin/env bash -l

# Define files
SOURCEMAP=/Volumes/T7/data/sky_map_creation_data/srcMaps.fit
AGNS=./simulated_data/agns.xml
PULSARS=./simulated_data/pulsars.xml

EXPCUBE=/Volumes/T7/data/exposure_cube.fits
BINEXPCUBE=/Volumes/T7/data/binned_exposure_cube.fits
EXPCUBESPACECRAFT=/Volumes/T7/data/exposure_spacecraft.fits

COUNTSMAP=/Volumes/T7/data/sky_map.fits

# Using P8R3_ULTRACLEANVETO_V3 as I am doing an analysis that involves most of the sky - see Fermi recommendations
IRF=


# NEED TO CHANGE FOR USER
source /Users/milli/miniconda3/bin/activate

conda init

# Enter Fermi environment
conda activate fermi

# Read all xml files into one
touch ./simulated_data/sources.xml

cat "$AGNS" > ./simulated_data/sources.xml
cat "$PULSARS" >> ./simulated_data/sources.xml

# Create binned exposure from real Fermi data

# THINK I ALREADY HAVE gtlcube file - JUST ADD IN CODE USED TO GENERATE IT FROM OTHER FILES - JUST FOLLOW TUTORIALS TO GET TO THIS POINT
# Followed example 2 in the documentation

gtexpcube2

# Convolve source models with IRF - also generates source maps for point sources
# gtsrcmaps scfile=$EXPCUBESPACECRAFT expcube=$EXPCUBE cmap=none srcmdl=sources.xml bexpmap=$BINEXPCUBE outfile=$SOURCEMAP irfs=P8R3_ULTRACLEANVETO_V3 ptsrc=yes

# Create model counts map based on fit parameters (here are fit parameters are randomly generated)
# CHECK IF I SHOULD CHANGE evtype to 3 (AS THIS IS LAST STEP CAN I Just SET TO EV 3 EVEN THOUGH IRF HAS evtype 1024
# gtmodel srcmaps=$SOURCEMAP srcmdl=./simulated_data/sources.xml outfile=$COUNTSMAP irfs=P8R3_ULTRACLEANVETO_V3 expcube=EXPCUBE bexpmap=BINEXPCUBE

# Deactivate environment
conda deactivate

# REFERENCES

# Conda Init Error - https://stackoverflow.com/questions/77901825/unable-to-activate-environment-conda-prompted-to-run-
# conda-init-before-cond
# IRF Justification - https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/lat_data_selection.html
