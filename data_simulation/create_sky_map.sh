#!/usr/bin/env bash -l

# N.B. Removed /Volumes/T7/data/weekly/photon/lat_photon_weekly_w539_p305_v001.fits as it caused problems

# Define files
#SOURCEMAP=/Volumes/T7/data/sky_map_creation_data/srcMaps.fit
#AGNS=./simulated_data/agns.xml
#PULSARS=./simulated_data/pulsars.xml
#
#EXPCUBE=/Volumes/T7/data/exposure_cube.fits
#BINEXPCUBE=/Volumes/T7/data/binned_exposure_cube.fits
#EXPCUBESPACECRAFT=/Volumes/T7/data/exposure_spacecraft.fits
#
#COUNTSMAP=/Volumes/T7/data/sky_map.fits

FERMIDATACUT=/Volumes/T7/data/sky_map_creation_data/fermi_filtered.fits
FERMIDATACUTGTI=/Volumes/T7/data/sky_map_creation_data/fermi_filtered_gti.fits
LTCUBE=/Volumes/T7/data/sky_map_creation_data/fermi_filtered_ltcube.fits
EXPMAP=/Volumes/T7/data/sky_map_creation_data/fermi_filtered_exposure_map.fits

# Using P8R3_ULTRACLEANVETO_V3 as I am doing an analysis that involves most of the sky - see Fermi recommendations
#IRF=

# NEED TO CHANGE FOR USER
source /Users/milli/miniconda3/bin/activate

conda init

# Enter Fermi environment
conda activate fermi

# Read all xml files into one
touch ./simulated_data/sources.xml

cat "$AGNS" > ./simulated_data/sources.xml
cat "$PULSARS" >> ./simulated_data/sources.xml

# Select relevant Fermi data
#ls /Volumes/T7/data/weekly/photon/*.fits > /Volumes/T7/data/weekly/photon/events.txt
#
#cat /Volumes/T7/data/weekly/photon/events.txt

# RAN BELOW LINE

# Select front and back events (ID8) with energy between 300 MeV and 200 GeV (recommended by ID8) and zenith cut is 100 degrees
# gtselect evclass=128 evtype=3 infile=@/Volumes/T7/data/weekly/photon/events.txt outfile=$FERMIDATACUT ra=INDEF dec=INDEF rad=INDEF tmin=INDEF tmax=INDEF emin=300 emax=200000 zmax=100

# Select corresponding good time intervals associated with this above selection - data filter is from ID8
# Do NOT perform zenith cut at this stage - as we are working on the whole sky.
# gtmktime scfile=@/Volumes/T7/data/lat_spacecraft_merged.fits evfile=$FERMIDATACUT outfile=$FERMIDATACUTGTI filter="(DATA_QUAL == 1) && (LAT_CONFIG == 1) && (IN_SAA != T)" roicut=no

# Construct livetime cube - setting zenith cut (based on recommendations of 100) here instead of gtmktime because
# a lot of data would be lost otherwise when performing the operation on the whole sky
# https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/data_preparation.html
# gtltcube evfile=$FERMIDATACUTGTI scfile=/Volumes/T7/data/lat_spacecraft_merged.fits outfile=$LTCUBE zmax=100 dcostheta=0.025 binsz=1

# Construct set of exposure maps for different energies - followed example 2 in documention for the whole sky
# Choice of IRF is based on recommendations in Fermi documentation - working on all-sky analysis
# gtexpcube2 evtype=3 binsz=1 infile=$LTCUBE cmap=none outfile=$EXPMAP irfs=P8R3_ULTRACLEANVETO_V3 nxpix=360 nypix=180 xref=0 yref=0 axisrot=0 coordsys=GAL proj=AIT emin=300 emax=200000 enumbins=6

# Convolve source maps with instrument response

# DID NOT RUN THE FUNCTION BELOW BUT DID RUN ALL ABOVE - NEED TO INCLUDE BACKGROUND MODELS FIRST BEFORE RUNNING BELOW FUNCTION

# gtsrcmaps scfile=/Volumes/T7/data/lat_spacecraft_merged.fits expcube=$EXPMAP cmap=none srcmdl=@./simulated_data/sources.xml







# INCLUDE THE BACKGROUND AND ISOTROPIC BACKGROUND HERE

# THEN gtsrcmaps

# THEN gtmodel







# Create binned exposure from real Fermi data

# THINK I ALREADY HAVE gtlcube file - JUST ADD IN CODE USED TO GENERATE IT FROM OTHER FILES - JUST FOLLOW TUTORIALS TO GET TO THIS POINT
# Followed example 2 in the documentation

#gtexpcube2

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
