#!/usr/bin/env bash -l

# N.B. Removed /Volumes/T7/data/weekly/photon/lat_photon_weekly_w539_p305_v001.fits as it caused problems

# Define files
#SOURCEMAP=/Volumes/T7/data/sky_map_creation_data/srcMaps.fit
AGNS=./simulated_data/agns.xml
PULSARS=./simulated_data/pulsars.xml
BACKGROUND=./simulated_data/background.xml
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
SRCMAP=/Volumes/T7/data/sky_map_creation_data/fermi_filtered_convolved.fits
ASIMOV=/Volumes/T7/data/sky_map_creation_data/fermi_asimov.fits

FERMIBACKGROUND=/Volumes/T7/data/background_models/gll_iem_v07.fits
FERMIBACKGROUNDFILTERED=/Volumes/T7/data/sky_map_creation_data/fermi_background_filtered.fits
FERMIBACKGROUNDFILTEREDGTI=/Volumes/T7/data/sky_map_creation_data/fermi_background_filtered_gti.fits
FERMIBACKGROUNDCMAP=/Volumes/T7/data/sky_map_creation_data/fermi_background_filtered_gti_cmap.fits

# NEED TO CHANGE FOR USER
source /Users/milli/miniconda3/bin/activate

conda init

# Enter Fermi environment
conda activate fermi

# Read all xml files into one
touch ./simulated_data/sources.xml

#cat "$BACKGROUND" > ./simulated_data/sources.xml
#cat "$AGNS" >> ./simulated_data/sources.xml
#cat "$PULSARS" >> ./simulated_data/sources.xml

SOURCEXML=./simulated_data/sources.xml

# RAN BELOW LINE

# Select front and back events (ID8) with energy between 300 MeV and 200 GeV (recommended by ID8) and zenith cut is 100 degrees
gtselect evclass=128 evtype=3 infile=@/Volumes/T7/data/weekly/photon/events.txt outfile=$FERMIDATACUT ra=INDEF dec=INDEF rad=INDEF tmin=INDEF tmax=INDEF emin=300 emax=200000 zmax=100

# Select corresponding good time intervals associated with this above selection - data filter is from ID8
# Do NOT perform zenith cut at this stage - as we are working on the whole sky.
gtmktime scfile=@/Volumes/T7/data/lat_spacecraft_merged.fits evfile=$FERMIDATACUT outfile=$FERMIDATACUTGTI filter="(DATA_QUAL == 1) && (LAT_CONFIG == 1) && (IN_SAA != T)" roicut=no

# Construct livetime cube - setting zenith cut (based on recommendations of 100) here instead of gtmktime because
# a lot of data would be lost otherwise when performing the operation on the whole sky
# https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/data_preparation.html


# FROM BELOW - NEED TO CHANGE ENERGY BINS SO 5 (OTHERWISE AUTOSOURCEID WILL NOT WORK!!!!!)
# VERY VERY IMPortANT















gtltcube evfile=$FERMIDATACUTGTI scfile=/Volumes/T7/data/lat_spacecraft_merged.fits outfile=$LTCUBE zmax=100 dcostheta=0.025 binsz=1

# Construct set of binned exposure maps for different energies - followed example 2 in documention for the whole sky
# Choice of IRF is based on recommendations in Fermi documentation - working on all-sky analysis
gtexpcube2 evtype=3 binsz=1 infile=$LTCUBE cmap=none outfile=$EXPMAP irfs=P8R3_ULTRACLEANVETO_V3 nxpix=360 nypix=180 xref=0 yref=0 axisrot=0 coordsys=GAL proj=AIT emin=300 emax=200000 enumbins=6

# Create count map involving Fermi data (only used as count map and my sources will be fit - CHECK THIS WITH ANTHONY)
# Picked order of 8 (256). Need for count map
gtbin evfile=$FERMIDATACUTGTI scfile=NONE outfile=$FERMIBACKGROUNDCMAP algorithm=HEALPIX ebinalg=LOG emin=300 emax=200000 enumbins=6 nxpix=360 nypix=180 binsz=1 hpx_ordering_scheme=RING hpx_order=8 xref=0 axisrot=0 proj=AIT yref=0

# Convolve source maps with instrument response
gtsrcmaps cmap=$FERMIBACKGROUNDCMAP scfile=/Volumes/T7/data/lat_spacecraft_merged.fits srcmdl=$SOURCEXML expcube=$LTCUBE bexpmap=$EXPMAP ptsrc=yes outfile=$SRCMAP irfs=P8R3_ULTRACLEANVETO_V3

gtmodel srcmaps=$SRCMAP srcmdl=$SOURCEXML outfile=$ASIMOV irfs=P8R3_ULTRACLEANVETO_V3 expcube=$LTCUBE bexpmap=$EXPMAP

# Deactivate environment
conda deactivate

# REFERENCES

# Conda Init Error - https://stackoverflow.com/questions/77901825/unable-to-activate-environment-conda-prompted-to-run-
# conda-init-before-cond
# Fermitools Documentation - https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/overview.html
# IRF Justification - https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/lat_data_selection.html
# Selection Recommendations - https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/lat_data_selection.html
