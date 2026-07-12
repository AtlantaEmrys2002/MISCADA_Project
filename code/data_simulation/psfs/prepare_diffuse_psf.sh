#!/usr/bin/env bash -l

source /Users/milli/miniconda3/bin/activate

conda init

RAWDATA=/Volumes/T7/project_data/real_data/diffuse_psf_roi/events.txt
SPACECRAFTFILE=/Volumes/T7/project_data/real_data/diffuse_psf_roi/L2606291550576C64FAA996_SC00.fits
MODELS=./region.xml


FERMIDATACUT=/Volumes/T7/project_data/real_data/diffuse_psf_roi/roi_filtered.fits
FERMIDATACUTGTI=/Volumes/T7/project_data/real_data/diffuse_psf_roi/roi_filtered_gti.fits
FERMIDATALIVETIMECUBE=/Volumes/T7/project_data/real_data/diffuse_psf_roi/roi_filtered_livetime.fits
FERMIDATABINNED=/Volumes/T7/project_data/real_data/diffuse_psf_roi/roi_filted_bins.fits
FERMIEXPMAP=/Volumes/T7/project_data/real_data/diffuse_psf_roi/roi_filted_exposure.fits
FERMISOURCEMAP=/Volumes/T7/project_data/real_data/diffuse_psf_roi/roi_filted_source_map.fits
FERMICOUNTMAP=/Volumes/T7/project_data/real_data/diffuse_psf_roi/count_map.fits

# Enter Fermi environment
conda activate fermi

# Create list of files
# ls /Volumes/T7/project_data/real_data/diffuse_psf_roi/*PH*.fits > $RAWDATA

# Perform the same data cut as for the whole sky
# gtselect evclass=128 evtype=3 infile=@$RAWDATA outfile=$FERMIDATACUT ra=INDEF dec=INDEF rad=30 tmin=INDEF tmax=INDEF emin=300 emax=200000 zmax=100

# Correct good time intervals
# gtmktime scfile=$SPACECRAFTFILE evfile=$FERMIDATACUT outfile=$FERMIDATACUTGTI filter="(DATA_QUAL == 1) && (LAT_CONFIG == 1) && (IN_SAA != T)" roicut=no

# Create livetime cube
# gtltcube evfile=$FERMIDATACUTGTI scfile=$SPACECRAFTFILE outfile=$FERMIDATALIVETIMECUBE dcostheta=0.025 binsz=1


# ORIGINAL

# Create binned count map - this is different from the main function - we want this to be in a non-healpix format
gtbin evfile=$FERMIDATACUTGTI scfile=NONE outfile=$FERMIDATABINNED algorithm=CCUBE coordsys=GAL ebinalg=LOG proj=AIT nxpix=1200 nypix=1200 emin=300 emax=200000 enumbins=16 binsz=0.05 xref=90 axisrot=0 yref=40

# Create exposure map
gtexpcube2 infile=$FERMIDATALIVETIMECUBE cmap=$FERMIDATABINNED outfile=$FERMIEXPMAP irfs=P8R3_ULTRACLEANVETO_V3 nxpix=1200 nypix=1200 xref=90 yref=40 axisrot=0 proj=AIT coordsys=GAL enumbins=16 emin=300 emax=200000 binsz=0.05 evtype=3

# Convolve source model components with instrument response
gtsrcmaps scfile=$SPACECRAFTFILE expcube=$FERMIDATALIVETIMECUBE cmap=$FERMIDATABINNED bexp=$FERMIEXPMAP emapbnds=no outfile=$FERMISOURCEMAP irfs=P8R3_ULTRACLEANVETO_V3 srcmdl=$MODELS

# Model - it does not matter what the pivot energy and flux density of the point source is - as we are looking at spatial parameters
gtmodel srcmaps=$FERMISOURCEMAP srcmdl=$MODELS outfile=$FERMICOUNTMAP irfs=P8R3_ULTRACLEANVETO_V3 expcube=$FERMIDATALIVETIMECUBE bexp=$FERMIEXPMAP outtype=ccube

# EXPERIMENTAL

#punlearn gtbin

# ORDER SHOULD BE 9 AND HPXREGION = "" AND YES TO ENERGY BINNING

# Create binned count map - this is different from the main function - we want this to be in a non-healpix format
#gtbin evfile=$FERMIDATACUTGTI scfile=NONE outfile=$FERMIDATABINNED algorithm=HEALPIX coordsys=GAL ebinalg=LOG proj=AIT nxpix=1200 nypix=1200 emin=300 emax=200000 enumbins=16 binsz=0.05 xref=90 axisrot=0 yref=40
#
## Create exposure map
#gtexpcube2 infile=$FERMIDATALIVETIMECUBE cmap=$FERMIDATABINNED outfile=$FERMIEXPMAP irfs=P8R3_ULTRACLEANVETO_V3 nxpix=1200 nypix=1200 xref=90 yref=40 axisrot=0 proj=AIT coordsys=GAL enumbins=16 emin=300 emax=200000 binsz=0.05 evtype=3
#
## Convolve source model components with instrument response
#gtsrcmaps scfile=$SPACECRAFTFILE expcube=$FERMIDATALIVETIMECUBE cmap=$FERMIDATABINNED bexp=$FERMIEXPMAP emapbnds=no outfile=$FERMISOURCEMAP irfs=P8R3_ULTRACLEANVETO_V3 srcmdl=$MODELS
#
## Model - it does not matter what the pivot energy and flux density of the point source is - as we are looking at spatial parameters
#gtmodel srcmaps=$FERMISOURCEMAP srcmdl=$MODELS outfile=$FERMICOUNTMAP irfs=P8R3_ULTRACLEANVETO_V3 expcube=$FERMIDATALIVETIMECUBE bexp=$FERMIEXPMAP








conda deactivate
