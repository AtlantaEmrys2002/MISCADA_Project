#!/usr/bin/env bash -l

source /Users/milli/miniconda3/bin/activate

conda init

RAWDATA=/Volumes/T7/project_data/real_data/diffuse_psf_roi/events.txt
SPACECRAFTFILE=/Volumes/T7/project_data/real_data/diffuse_psf_roi/L2606291550576C64FAA996_SC00.fits

FERMIDATACUT=/Volumes/T7/project_data/real_data/diffuse_psf_roi/roi_filtered.fits
FERMIDATACUTGTI=/Volumes/T7/project_data/real_data/diffuse_psf_roi/roi_filtered_gti.fits
FERMIDATALIVETIMECUBE=/Volumes/T7/project_data/real_data/diffuse_psf_roi/roi_filtered_livetime.fits
FERMIDATABINNED=/Volumes/T7/project_data/real_data/diffuse_psf_roi/roi_filted_bins.fits

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

# Create binned count map - this is different from the main function - we want this to be in a non-healpix format
gtbin evfile=$FERMIDATACUTGTI scfile=NONE outfile=$FERMIDATABINNED algorithm=HEALPIX coordsys=GAL ebinalg=LOG emin=300 emax=200000 enumbins=5 binsz=0.05 hpx_ordering_scheme=RING hpx_order=8 xref=90 axisrot=0 proj=AIT yref=40





conda deactivate
