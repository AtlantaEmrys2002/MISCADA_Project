#!/usr/bin/env bash -l

# RENAME THE FILE - create_dataset.sh

# File names - change to match the desired locations

# Raw data from the Fermi LAT collaboration
RAWDATAFILES=/Volumes/T7/data/weekly/photon
RAWDATA=/Volumes/T7/data/weekly/photon/events.txt
LIST=/Volumes/T7/data/weekly/photon/events.txt
SPACECRAFTFILE=/Volumes/T7/data/lat_spacecraft_merged.fits

# Stages of data preparation
FERMIDATACUT=/Volumes/T7/project_data/real_data/fermi_filtered.fits
FERMIDATACUTGTI=/Volumes/T7/project_data/real_data/fermi_filtered_gti.fits
FERMILIVETIMECUBE=/Volumes/T7/project_data/real_data/fermi_filtered_gti_livetime_cube.fits
FERMIBINNED=/Volumes/T7/project_data/real_data/fermi_filtered_gti_binned.fits
FERMIEXPMAP=/Volumes/T7/project_data/real_data/fermi_filtered_gti_exposure_map.fits

# Preparing PSF
POINTSOURCEPSF=/Volumes/T7/project_data/real_data/pointsource_psf.fits

source /Users/milli/miniconda3/bin/activate

conda init

# Enter Fermi environment
conda activate fermi

# Store the names of all relevant .fits files in events.txt
# ls $RAWDATAFILES/*.fits > $LIST

# Select front and back events (ID8) with energy between 300 MeV and 200 GeV (recommended by ID8) and zenith cut is 100
# degrees (ID25)
# gtselect evclass=128 evtype=3 infile=@/Volumes/T7/data/weekly/photon/events.txt outfile=$FERMIDATACUT ra=INDEF dec=INDEF rad=INDEF tmin=INDEF tmax=INDEF emin=300 emax=200000 zmax=100

# Select corresponding good time intervals (GTI) - we do not apply a zenith cut here, as we will loose all our data
# - therefore, we apply the zenith cut when running the gtltcube function
# gtmktime scfile=$SPACECRAFTFILE evfile=$FERMIDATACUT outfile=$FERMIDATACUTGTI filter="(DATA_QUAL == 1) && (LAT_CONFIG == 1) && (IN_SAA != T)" roicut=no

# Generate live time cube and apply zenith cut to data here - kept the default values for the spatial grid and
# inclination angle
# gtltcube evfile=$FERMIDATACUTGTI scfile=$SPACECRAFTFILE outfile=$FERMILIVETIMECUBE zmax=100 dcostheta=0.025 binsz=1

# Bin data and create count maps - this is to ensure that gtexpcube2 produces Healpix-style exposure maps (rather than
# easy 2D arrays) - this is a major flaw in the Fermitools with no documentation. YOU MAY HAVE TO CLICK ENTER TO SAY ALL-SKY AND type yes for energy binning
# gtbin evfile=$FERMIDATACUTGTI scfile=NONE outfile=$FERMIBINNED algorithm=HEALPIX coordsys=GAL ebinalg=LOG emin=300 emax=200000 enumbins=5 nxpix=360 nypix=180 binsz=0.23 hpx_ordering_scheme=RING hpx_order=8 xref=0 axisrot=0 proj=AIT yref=0

# Generate exposure maps - N.B. chose to use latitude, as this is consistent with the "robust neural ..." paper. bins`
# came from the introductory paragraph of section 2 of the "robust neural..." paper. Used AIT projection, but it may be
# Cartesian. Pass FERMIBINNED to ensure it is is in Healpix format
# gtexpcube2 infile=$FERMILIVETIMECUBE cmap=$FERMIBINNED outfile=$FERMIEXPMAP irfs=P8R3_ULTRACLEANVETO_V3 enumbins=5 emin=300 emax=200000 binsz=0.23 evtype=3

# ABOVE IS ORIGINAL AND WORKS - BELOW IS TO SEE IF I CAN PRODUCE A BETTER FUNCTION
gtexpcube2 infile=$FERMILIVETIMECUBE cmap=$FERMIBINNED outfile=$FERMIEXPMAP irfs=P8R3_ULTRACLEANVETO_V3 enumbins=16 emin=300 emax=200000 binsz=0.23 evtype=3

# Generate the PSF for handling pointlike sources - want at l = 90 degrees, b = 40 degrees - converted to RA DEC J2000
# which produces 252.131996, 41.585827°
# gtpsf expcube=$FERMILIVETIMECUBE outfile=$POINTSOURCEPSF irfs=P8R3_ULTRACLEANVETO_V3 ra=252.131996 dec=41.585827 emin=300 emax=200000 nenergies=5 thetamax=50 ntheta=300 evtype=3

# Deactivate Fermi environment
conda deactivate

# REFERENCES

# Fermitools Formats - https://fermi-hero.readthedocs.io/en/latest/galactic_center/science_tool_images.html
# Fermitools Likelihood Tutorial - https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/binned_likelihood_tutorial.
# html
# Preparing Fermi Data - http://cta.irap.omp.eu/ctools/users/tutorials/howto/fermi/howto_fermi_prepare.html

