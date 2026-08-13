#!/usr/bin/env python

source gamma_source_detection/bin/activate

# Create mock catalogs
# python create_mock_catalogs.py --catalog=/Volumes/T7/data/catalog/4FGL_DR4.fit --number=10 --analyse=yes --verify=first

# Create mock all-sky count maps for AGN, pulsar, and background
# python create_mock_skymaps.py --exposure_fits=/Volumes/T7/project_data/real_data/fermi_filtered_gti_exposure_map.fits --number_catalogs=10 --pointsource_psf=/Volumes/T7/project_data/real_data/pointsource_psf.fits --diffuse_source_psf_roi=/Volumes/T7/project_data/real_data/diffuse_psf_roi/count_map.fits --isotropic_background=/Volumes/T7/data/background_models/iso_P8R3_SOURCE_V3_v1.txt --galactic_background=/Volumes/T7/data/background_models/gll_iem_v07.fits --num_maps_per_catalog=10

# python create_mock_skymaps_new.py --exposure_fits=/Volumes/T7/project_data/real_data/fermi_filtered_gti_exposure_map.fits --number_catalogs=1 --pointsource_psf=/Volumes/T7/project_data/real_data/pointsource_psf.fits --diffuse_source_psf_roi=/Volumes/T7/project_data/real_data/diffuse_psf_roi/count_map.fits --isotropic_background=/Volumes/T7/data/background_models/iso_P8R3_SOURCE_V3_v1.txt --galactic_background=/Volumes/T7/data/background_models/gll_iem_v07.fits --num_maps_per_catalog=1
#

python create_dataset_2.py --num_catalogs=1 --num_skymaps_per_catalog=1 --num_energy_bins=5 --patch_directory=./simulated_data


# python create_dataset.py --num_catalogs=10 --num_skymaps_per_catalog=10 --num_energy_bins=5 --patch_directory=./simulated_data

# python create_infinite_stats_patches.py --num_catalogs=10 --num_skymaps_per_catalog=10 --num_energy_bins=5 --patch_directory=./simulated_data

# Sources with too-low energy fluxes removed
# python create_dataset_flux_dependent.py --num_catalogs=10 --num_skymaps_per_catalog=10 --num_energy_bins=5 --patch_directory=./simulated_data

deactivate


# REFERENCES

# Argparse Choices - https://stackoverflow.com/questions/15836713/allowing-specific-values-for-an-argparse-argument
# Argparse Tutorial - https://medium.com/@evaGachirwa/running-python-script-with-arguments-in-the-command-line-
# 93dfa5f10eff
# Venv Creation - https://www.w3schools.com/PYTHON/python_virtualenv.asp
