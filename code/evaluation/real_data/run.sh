#!/usr/bin/env python

source gamma_source_detection/bin/activate

python create_real_data_patches.py --num_energy_bins=5 --patch_directory=./real_patches --fermi_catalog_location=/Volumes/T7/data/catalog/4FGL_DR4.fit --binned_count_maps=/Volumes/T7/project_data/real_data/fermi_filtered_gti_binned_for_evaluation.fits

deactivate