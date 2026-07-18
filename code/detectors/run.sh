#!/usr/bin/env python

source gamma_source_detection/bin/activate

python run_benchmarks.py --patch_location=./../data_simulation/simulated_data/patches --num_patches=20

source deactivate
