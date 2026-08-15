#!/usr/bin/env python

# This sets up a virtual environment for running simulations with all the correct dependencies

# Create a virtual environment
python3 -m venv gamma_source_detection

# Activate environment
source gamma_source_detection/bin/activate

# Install dependencies
pip install -r ./requirements.txt

deactivate
