#!/usr/bin/env python

source gamma_source_detection/bin/activate

# Create mock catalogs

python create_mock_catalogs.py --catalog=/Volumes/T7/data/catalog/4FGL_DR4.fit --number=3 --analyse=no --verify=first

deactivate


# REFERENCES

# Argparse Choices - https://stackoverflow.com/questions/15836713/allowing-specific-values-for-an-argparse-argument
# Argparse Tutorial - https://medium.com/@evaGachirwa/running-python-script-with-arguments-in-the-command-line-
# 93dfa5f10eff
# Venv Creation - https://www.w3schools.com/PYTHON/python_virtualenv.asp
