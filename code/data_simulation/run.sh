#!/usr/bin/env python

source gamma_source_detection/bin/activate

# Create mock catalogs

python create_mock_catalogs.py --catalog=/Volumes/T7/data/catalog/4FGL_DR4.fit --number=3 --analyse=no --verify=no

deactivate
