# MISCADA_Project

## Set-Up

### ``fermitools`` Installation

This package makes use of the Fermi LAT Collaboration's ``fermitools`` in a series of ``bash`` scripts for filtering and
formatting the telescope data, as well as when simulating the data. Whilst these scripts can be executed automatically,
the ``fermitools`` software must be pre-installed. The instructions for installing the software can be found
[here](https://fermi.gsfc.nasa.gov/ssc/data/analysis/software/) and this author would recommend using miniconda to
manage the ``fermitools`` installation (``miniconda`` also requires installation - instructions can be found
[here](https://www.anaconda.com/docs/getting-started/miniconda/main)).

### Package Dependencies

This software requires the pre-installation of several popular Python packages. Each package dependency and version is 
listed in [``requirements.txt``](./requirements.txt). To install the packages, run the following command in the root 
directory:

```bash
pip install -r ./requirements.txt
```