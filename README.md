# MISCADA Project

This software package was constructed for my final research project undertaken during a Master's in Scientific Computing
and Data Analysis (Astrophysics) course.

## Set-Up

### ``fermitools`` Installation

This package makes use of the Fermi LAT Collaboration's ``fermitools`` in a series of ``bash`` scripts for filtering and
formatting the telescope data, as well as when simulating the data. Whilst these scripts can be executed automatically,
the ``fermitools`` software must be pre-installed. The instructions for installing the software can be found
[here](https://fermi.gsfc.nasa.gov/ssc/data/analysis/software/) and this author would recommend using miniconda to
manage the ``fermitools`` installation (``miniconda`` also requires installation - instructions can be found
[here](https://www.anaconda.com/docs/getting-started/miniconda/main)). Fermitools documentation can be found 
[here](https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/overview.html) - this documentation has been consulted 
heavily throughout the course of this project.

### Package Dependencies

This software requires the pre-installation of several popular Python packages. Each package dependency and version is 
listed in [``requirements.txt``](./requirements.txt). To install the packages, run the following command in the root 
directory:

```bash
pip install -r ./requirements.txt
```

### Accessing Fermi-LAT Data

**NEED TO FILL IN**

## Create Simulation Environment and Dataset

**NEED TO FILL IN**





## References

Within each file, I have noted any links or materials that I consulted whilst developing specfic functions included in 
that file. However, the documentation for certain Python packages, e.g. scipy, was utilised heavily throughout this 
entire project and, therefore, is listed below (along with a description of its functionality). I would recommend 
consulting this documentation when studying the code. *N.B. I have also consulted the official 
[Python documentation](https://docs.python.org/3/library/xml.dom.minidom.html#module-xml.dom.minidom)*.

- [Astropy Documentation](https://www.astropy.org) - package for functionality relating to astronomical analysis, 
calculations, and conversions (the efficient conversion of celestial/equatorial to galactic coordinates and vice 
versa has proved invaluable).
- [Healpy Documentation](https://healpy.readthedocs.io/en/latest/index.html) - HEALPix is a data format for efficiently 
and logically storing large-scale maps (as well as other astronomical data). The access, manipulation, and visualisation 
of data in this format is achieved through ``healpy``.
- [Matplotlib Documentation](https://matplotlib.org) - used to create visual illustrations of scientific data, e.g. 
luminosity function histograms of 4FGL data.
- [Numpy Documentation](https://numpy.org/doc/) - well-known Python package for manipulating and performing calculations
on arrays and matrices of data.
- [Reproject Documentation](https://reproject.readthedocs.io/en/stable/) - this package is utilised to convert data 
stored in ``.fits`` files from one world coordinate system (see ``astropy`` documentation) to another (especially useful
when reading in background emission models provided by Fermi LAT Collaboration).
- [Scipy Documentation](https://docs.scipy.org/doc/scipy/) - well-known package for performing complex scientific and 
mathematical calculations in a computationally efficient manner. This project has primarily made use of the ``stats``
submodule for analysing 4FGL data, the ``optimize`` submodule for fitting functions to data, the ``spatial`` submodule
for calculating metrics, and the ``integrate`` submodule for performing calculus.
- [skLearn Documentation](https://scikit-learn.org/stable/index.html) - includes machine learning and statistics 
functionality.

&copy; Millicent Riordan, 2026 