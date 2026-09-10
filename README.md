# Gamma-Ray Astronomical Source Simulation and Extraction (GRASSE) Package

This software package was constructed for my final research project undertaken during a Master's in Scientific Computing
and Data Analysis (Astrophysics) course. This code is not fully formatted - I may return to this project in the future,
tidy up files, and implement more functionality.

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

Please follow the information provided by the *Fermi*-LAT Collaboration regarding access to weekly photon files and 
catalogues. The bash scripts in ``data_simulation`` will recreate any *Fermi* data used throughout this project. 

## Modules

- ``data_simulation`` - provides a method for simulating realistic, unbiased energy-binned *Fermi*-LAT count maps, 
combining and improving upon methods presented in 
[Panes *et al. (2021)*](https://www.aanda.org/articles/aa/full_html/2021/12/aa41193-21/aa41193-21.html) and
[Eckner *et al.* (2025)](https://arxiv.org/pdf/2505.02906). Functionality includes an analysis pipeline for fitting
suitable [probability density functions](https://en.wikipedia.org/wiki/Probability_density_function) to the histograms
of [4FGL](https://heasarc.gsfc.nasa.gov/w3browse/fermi/fermilpsc.html) $\gamma$-ray source spectral and spatial 
parameters, bash scripts for extracting exposure maps and point spread 
functions (PSFs) from unprocessed *Fermi* observations, and an Asimovian method for simulating count maps. 
``simulated_data`` contains a pre-built dataset of 76,800 10$^\circ \times$ 10$^\circ$ count map patches.
- ``source_extractors`` - this section details a pipeline, as well as several custom/reimplemented algorithms for 
segmenting, localising, and classifying sources in both simulated and real *Fermi*-LAT count maps. These algorithms make
use of machine learning and deep learning techniques. The pipeline is designed such that new algorithms can be easily
introduced.
- ``evaluation`` - this provides implementations of several popular (and custom) metrics for evaluating each stage of a
source extraction algorithm.
- ``report``- this is a work-in-progress file for building plots for my final project report.
- ``results`` - this provides an overview of my results when applying the source extraction algorithms to my simulated 
data and real *Fermi*-LAT count maps. We may have found several new $\gamma$-ray sources.
- ``Previous Work`` - this is my working area. I used this file to store any code from when I was learning about
*Fermi* data and count map simulation. It also contains a Python version of 
[Parkinson *et al.*](https://iopscience.iop.org/article/10.3847/0004-637X/820/1/8/meta)'s machine learning algorithms
for classifying unassociated *Fermi* sources according to their spectral features in the 4FGL.

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
- [scikit-image Documentation](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_trainable_segmentation.html#id4) - 
a library of image processing functions that can be used alongside traditional ML algorithms implemented in ``sklearn``.
- [skLearn Documentation](https://scikit-learn.org/stable/index.html) - includes machine learning and statistics 
functionality.

&copy; Millicent Riordan, 2026 