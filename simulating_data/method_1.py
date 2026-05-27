# This method is adapted from ID4. All code is my own (except where indicated), but  Python
# implementation provided by the authors to *access* datacan be found here
# are as follows:
# 1) I wanted to find out how to simulate Fermi data and the description in the paper provided a step-by-step method.
# 2) I hoped to improve upon their implementation performance-wise - by implementing from scratch, I am familiar with
# the code and can improve it more easily.
# 3) I wanted to understand the method so that I could reimplement the code in C/C++ to make use of parallel processing
# and GPUs.
# 4) Once I had reimplemented (possibly in two languages) and made optimisations, I could then improve simulation
# techniques and bring in new ideas, e.g. time data, light curves, etc.
# 5) NOT ALL THE METHODS FOR SIMULATING DATA WERE PROVIDED IN THE ABOVE CODE - MORE ABOUT ACCESSING PRE-GENERATED DATA!
# - CHECK - IT'S ALL ABOUT ACCESSING PREGENERATED DATA - https://git.io/JO5FP - COULD USE TO READ MY XML FILES AND
# GENERATE PATCHES.

#TODO
# 1. Luminosity Function - use 3FGL (and cite the paper in notes so can cite in final report) to generate sources
# according to luminosity function - see graph in paper.
# 2. Simulate other source types, e.g. SN, differentiate between AGN types
# 3. XML saving
# 4. DON'T FORGET TO ADD IN DIFFUSE BACKGROUND (GALACTIC AND INTERGALACTIC TO XML FILES)
# 5. CHECK MIN-MAX VALUES IN XML

# LIBRARIES
import numpy as np
from scipy.integrate import quad
from astropy.table import QTable
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5
import matplotlib.pyplot as plt
import warnings
import xml.etree.ElementTree as ET
from xml.dom import minidom

# SPECTRAL MODELS

def agn_spectral_model(E, E_0, F_0, alpha, beta):

    division = E/E_0

    # with suppress(RuntimeWarning):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exponent = - alpha - (beta * np.log(division))

    dF_dE = F_0 * np.power(division, exponent)

    return E * dF_dE


def pulsar_spectral_model(E, F_0, E_0, Gamma, a, b):

    exponent = a * (np.power(E_0, b) - np.power(E, b))

    dF_dE = F_0 * np.power((E/E_0), -Gamma) * np.exp(exponent)

    return E * dF_dE


def energy_flux_agn(pivot_energy, flux_density, spectral_slope, curvature):

    # Integrate over 0.1 - 100 GeV
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        energy = quad(agn_spectral_model, 0.1, 100, args=(pivot_energy, flux_density, spectral_slope, curvature))[0]

    return energy


def s1_agn(pivot_energy, flux_density, spectral_slope, curvature):

    # Integrate above 1 GeV
    s1 = quad(agn_spectral_model, 1, np.inf, args=(pivot_energy, flux_density, spectral_slope, curvature))[0]

    return s1


def s10_agn(pivot_energy, flux_density, spectral_slope, curvature):

    # Integrate above 1 GeV
    s1 = quad(agn_spectral_model, 10, np.inf, args=(pivot_energy, flux_density, spectral_slope, curvature))[0]

    return s1


def energy_flux_pulsar(pivot_energy, flux_density, spectral_slope, exponential_index, exponential_factor):

    # Integrate over 0.1 - 100 GeV
    energy = quad(pulsar_spectral_model, 0.1, 100, args=(pivot_energy, flux_density, spectral_slope,
                                                         exponential_index, exponential_factor))[0]

    return energy


# DATA PREP

def catalog_data_preparation(file_name):

    # Access 4FGL catalog - note, file originally called gll_psc_v35.fit
    # Read catalog data and format as astropy QTable (allowing for units to be associated with each column, if necessary
    # Assume catalog data conforms to standard NASA format
    catalog = QTable.read(file_name, format='fits', hdu=1)

    # Select relevant columns
    columns = ('Pivot_Energy', 'LP_Flux_Density', 'PLEC_Flux_Density', 'LP_Index', 'LP_beta', 'PLEC_IndexS',
               'PLEC_Exp_Index', 'PLEC_ExpfactorS', 'CLASS1')
    catalog = catalog[columns]

    # Reformat CLASS1 column - remove empty spaces and make all lower case
    catalog['CLASS1'].name = 'Prev_CLASS1'
    catalog['CLASS1'] = np.asarray([k.strip().lower() for k in catalog['Prev_CLASS1']])
    catalog.remove_column('Prev_CLASS1')

    # Select all rows that describe pulsars
    pulsar_mask = (catalog['CLASS1'] == 'psr')

    # Select all rows that describe AGN
    agn_mask = ((catalog['CLASS1'] == 'bcu') | (catalog['CLASS1'] == 'sey') | (catalog['CLASS1'] == 'ssrq') |
                (catalog['CLASS1'] == 'bll') | (catalog['CLASS1'] == 'fsrq') | (catalog['CLASS1'] == 'rdg') |
                (catalog['CLASS1'] == 'nlsy1') | (catalog['CLASS1'] == 'agn'))

    catalog.remove_column('CLASS1')

    # Separate into AGN and pulsars
    pulsars = catalog[pulsar_mask]
    agns = catalog[agn_mask]

    return agns, pulsars


def agn_statistics(agns):

    # Select alpha values and convert from masked to ordinary numpy array
    alphas = agns['LP_Index'].data.filled(np.nan)

    # Select beta values and convert from masked to ordinary numpy array
    betas = agns['LP_beta'].data.filled(np.nan)

    # Select pivot energy values and convert from MeV to GeV
    pivot_energies = agns['Pivot_Energy'].to(u.GeV).value

    # Select flux density values and convert from ph / (cm2 MeV s) to ph / (cm2 GeV s)
    flux_densities = agns['LP_Flux_Density'].to(u.ph / (u.cm * u.cm * u.GeV * u.s)).value.filled(np.nan)

    log_flux_densities = np.log(flux_densities)

    mean_alpha, std_alpha = np.nanmean(alphas), np.nanstd(alphas, ddof=1)

    mean_pivot_energy, std_pivot_energy = np.nanmean(pivot_energies), np.nanstd(pivot_energies, ddof=1)

    # mean_log_flux_density, std_log_flux_density = np.nanmean(log_flux_densities), np.nanstd(log_flux_densities,
    # ddof=1)

    mean_flux_density, std_flux_density = np.nanmean(flux_densities), np.nanstd(flux_densities, ddof=1)

    tmp = (mean_flux_density**2) / (np.sqrt((mean_flux_density**2) + std_flux_density**2))

    tmp2 = np.log(1 + ((std_flux_density**2)/(mean_flux_density**2)))

    mean_log_flux_density, std_log_flux_density = np.log(tmp), np.sqrt(tmp2)

    return (mean_alpha, std_alpha, mean_pivot_energy, std_pivot_energy, mean_log_flux_density, std_log_flux_density,
            betas)


def pulsar_statistics(pulsars):

    # Select pivot energy values and convert from MeV to GeV
    pivot_energies = pulsars['Pivot_Energy'].to(u.GeV).value

    # Select Gamma values and convert from masked to ordinary numpy array
    Gammas = pulsars['PLEC_IndexS'].data.filled(np.nan)

    # Select exponential indices
    b_values = pulsars['PLEC_Exp_Index'].data.filled(np.nan)

    # Select exponential factors - CHECK (SAYS IN UNITS OF MeV^-b BUT NEED with GeV)
    a_values = pulsars['PLEC_ExpfactorS'].data

    # Select flux density values and convert from ph / (cm2 MeV s) to ph / (cm2 GeV s)
    flux_densities = pulsars['PLEC_Flux_Density'].to(u.ph / (u.cm * u.cm * u.GeV * u.s)).value
    log_flux_densities = np.log(flux_densities)

    mean_Gamma, std_Gamma = np.nanmean(Gammas), np.nanstd(Gammas, ddof=1)

    mean_b, std_b = np.nanmean(b_values), np.nanstd(b_values, ddof=1)

    mean_a, std_a = np.nanmean(a_values), np.nanstd(a_values, ddof=1)

    mean_log_flux_density, std_log_flux_density = np.nanmean(log_flux_densities), np.nanstd(log_flux_densities)

    mean_pivot_energy, std_pivot_energy = np.nanmean(pivot_energies), np.nanstd(pivot_energies, ddof=1)

    return (mean_Gamma, std_Gamma, mean_b, std_b, mean_a, std_a, mean_log_flux_density, std_log_flux_density,
            mean_pivot_energy, std_pivot_energy)


def generate_mock_agn_catalog(agn_stats, num_agns=4000):

    # SPECTRAL PARAMETERS

    (mean_alpha_agn, std_alpha_agn, mean_pivot_energy_agn, std_pivot_energy_agn, mean_log_flux_density_agn,
     std_log_flux_density_agn, betas_agn) = agn_stats

    # Generate new pivot energies
    pivot_energies = np.random.normal(loc=mean_pivot_energy_agn, scale=std_pivot_energy_agn, size=num_agns)

    # Generate new flux densities - log-normal for flux densities
    log_flux_densities = np.random.normal(loc=mean_log_flux_density_agn, scale=std_log_flux_density_agn, size=num_agns)

    # CHECK BELOW LINE

    flux_densities = np.exp(mean_log_flux_density_agn + (std_log_flux_density_agn * log_flux_densities))

    # flux_densities = np.exp(log_flux_densities)

    # Generate new spectral slopes (alphas)
    spectral_slopes = np.random.normal(loc=mean_alpha_agn, scale=std_alpha_agn, size=num_agns)

    # Generate new curvatures by directly sampling 4FGL
    betas = np.random.choice(betas_agn, size=num_agns, replace=True)

    # Combine into one array

    # DO NOT CHANGE ORDER OF STACKING - XML WRITER INDEXES FLUX DENSITY AT 1
    parameters = np.stack((pivot_energies, flux_densities, spectral_slopes, betas), axis=-1)

    # Energy fluxes
    energy_fluxes = np.fromiter((energy_flux_agn(x[0], x[1], x[2], x[3]) for x in parameters), np.float64)

    # Select rows with valid energy fluxes
    mask = ~np.isnan(energy_fluxes)
    energy_fluxes = energy_fluxes[mask]
    parameters = parameters[mask]

    # Combine two arrays to create mock catalog's spectral parameters
    parameters = np.concatenate((parameters, np.array([energy_fluxes]).T), axis=1)

    # SPATIAL PARAMETERS

    # IMPOrtANT - TAKE THESE TO BE AT INDEX 4 and 5 IN A GIVEN ROW

    galactic_longitudes = np.random.uniform(low=0, high=2*np.pi, size=len(parameters))

    sin_galactic_latitudes = np.random.uniform(low=-1, high=1, size=len(parameters))
    galactic_latitudes = np.arcsin(sin_galactic_latitudes)

    # CHECK - LUMINOSITY FUNCTION!!!! - WHAT ARE THEY DOING?

    plt.xscale('log')

    # print(energy_fluxes.shape)

    # print(energy_fluxes)
    #
    # bin_edges = 10**np.linspace(-16, -5, 400)
    #
    # print(bin_edges)
    #
    # counts, bins = np.histogram(energy_fluxes, bins=bin_edges, cumulative=True)

    # counts, bins = np.histogram(energy_fluxes, bins=10)
    #
    # plt.stairs(counts, bins)

    plt.xlabel('Energy Flux')
    plt.ylabel('No. Sources')


    plt.show()

    # NOT FINISHED -


def generate_mock_pulsar_catalog(pulsar_stats, num_pulsars=350):

    (mean_Gamma_pulsars, std_Gamma_pulsars, mean_b_pulsars, std_b_pulsars, mean_a_pulsars, std_a_pulsars,
     mean_log_flux_density_pulsars, std_log_flux_density_pulsars, mean_pivot_energy_pulsars, std_pivot_energy_pulsars) \
        = pulsar_stats

    # NOT FINISHED


def agn_xml_writer(sources):

    # CREATE DOCUMENT

    root = minidom.Document()

    xml = root.createElement('source_library')

    xml.setAttribute('title', 'source library')

    root.appendChild(xml)

    # ADD SOURCES

    # Ranges are taken from https://git.io/JO5FP - i.e. recommended by ID4
    agn_parameters = ["norm", "alpha", "Eb", "beta"]
    ranges_of_agn_parameters = [("0.001", "1000.0"), ("-5000.0", "1000.0"), ("0.0000001", "10000000000000.0"),
                                ("-100.0", "100")]

    for k in range(len(sources)):

        source = root.createElement("source")

        source.setAttribute("name", "AGN_" + str(k))
        source.setAttribute("type", "PointSource")

        xml.appendChild(source)

        # SPECTRAL

        spectrum = root.createElement("spectrum")

        spectrum.setAttribute("type", "LogParabola")

        for x in range(len(agn_parameters)):

            param = root.createElement("parameter")
            param.setAttribute("free", "1")
            param.setAttribute("max", str(ranges_of_agn_parameters[x][1]))
            param.setAttribute("min", str(ranges_of_agn_parameters[x][0]))
            param.setAttribute("name", agn_parameters[x])

            if x % 2 != 0:
                param.setAttribute("scale", "-1.0")
                if x == 1:
                    param.setAttribute("value", str(sources[k][2]))
                else:
                    param.setAttribute("value", str(sources[k][3]))
            elif x == 2:
                param.setAttribute("scale", "1.0")
                param.setAttribute("value", str(sources[k][0]))
            else:
                # Set to value of
                param.setAttribute("scale", str(sources[k][1]))
                param.setAttribute("value", str(1))

            spectrum.appendChild(param)

        source.appendChild(spectrum)

        # SPATIAL

        spatial = root.createElement("spatialModel")

        spatial.setAttribute("type", "SkyDirFunction")

        ra = root.createElement("parameter")
        dec = root.createElement("parameter")

        ra.setAttribute("free", "0")
        ra.setAttribute("max", "360.")
        ra.setAttribute("min", "-360.")
        ra.setAttribute("name", "RA")
        ra.setAttribute("scale", "1.0")

        dec.setAttribute("free", "0")
        dec.setAttribute("max", "90.")
        dec.setAttribute("min", "-90.")
        dec.setAttribute("name", "DEC")
        dec.setAttribute("scale", "1.0")

        # Convert galactic coordinates to 
        gc = SkyCoord(l=sources[k][4])

        # ra.setAttribute("value", str(sources[k][4]))



    # FORMAT

    xml_str = root.toprettyxml(indent="\t")

    # SAVE

    save_path_file = "sources.xml"

    with open(save_path_file, "w") as f:
        f.write(xml_str)



# agn_rows, pulsar_rows = catalog_data_preparation("/Volumes/T7/data/catalog/4FGL_DR4.fit")
#
# pulsar_statistics(pulsar_rows)
#
#
# generate_mock_agn_catalog(agn_statistics(agn_rows), 4000)

# generate_mock_pulsar_catalog(pulsar_statistics(pulsar_rows), 10)

agn_xml_writer(sources=[[1, 2, 3, 4, 0, 45], [2, 4, 6, 8, 0, 45]])


# REFERENCES

# Astropy Documentation - https://docs.astropy.org/en/stable/
# ID8 Paper - Identification of point sources in gamma rays using U-shaped convolutional neural networks and a data
# challenge
# Masked to Ordinary Numpy Array - https://www.w3resource.com/python-exercises/numpy/convert-masked-numpy-array-to-regul
# ar-array-with-nan.php
# Numpy Documentation - https://numpy.org/doc/stable/user/index.html
# Scipy Documentation - https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html#scipy.integrate.
# quad

