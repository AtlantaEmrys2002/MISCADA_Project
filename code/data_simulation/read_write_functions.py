from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable
import healpy as hp
import numpy as np
from pathlib import Path
from source_generation.agn_spectral_parameters import integral_photon_flux_agn, energy_flux_agn
from source_generation.pulsar_spectral_parameters import integral_photon_flux_pulsar, energy_flux_pulsar
from xml.dom import minidom, Node


def agn_xml_writer(sources, root, xml):

    # ADD SOURCES

    num_sources = len(sources)

    # Ranges are taken from https://git.io/JO5FP - i.e. recommended by ID8
    agn_parameters = ["norm", "alpha", "Eb", "beta"]
    ranges_of_agn_parameters = [("0.001", "1000.0"), ("-5000.0", "1000.0"), ("0.0000001", "10000000000000.0"),
                                ("-100.0", "100")]

    # Convert galactic coordinates to equatorial
    ra_dec = SkyCoord(l=sources[:, 5] * u.rad, b=sources[:, 6] * u.rad, frame='galactic').transform_to('icrs')

    ra_dec = [(str(ra_dec[k].ra.to_value(u.degree)), str(ra_dec[k].dec.to_value(u.degree))) for k in range(num_sources)]

    for k in range(num_sources):

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
                # CHANGED THIS LINE HERE - FROM -1.0 to 1.0
                param.setAttribute("scale", "1.0")
                if x == 1:
                    param.setAttribute("value", str(sources[k][2]))
                else:
                    param.setAttribute("value", str(sources[k][3]))
            elif x == 2:
                param.setAttribute("scale", "1.0")
                param.setAttribute("value", str(sources[k][0]))
            else:
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

        ra.setAttribute("value", ra_dec[k][0])
        dec.setAttribute("value", ra_dec[k][1])

        spatial.appendChild(ra)
        spatial.appendChild(dec)

        source.appendChild(spatial)


# def agn_xml_writer(sources, root, xml):
#
#     # ADD SOURCES
#
#     # Ranges are taken from https://git.io/JO5FP - i.e. recommended by ID8
#     agn_parameters = ["norm", "alpha", "Eb", "beta"]
#     ranges_of_agn_parameters = [("0.001", "1000.0"), ("-5000.0", "1000.0"), ("0.0000001", "10000000000000.0"),
#                                 ("-100.0", "100")]
#
#     for k in range(len(sources)):
#
#         source = root.createElement("source")
#
#         source.setAttribute("name", "AGN_" + str(k))
#         source.setAttribute("type", "PointSource")
#
#         xml.appendChild(source)
#
#         # SPECTRAL
#
#         spectrum = root.createElement("spectrum")
#
#         spectrum.setAttribute("type", "LogParabola")
#
#         for x in range(len(agn_parameters)):
#
#             param = root.createElement("parameter")
#             param.setAttribute("free", "1")
#             param.setAttribute("max", str(ranges_of_agn_parameters[x][1]))
#             param.setAttribute("min", str(ranges_of_agn_parameters[x][0]))
#             param.setAttribute("name", agn_parameters[x])
#
#             if x % 2 != 0:
#                 # CHANGED THIS LINE HERE - FROM -1.0 to 1.0
#                 param.setAttribute("scale", "1.0")
#                 if x == 1:
#                     param.setAttribute("value", str(sources[k][2]))
#                 else:
#                     param.setAttribute("value", str(sources[k][3]))
#             elif x == 2:
#                 param.setAttribute("scale", "1.0")
#                 param.setAttribute("value", str(sources[k][0]))
#             else:
#                 param.setAttribute("scale", str(sources[k][1]))
#                 param.setAttribute("value", str(1))
#
#             spectrum.appendChild(param)
#
#         source.appendChild(spectrum)
#
#         # SPATIAL
#
#         spatial = root.createElement("spatialModel")
#
#         spatial.setAttribute("type", "SkyDirFunction")
#
#         ra = root.createElement("parameter")
#         dec = root.createElement("parameter")
#
#         ra.setAttribute("free", "0")
#         ra.setAttribute("max", "360.")
#         ra.setAttribute("min", "-360.")
#         ra.setAttribute("name", "RA")
#         ra.setAttribute("scale", "1.0")
#
#         dec.setAttribute("free", "0")
#         dec.setAttribute("max", "90.")
#         dec.setAttribute("min", "-90.")
#         dec.setAttribute("name", "DEC")
#         dec.setAttribute("scale", "1.0")
#
#         # Convert galactic coordinates to equatorial
#         ra_dec = SkyCoord(l=sources[k][5] * u.rad, b=sources[k][6] * u.rad, frame='galactic').transform_to('icrs')
#
#         ra.setAttribute("value", str(ra_dec.ra.to_value(u.degree)))
#         dec.setAttribute("value", str(ra_dec.dec.to_value(u.degree)))
#
#         spatial.appendChild(ra)
#         spatial.appendChild(dec)
#
#         source.appendChild(spatial)


def catalog_data_preparation(file_name: str):

    # Access 4FGL catalog - note, file originally called gll_psc_v35.fit
    # Read catalog data and format as astropy QTable (allowing for units to be associated with each column, if necessary
    # Assume catalog data conforms to standard NASA format
    catalog = QTable.read(file_name, format='fits', hdu=1)

    # Select relevant columns
    columns = ("Pivot_Energy", "LP_Flux_Density", "PLEC_Flux_Density", "LP_Index", "LP_beta", "PLEC_IndexS",
               "PLEC_Exp_Index", "PLEC_ExpfactorS", "CLASS1", "GLAT")

    # Find detection threshold of sources (characterised by minimum energy flux)
    source_detection_threshold = np.min(catalog["Energy_Flux100"].value)

    # Used to determine the number of sources to generate
    energy_fluxes_4fgl = catalog["Energy_Flux100"].value

    # Select relevant columns
    catalog = catalog[columns]

    # Reformat CLASS1 column - remove empty spaces and make all lower case
    catalog["CLASS1"] = np.asarray([k.decode('utf-8').strip().lower() for k in catalog["CLASS1"].value.filled('-')])

    # Select all rows that describe pulsars
    pulsar_mask = (catalog["CLASS1"] == "psr")

    # Select all rows that describe AGN
    agn_mask = np.isin(catalog["CLASS1"].data, np.array(["bcu", "sey", "ssrq", "bll", "fsrq", "rdg", "nlsy1", "agn"]))

    # Delete unnecessary column
    catalog.remove_column("CLASS1")

    agn_data = catalog[agn_mask].copy()
    pulsar_data = catalog[pulsar_mask].copy()

    # Select relevant columns for each source type

    # N.B. remove spatial column for AGNS - AGNs are known to be approximately isotropically distributed on the sky
    agn_data = agn_data["LP_Flux_Density", "Pivot_Energy", "LP_Index", "LP_beta"]

    pulsar_data = pulsar_data[
        ("PLEC_Flux_Density", "Pivot_Energy", "PLEC_IndexS", "PLEC_Exp_Index", "PLEC_ExpfactorS", "GLAT")]

    # Convert latitudes from degrees to radians
    pulsar_data["GLAT"] = pulsar_data["GLAT"].to(u.rad)

    # Separate into AGN and pulsars
    return agn_data, pulsar_data, source_detection_threshold, energy_fluxes_4fgl


def cleanEmptyTextNodes(node):
    # THIS RECURSIVE FUNCTION COMES FROM HERE - https://stackoverflow.com/questions/10034747/python-issue-using-xml-dom
    # -minidom-document-extra-empty-lines-between-child-ele

    for child in node.childNodes:

        if child.nodeType == Node.TEXT_NODE:
            child.data = ''

        elif child.nodeType == Node.ELEMENT_NODE:

            cleanEmptyTextNodes(child)

    node.normalize()


def pulsar_xml_writer(sources, root, xml):

    # PARAMETERS

    num_bins = len(sources)

    pulsar_parameters = ["Prefactor", "Index1", "Scale", "Expfactor", "Index2"]

    # Ranges are taken from https://git.io/JO5FP - i.e. recommended by ID8
    ranges_of_pulsar_parameters = [("0.00000001", "1000000000.0"), ("-50000.0", "5000.0"),
                                   ("-3000000.0", "3000000000000.0"), ("-100000000", "1000000"), ("0", "20")]

    # Convert galactic coordinates to equatorial
    ra_dec = SkyCoord(l=sources[:, -2] * u.rad, b=sources[:, -1] * u.rad, frame='galactic').transform_to('icrs')

    ra_dec = [(str(ra_dec[k].ra.to_value(u.degree)), str(ra_dec[k].dec.to_value(u.degree))) for k in range(num_bins)]

    for k in range(num_bins):

        source = root.createElement("source")

        source.setAttribute("name", "PSR_" + str(k))
        source.setAttribute("type", "PointSource")

        xml.appendChild(source)

        # SPECTRAL

        spectrum = root.createElement("spectrum")

        spectrum.setAttribute("type", "PLSuperExpCutoff2")

        for x in range(len(pulsar_parameters)):

            param = root.createElement("parameter")

            param.setAttribute("max", str(ranges_of_pulsar_parameters[x][1]))
            param.setAttribute("min", str(ranges_of_pulsar_parameters[x][0]))
            param.setAttribute("name", pulsar_parameters[x])

            if x == 0:

                # FLUX DENSITY OR PREFACTOR

                param.setAttribute("free", "1")
                param.setAttribute("scale", str(sources[k][1]))
                param.setAttribute("value", "1")

            elif x == 1:

                # SPECTRAL SLOPE OR GAMMA OR INDEX1

                param.setAttribute("free", "1")
                param.setAttribute("scale", "1.0")
                param.setAttribute("value", str(sources[k][2]))

            elif x == 2:

                # SCALE Eb OR PIVOT ENERGY

                param.setAttribute("free", "0")
                param.setAttribute("scale", "1.0")
                param.setAttribute("value", str(sources[k][0]))

            elif x == 3:

                # EXPONENTIAL FACTOR A

                param.setAttribute("free", "1")
                param.setAttribute("scale", "1.0")
                param.setAttribute("value", str(sources[k][4]))

            else:

                # INDEX2 OR B OR EXPONENTIAL INDEX

                param.setAttribute("free", "0")
                param.setAttribute("scale", "1")
                param.setAttribute("value", str(sources[k][3]))

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

        ra.setAttribute("value", ra_dec[k][0])
        dec.setAttribute("value", ra_dec[k][1])

        spatial.appendChild(ra)
        spatial.appendChild(dec)

        source.appendChild(spatial)


# def pulsar_xml_writer(sources, root, xml):
#
#     # PARAMETERS
#
#     pulsar_parameters = ["Prefactor", "Index1", "Scale", "Expfactor", "Index2"]
#
#     # Ranges are taken from https://git.io/JO5FP - i.e. recommended by ID8
#     ranges_of_pulsar_parameters = [("0.00000001", "1000000000.0"), ("-50000.0", "5000.0"),
#                                    ("-3000000.0", "3000000000000.0"), ("-100000000", "1000000"), ("0", "20")]
#
#     for k in range(len(sources)):
#
#         source = root.createElement("source")
#
#         source.setAttribute("name", "PSR_" + str(k))
#         source.setAttribute("type", "PointSource")
#
#         xml.appendChild(source)
#
#         # SPECTRAL
#
#         spectrum = root.createElement("spectrum")
#
#         spectrum.setAttribute("type", "PLSuperExpCutoff2")
#
#         for x in range(len(pulsar_parameters)):
#
#             param = root.createElement("parameter")
#
#             param.setAttribute("max", str(ranges_of_pulsar_parameters[x][1]))
#             param.setAttribute("min", str(ranges_of_pulsar_parameters[x][0]))
#             param.setAttribute("name", pulsar_parameters[x])
#
#             if x == 0:
#
#                 # FLUX DENSITY OR PREFACTOR
#
#                 param.setAttribute("free", "1")
#                 param.setAttribute("scale", str(sources[k][1]))
#                 param.setAttribute("value", "1")
#
#             elif x == 1:
#
#                 # SPECTRAL SLOPE OR GAMMA OR INDEX1
#
#                 param.setAttribute("free", "1")
#                 param.setAttribute("scale", "1.0")
#                 param.setAttribute("value", str(sources[k][2]))
#
#             elif x == 2:
#
#                 # SCALE Eb OR PIVOT ENERGY
#
#                 param.setAttribute("free", "0")
#                 param.setAttribute("scale", "1.0")
#                 param.setAttribute("value", str(sources[k][0]))
#
#             elif x == 3:
#
#                 # EXPONENTIAL FACTOR A
#
#                 param.setAttribute("free", "1")
#                 param.setAttribute("scale", "1.0")
#                 param.setAttribute("value", str(sources[k][4]))
#
#             else:
#
#                 # INDEX2 OR B OR EXPONENTIAL INDEX
#
#                 param.setAttribute("free", "0")
#                 param.setAttribute("scale", "1")
#                 param.setAttribute("value", str(sources[k][3]))
#
#             spectrum.appendChild(param)
#
#         source.appendChild(spectrum)
#
#         # SPATIAL
#
#         spatial = root.createElement("spatialModel")
#
#         spatial.setAttribute("type", "SkyDirFunction")
#
#         ra = root.createElement("parameter")
#         dec = root.createElement("parameter")
#
#         ra.setAttribute("free", "0")
#         ra.setAttribute("max", "360.")
#         ra.setAttribute("min", "-360.")
#         ra.setAttribute("name", "RA")
#         ra.setAttribute("scale", "1.0")
#
#         dec.setAttribute("free", "0")
#         dec.setAttribute("max", "90.")
#         dec.setAttribute("min", "-90.")
#         dec.setAttribute("name", "DEC")
#         dec.setAttribute("scale", "1.0")
#
#         # Convert galactic coordinates to equatorial
#
#         ra_dec = SkyCoord(l=sources[k][-2] * u.rad, b=sources[k][-1] * u.rad, frame='galactic').transform_to('icrs')
#
#         ra.setAttribute("value", str(ra_dec.ra.to_value(u.degree)))
#         dec.setAttribute("value", str(ra_dec.dec.to_value(u.degree)))
#
#         spatial.appendChild(ra)
#         spatial.appendChild(dec)
#
#         source.appendChild(spatial)


# def save_count_maps(count_maps, directory, catalog_id):
#
#     num_bins = len(count_maps)
#
#     path = directory + "map{}/".format(catalog_id)
#
#     # Create directory if it does not already exist
#     Path(path).mkdir(parents=True, exist_ok=True)
#
#     for c in range(num_bins):
#
#         filename = "count_map_bin_{}.png".format(c)
#
#         count_map = count_maps[c]
#
#         hp.fitsfunc.write_map(filename=path + filename, m=count_map, nest=False, coord="G", dtype=np.float64,
#                               overwrite=True)
#
#
#     # NOT FINISHED



def save_catalog(simulated_agns, simulated_pulsars, file_name: str):
    # XML FOR AGN

    root_agn = minidom.Document()

    xml_agn = root_agn.createElement('source_library')

    xml_agn.setAttribute('title', 'source library')

    root_agn.appendChild(xml_agn)

    agn_xml_writer(simulated_agns, root=root_agn, xml=xml_agn)

    # XML FOR PULSARS

    root_pulsar = minidom.Document()

    xml_pulsar = root_pulsar.createElement('source_library')

    xml_pulsar.setAttribute('title', 'source library')

    root_pulsar.appendChild(xml_pulsar)

    pulsar_xml_writer(simulated_pulsars, root=root_pulsar, xml=xml_pulsar)

    # FORMAT

    agn_xml_str = root_agn.toprettyxml()

    pulsar_xml_str = root_pulsar.toprettyxml()

    # SAVE

    # Create directory if it does not already exist
    Path(file_name).mkdir(parents=True, exist_ok=True)

    with open(file_name + "agns.xml", "w") as f:
        f.write(agn_xml_str)

    with open(file_name + "pulsars.xml", "w") as f:
        f.write(pulsar_xml_str)


def xml_parser(energy_bins, xml_file: str):

    # Read XML files to get latitude and longitude of each source (separate into AGN, pulsars, and background - if they
    # are in the same file), as well as the flux of the source

    docs = minidom.parse(xml_file)

    sources = docs.getElementsByTagName("source")

    num_bins = energy_bins.shape[0]

    coordinates = []
    fluxes = []

    # Remove diffuse sources - only processing point sources with this function
    sources = [sources[k] for k in range(len(sources)) if sources[k].getAttribute("type") != "DiffuseSource"]

    # Parse XML
    for source in sources:

        source_type = source.getAttribute("name")[:3]

        # PARSE SPATIAL PARAMETERS

        spatial_model = source.getElementsByTagName("spatialModel")[0]

        parameters = spatial_model.getElementsByTagName("parameter")

        coordinate = [0, 0]

        for param in parameters:
            name = param.getAttribute("name")

            if name == "RA":
                coordinate[0] = float(param.getAttribute("value"))
            else:
                coordinate[1] = float(param.getAttribute("value"))

        coordinates.append(coordinate)

        # PARSE SPECTRAL FEATURES AND CALCULATE FLUX

        spectral_model = source.getElementsByTagName("spectrum")[0]

        spectral_parameter_dictionary = dict()

        # Get spectral parameters

        parameters = spectral_model.getElementsByTagName("parameter")

        for param in parameters:
            name = param.getAttribute("name")

            scale = float(param.getAttribute("scale"))

            value = float(param.getAttribute("value"))

            actual_value = scale * value

            spectral_parameter_dictionary[name] = actual_value

        binned_fluxes = []

        if source_type == "AGN":

            # Parse spectral parameters

            # For each energy interval, calculate corresponding flux
            for f in range(num_bins - 1):

                # N.B. Integral photon flux is not the same as energy flux
                flux = integral_photon_flux_agn(pivot_energy=spectral_parameter_dictionary["Eb"],
                                                flux_density=spectral_parameter_dictionary["norm"],
                                                spectral_slope=spectral_parameter_dictionary["alpha"],
                                                curvature=spectral_parameter_dictionary["beta"],
                                                min_energy=energy_bins[f],
                                                max_energy=energy_bins[f + 1])

                binned_fluxes.append(flux)

        elif source_type == "PSR":

            # For each energy interval, calculate corresponding flux
            for f in range(num_bins - 1):

                # N.B. Integral photon flux is not the same as energy flux
                flux = integral_photon_flux_pulsar(pivot_energy=spectral_parameter_dictionary["Scale"],
                                          flux_density=spectral_parameter_dictionary["Prefactor"],
                                          spectral_slope=spectral_parameter_dictionary["Index1"],
                                          exponential_index=spectral_parameter_dictionary["Index2"],
                                          exponential_factor=spectral_parameter_dictionary["Expfactor"],
                                          min_energy=energy_bins[f], max_energy=energy_bins[f + 1])

                binned_fluxes.append(flux)

        else:

            # Unrecognised point source type - allows for debugging when adding in new source types to simulation
            raise TypeError("Cannot recognise source type {}".format(source_type))

        fluxes.append(binned_fluxes)

    # Have coordinates in format [RA, DEC] - need to convert them to Lat-lon

    # Convert coordinates to np array
    coordinates = np.array(coordinates)

    coordinates = SkyCoord(ra=coordinates[:, 0] * u.degree, dec=coordinates[:, 1] * u.degree, frame='icrs').galactic

    coordinates = np.array([coordinates.l.value, coordinates.b.value]).T

    # Get coordinates into numpy array then separate into list of lats and lons
    # coordinates = np.array([[c.l.value, c.b.value] for c in coordinates])

    # Convert fluxes to numpy
    fluxes = np.array(fluxes)

    return coordinates, fluxes


# REFERENCES

# Accessing Numpy Columns - https://stackoverflow.com/questions/8386675/extracting-specific-columns-in-numpy-array
# Accessing XML Tags - https://stackoverflow.com/questions/32387528/getting-list-of-tags-from-python-minidom-xml
# Astropy Documentation - https://docs.astropy.org/en/stable/
# Converting Coordinates - https://cloudlessnights.com/project/python-for-astronomy/convert-galactic-coordinates-to-ra-
# dec-and-alt-az-with-astropy/
# Converting Binary Strings - https://stackoverflow.com/questions/17615414/how-to-convert-binary-string-to-normal-string
# -in-python3
# Creating Dictionaries - https://stackoverflow.com/questions/8424942/creating-a-new-dictionary-in-python
# Creating XML Documents - https://www.geeksforgeeks.org/python/create-xml-documents-using-python/
# Example Fermitools XML Files - https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/xml_model_defs.html#logParabola
# Fermi Documention on XML Formatting - https://fermi.gsfc.nasa.gov/ssc/data/access/lat/BackgroundModels.html
# LAT IRF - https://fermi.gsfc.nasa.gov/ssc/data/analysis/lat_irfs/irf_overview.html
# Pandas Columns - https://stackoverflow.com/questions/20297332/how-do-i-retrieve-the-number-of-columns-in-a-pandas-data
# -frame
# Pandas Documentation - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html
# Pandas Iteration of Columns - https://stackoverflow.com/questions/28218698/how-to-iterate-over-columns-of-a-pandas-
# dataframe
# Parsing XML - https://www.geeksforgeeks.org/python/parse-xml-using-minidom-in-python/
# Parsing XML 2 - https://blog.pythonlibrary.org/2010/11/12/python-parsing-xml-with-minidom/
# RA-Dec Units - https://www.reddit.com/r/Astronomy/comments/1fkv3nv/how_do_i_convert_from_hhmmss_to_degrees/
# Source Descriptions - https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/source_models.html#LogParabola
# Strip Function - https://stackoverflow.com/questions/8270092/remove-all-whitespace-in-a-string
