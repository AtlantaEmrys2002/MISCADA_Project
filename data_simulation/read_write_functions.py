from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable
import numpy as np
from xml.dom import minidom, Node


# def agn_xml_writer(sources, save_path_file):
#
#     # CREATE DOCUMENT
#
#     root = minidom.Document()
#
#     xml = root.createElement('source_library')
#
#     xml.setAttribute('title', 'source library')
#
#     root.appendChild(xml)
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
#                 param.setAttribute("scale", "-1.0")
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
#         ra_dec = SkyCoord(l=sources[k][4] * u.rad, b=sources[k][5] * u.deg, frame='galactic').transform_to('icrs')
#         ra.setAttribute("value", str(ra_dec.ra.to_value(u.degree)))
#         dec.setAttribute("value", str(ra_dec.dec.to_value(u.degree)))
#
#         spatial.appendChild(ra)
#         spatial.appendChild(dec)
#
#         source.appendChild(spatial)
#
#     # FORMAT
#
#     xml_str = root.toprettyxml(indent="\t")
#
#     # SAVE
#
#     with open(save_path_file, "w") as f:
#         f.write(xml_str)


def agn_xml_writer(sources, root, xml):

    # CREATE DOCUMENT

    # root = minidom.Document()
    #
    # xml = root.createElement('source_library')
    #
    # xml.setAttribute('title', 'source library')
    #
    # root.appendChild(xml)

    # ADD SOURCES

    # Ranges are taken from https://git.io/JO5FP - i.e. recommended by ID8
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

        # Convert galactic coordinates to equatorial
        ra_dec = SkyCoord(l=sources[k][4] * u.rad, b=sources[k][5] * u.deg, frame='galactic').transform_to('icrs')
        ra.setAttribute("value", str(ra_dec.ra.to_value(u.degree)))
        dec.setAttribute("value", str(ra_dec.dec.to_value(u.degree)))

        spatial.appendChild(ra)
        spatial.appendChild(dec)

        source.appendChild(spatial)

    # # FORMAT
    #
    # xml_str = root.toprettyxml(indent="\t")
    #
    # # SAVE
    #
    # with open(save_path_file, "a") as f:
    #     f.write(xml_str)



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

            # print(child)

    node.normalize()

# def pulsar_xml_writer(sources, save_path_file: str):
#
#     # CREATE DOCUMENT
#
#     root = minidom.Document()
#
#     xml = root.createElement("source_library")
#
#     xml.setAttribute('title', 'source library')
#
#     root.appendChild(xml)
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
#         ra_dec = SkyCoord(l=sources[k][6] * u.rad, b=sources[k][7] * u.deg, frame='galactic').transform_to('icrs')
#         ra.setAttribute("value", str(ra_dec.ra.to_value(u.degree)))
#         dec.setAttribute("value", str(ra_dec.dec.to_value(u.degree)))
#
#         spatial.appendChild(ra)
#         spatial.appendChild(dec)
#
#         source.appendChild(spatial)
#
#     # SAVE
#
#     xml_str = root.toprettyxml(indent="\t")
#
#     with open(save_path_file, "w") as f:
#         f.write(xml_str)


def pulsar_xml_writer(sources, root, xml):

    # # CREATE DOCUMENT
    #
    # root = minidom.Document()
    #
    # xml = root.createElement("source_library")
    #
    # xml.setAttribute('title', 'source library')
    #
    # root.appendChild(xml)

    # PARAMETERS

    pulsar_parameters = ["Prefactor", "Index1", "Scale", "Expfactor", "Index2"]

    # Ranges are taken from https://git.io/JO5FP - i.e. recommended by ID8
    ranges_of_pulsar_parameters = [("0.00000001", "1000000000.0"), ("-50000.0", "5000.0"),
                                   ("-3000000.0", "3000000000000.0"), ("-100000000", "1000000"), ("0", "20")]

    for k in range(len(sources)):

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

        # Convert galactic coordinates to equatorial

        ra_dec = SkyCoord(l=sources[k][6] * u.rad, b=sources[k][7] * u.deg, frame='galactic').transform_to('icrs')
        ra.setAttribute("value", str(ra_dec.ra.to_value(u.degree)))
        dec.setAttribute("value", str(ra_dec.dec.to_value(u.degree)))

        spatial.appendChild(ra)
        spatial.appendChild(dec)

        source.appendChild(spatial)

    # SAVE

    # xml_str = root.toprettyxml(indent="\t")
    #
    # with open(save_path_file, "w") as f:
    #     f.write(xml_str)


def save_results(simulated_agns, simulated_pulsars, file_name="./simulated_data/sources.xml"):

    # CREATE DOCUMENT

    root = minidom.Document()

    xml = root.createElement('source_library')

    xml.setAttribute('title', 'source library')

    root.appendChild(xml)

    # BACKGROUND SOURCES
    background_models = minidom.parse("./simulated_data/background.xml").getElementsByTagName('source')

    for model in background_models:

        cleanEmptyTextNodes(model)

        xml.appendChild(model)

    # XML for AGN sources
    agn_xml_writer(simulated_agns, root=root, xml=xml)

    # XML for pulsar sources
    pulsar_xml_writer(simulated_pulsars, root=root, xml=xml)

    # FORMAT

    xml_str = root.toprettyxml(indent="\t")

    # SAVE

    with open(file_name, "w") as f:
        f.write(xml_str)


# REFERENCES

# Astropy Documentation - https://docs.astropy.org/en/stable/
