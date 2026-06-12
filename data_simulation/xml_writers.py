from astropy import units as u
from astropy.coordinates import SkyCoord
from xml.dom import minidom


def agn_xml_writer(sources):

    # CREATE DOCUMENT

    root = minidom.Document()

    xml = root.createElement('source_library')

    xml.setAttribute('title', 'source library')

    root.appendChild(xml)

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

    # FORMAT

    xml_str = root.toprettyxml(indent="\t")

    # SAVE

    save_path_file = "agns.xml"

    with open(save_path_file, "w") as f:
        f.write(xml_str)


def pulsar_xml_writer(sources):

    # CREATE DOCUMENT

    root = minidom.Document()

    xml = root.createElement('source_library')

    xml.setAttribute('title', 'source library')

    root.appendChild(xml)

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

    xml_str = root.toprettyxml(indent="\t")

    save_path_file = "pulsars.xml"

    with open(save_path_file, "w") as f:
        f.write(xml_str)
