from astropy.io import fits
import healpy as hp
import numpy as np
from scipy.optimize import curve_fit
from . utils import dual_function


def fit_diffuse_source_psf(roi_count_map, nside=512):

    # N.B. The energy bounds returned by gtmodel are in KeV and not in eV or MeV
    # As we are integrating over them it does not matter.

    # CHECK UNDERSTANDING - I think you don't have to integrate as already over energy
    # ranges I want and don't have to "energy average"

    # Creating PSF for diffuse sources
    # N.B. nside does not have to match the final count map nside

    # Change to NSIDE OF convolved diffuse (see https://arxiv.org/html/2410.12951v2)

    # with fits.open("/Volumes/T7/project_data/real_data/diffuse_psf_roi/roi_filted_source_map.fits") as hdul:
    #
    #     print(hdul.info())

    import matplotlib.pyplot as plt

    with fits.open(roi_count_map) as hdul:

        # FOR LABELLING ONLY - CONVERT FROM KeV to MeV
        energy_bins = [np.float64(k[1]) / 1000 for k in np.array(hdul[2].data)]

        energy_bins.append(hdul[2].data[-1][2])

        energy_bins = [np.round(k) for k in energy_bins]

        # Centre of data
        midpoint = hdul[0].data[0].shape[0] // 2

        # lmax = 3 * nside

        num_bins = hdul[0].data.shape[0]

        psfs = []

        for b in range(num_bins):

            counts = hdul[0].data[b]

            # Average x and y axis through point
            x_axis = counts[midpoint][midpoint:]
            y_axis = counts[:, midpoint][midpoint:]

            values = (x_axis + y_axis) / 2

            # NORMALISE BY MAXIMUM ATTAINED VALUE
            values /= np.max(values)

            # CONVERT FROM PIXELS TO RADIANS FROM CENTRE
            side_length_pixel = np.sqrt((4 * np.pi) / (12 * nside ** 2))

            # IN DEGREES
            radius = side_length_pixel * np.array(list(range(len(x_axis))))

            minus_included = list(range(-len(x_axis), len(x_axis)))

            # TO RADIANS
            radius *= (np.pi / 180)







            # NEED TO FIGURE OUT IF WE SHOULD HAVE WINDOW FUNCTION AS A COMPLETE BElL CURVE WITH
            # THETA FROM -x to x CENTRED AT ZERO OR JUST FROM 0 TO RADIUS


            beam = hp.sphtfunc.bl2beam(bl=values, theta=radius)

            psfs.append(beam)

            # CHECK BY APPLYING BEAM TO IMAGE WITH SINGLE PIXEL AT CENTER EQUAL TO MAXIMUM VALUE (BEFORE NORMALISATION)
            # AT THAT INTERVAL

            npix = np.array([x for x in range(hp.pixelfunc.nside2npix(nside=nside))])

            pix = hp.pixelfunc.ang2pix(nside, theta=90, phi=40, lonlat=True)

            test = np.zeros_like(npix)

            test[pix] = (energy_bins[b] + energy_bins[b + 1]) / 2

            test_smoothed = hp.sphtfunc.smoothing(map_in=test, beam_window=beam)

            hp.visufunc.mollview(map=test_smoothed)

            plt.show()

            plt.close()

    return psfs

            # plt.plot(range(0, len(x_axis)), values, linestyle='--', label="{}-{} MeV".format(energy_bins[b], energy_bins[b + 1]))



        # plt.xlabel("\\theta [pixels]")
        # plt.ylabel("PSF(\\theta) [per pixel]")
        #
        # plt.xlim(0, 30)
        # # plt.ylim(0, 600)
        #
        # plt.legend()
        #
        # plt.show()




        # lmax = 3 * nside  # chose 3 based on above paper
        #
        # # Can plot below with plt.imshow(counts
        # counts = hdul[0].data
        #
        # midpoint = counts.shape[0] // 2
        #
        # x_axis = counts[midpoint]
        # y_axis = counts[:, midpoint]
        #
        # # x and y axis to form a cross
        # values = (x_axis + y_axis) / 2
        #
        # max_value = np.max(values)
        #
        # # Closest value to half the intensity - https://stackoverflow.com/questions/8914491/finding-the-nearest-value-and-return-the-index-of-array-in-python
        # half_intensity = (np.abs(values - (max_value // 2))).argmin()
        #
        # # FWHM
        #
        # # Find the difference between half way along the axis and centre - 0.05 degrees represented by 1 pixel. Multiply by two as it goes across the mean
        #
        # # Approximation from CMB Estimation Paper - https://arxiv.org/html/2410.12951v2 - resolution how many degrees
        # # does side cover approximately
        # side_length_pixel = np.sqrt((4 * np.pi) / (12 * nside ** 2))
        #
        # degrees_difference = np.abs(midpoint - half_intensity) * side_length_pixel * 2
        #
        # # Convert to arcmin FROM degrees - NEW DOCUMENTATION SAYS RADIANS (PREVIOUS SAYS ARCMIN)
        # degrees_difference *= (np.pi / 180)
        #
        # beam = hp.sphtfunc.gauss_beam(fwhm=degrees_difference, lmax=lmax)
        #
        # # print(beam)
        #
        # # thetas = np.linspace(0, lmax, num=len(values)//2) * side_length_pixel
        #
        # import matplotlib.pyplot as plt
        #
        # # to_integrate = (2 * np.pi * beam * thetas)
        # #
        # # theta_diff = thetas[1:] - thetas[:-1]
        # # func_diff = (to_integrate[1: ] + to_integrate[:-1])/2
        # #
        # # integral = np.sum(theta_diff * func_diff)
        #
        # # plt.plot(thetas, values[midpoint:] / (np.pi * thetas ** 2))
        #
        # # print(thetas)
        #
        # # plt.yscale("log")
        #
        # # plt.show()

        # return beam


def fit_point_source_psf(file_name):

    function_params = []

    # The constants included as arguments above were derived in
    # https://iopscience.iop.org/article/10.1088/0004-637X/765/1/54/pdf

    with fits.open(file_name) as hdul:

        thetas = np.array([k[0] for k in hdul["THETA"].data])

        psf_data = hdul["PSF"].data

        num_bins = len(psf_data)

        for b in range(num_bins):

            # Lowest energy (MeV) of this bin
            energy_value = psf_data[b][0]

            # PSF values dP/dOmega - probability to find event in solid angle dOmega at offset r from point source
            psf_values = np.array(psf_data[b][2])

            psf_values = scale_psf(psf_values, energy_value)

            probs = normalise_psf(thetas, psf_values)

            # Fit King function (Moffat distribution to values to create a probability density function)
            popt, _ = curve_fit(dual_function, thetas, probs, maxfev=10000)

            function_params.append(popt)

    return np.array(function_params)


def normalise_psf(thetas, psf_values):

    # How to normalise a function -
    # https://math.stackexchange.com/questions/4806473/forcing-a-function-to-integrate-to-1
    # How to integrate over solid angle for this specfic PSF -
    # https://gamma-astro-data-formats.readthedocs.io/en/v0.1/irfs/psf/index.html#psf-pdf

    # Apply this function AFTER energy scaling

    probs = ((2 * np.pi * thetas) ** 2) * psf_values

    # Integrate over probs
    approx_integral = np.sum(np.array(
        [((probs[k + 1] + probs[k]) / 2) * (thetas[k + 1] - thetas[k]) for k in range(len(psf_values) - 1)]))

    # Normalise such that the integral is 1
    probs /= approx_integral

    return probs


def scale_psf(psf_values, energy_bin, c_0=3.5, c_1=0.15, beta=0.8):

    # The constants included as arguments above were derived in
    # https://iopscience.iop.org/article/10.1088/0004-637X/765/1/54/pdf

    # Calculate energy scale factor
    scale_factor = np.sqrt(((c_0 * (energy_bin / 100) ** (-beta)) ** 2) + c_1)

    # Scale PSF values
    psf_values /= scale_factor

    return psf_values


# REFERENCES

# Adjusting Curve Fit Iterations - https://stackoverflow.com/questions/15831763/scipy-curvefit-runtimeerroroptimal-
# parameters-not-found-number-of-calls-to-fun
# Custom PDFs - https://math.stackexchange.com/questions/3614107/how-do-you-create-a-custom-probability-density-function
# -from-a-discrete-distribu
# Function Fits - https://stackoverflow.com/questions/68523795/fit-a-custom-function-in-python
# FWHM - https://stackoverflow.com/questions/8914491/finding-the-nearest-value-and-return-the-index-of-array-in-python
# FWHM 2 - https://en.wikipedia.org/wiki/Full_width_at_half_maximum
# Normalising Functions - https://math.stackexchange.com/questions/4806473/forcing-a-function-to-integrate-to-1
# PDF > 1 - https://math.stackexchange.com/questions/1720053/how-can-a-probability-density-function-pdf-be-greater-
# than-1
# PDF from Data - https://math.stackexchange.com/questions/2325565/is-it-possible-to-calculate-probability-density-
# function-from-a-data-set
# Radial Profiles - https://cxc.cfa.harvard.edu/ciao/why/radial_profile_correction.html
