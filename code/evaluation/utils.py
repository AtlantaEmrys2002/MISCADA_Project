from astropy.table import QTable
import copy
import numpy as np
from scipy.integrate import quad
import warnings


def agn_photon_flux(E, E_0, F_0, alpha, beta):

    division = E/E_0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exponent = - alpha - (beta * np.log(division))

    dF_dE = F_0 * np.power(division, exponent)

    return dF_dE


def get_catalog_data(catalog_file: str):
    catalog = QTable.read(catalog_file, format='fits', hdu=1)

    # Select relevant columns
    columns = ("Source_Name", "Pivot_Energy", "LP_Flux_Density", "PLEC_Flux_Density", "LP_Index", "LP_beta",
               "PLEC_IndexS", "PLEC_Exp_Index", "PLEC_ExpfactorS", "CLASS1", "RAJ2000", "DEJ2000")

    catalog = catalog[columns]

    # Reformat CLASS1 column - remove empty spaces and make all lower case
    catalog["CLASS1"] = np.asarray([k.decode('utf-8').strip().lower() for k in catalog["CLASS1"].value.filled('-')])

    # Reformat Source_name column - remove empty spaces and make all lower case
    catalog["Source_Name"] = np.asarray([k.decode('utf-8').strip().lower()[5:] for k in catalog["Source_Name"].value])

    # Select all rows that describe pulsars
    pulsar_mask = (catalog["CLASS1"] == "psr")

    # Select all rows that describe AGN
    agn_mask = np.isin(catalog["CLASS1"].data, np.array(["bcu", "sey", "ssrq", "bll", "fsrq", "rdg", "nlsy1", "agn"]))

    # Delete unnecessary column
    catalog.remove_column("CLASS1")

    agn_data = catalog[agn_mask].copy()
    pulsar_data = catalog[pulsar_mask].copy()

    agn_data = (agn_data["Source_Name", "LP_Flux_Density", "Pivot_Energy", "LP_Index", "LP_beta", "RAJ2000", "DEJ2000"].
                to_pandas())

    pulsar_data = pulsar_data[
        ("Source_Name", "PLEC_Flux_Density", "Pivot_Energy", "PLEC_IndexS", "PLEC_Exp_Index", "PLEC_ExpfactorS",
         "RAJ2000", "DEJ2000")].to_pandas()

    # agn_pivot_energies = agn_data["Pivot_Energy"].to_numpy()
    # agn_flux_densities = agn_data["LP_Flux_Density"].to_numpy()
    # agn_spectral_slopes = agn_data["LP_Index"].to_numpy()  # alpha
    # agn_curvatures = agn_data["LP_beta"].to_numpy()  # beta
    #
    # pulsar_pivot_energies = pulsar_data["Pivot_Energy"].to_numpy()
    # pulsar_flux_densities = pulsar_data["PLEC_Flux_Density"].to_numpy()
    # pulsar_spectral_slopes = pulsar_data["PLEC_IndexS"].to_numpy()  # gamma
    # pulsar_exponential_indices = pulsar_data["PLEC_Exp_Index"].to_numpy()
    # pulsar_exponential_factors = pulsar_data["PLEC_ExpfactorS"].to_numpy()
    #
    # agn_integral_photon_fluxes = np.array([integral_photon_flux_agn(pivot_energy=agn_pivot_energies[s],
    #                                                                 flux_density=agn_flux_densities[s],
    #                                                                 spectral_slope=agn_spectral_slopes[s],
    #                                                                 curvature=agn_curvatures[s], min_energy=300.,
    #                                                                 max_energy=200000.)
    #                                        for s in range(len(agn_pivot_energies))])
    #
    # pulsar_integral_photon_fluxes = np.array([integral_photon_flux_pulsar(pivot_energy=pulsar_pivot_energies[s],
    #                                                                       flux_density=pulsar_flux_densities[s],
    #                                                                       spectral_slope=pulsar_spectral_slopes[s],
    #                                                                       exponential_index=
    #                                                                       pulsar_exponential_indices[s],
    #                                                                       exponential_factor=
    #                                                                       pulsar_exponential_factors[s])
    #                                           for s in range(len(pulsar_pivot_energies))])

    agn_ra = agn_data["RAJ2000"].to_numpy()
    agn_dec = agn_data["DEJ2000"].to_numpy()

    psr_ra = pulsar_data["RAJ2000"].to_numpy()
    psr_dec = pulsar_data["DEJ2000"].to_numpy()

    actual_source_locations_4fgl = np.vstack((np.vstack((agn_ra, agn_dec)).T, np.vstack((psr_ra, psr_dec)).T))

    actual_source_types = np.vstack((np.array([[1., 0., 0.,] for _ in range(agn_ra.shape[0])]),
                                     np.array([[0., 1., 0.] for _ in range(psr_ra.shape[0])])))

    return actual_source_locations_4fgl, actual_source_types


def get_classified_patches(predicted_locations_in_real_data_raw):
    # see if it would have been classified or not (i.e. if it was too close to the edge. If it was too close to
    # edge of 64 x 64 image, then remove.

    predicted_locations_in_real_data = []

    for p in range(len(predicted_locations_in_real_data_raw)):

        loc_in_patch = predicted_locations_in_real_data_raw[p]

        if loc_in_patch.shape[0] != 0:

            xs = loc_in_patch[:, 0]
            ys = loc_in_patch[:, 1]

            mask = np.logical_not(((xs - 3) < 0) | ((xs + 4) > 63) | ((ys - 3) < 0) | ((ys + 4) > 63))

            predicted_locations_in_real_data.append(copy.deepcopy(predicted_locations_in_real_data_raw[p][mask]))

        else:

            predicted_locations_in_real_data.append(np.array([]))

    return predicted_locations_in_real_data


def integral_photon_flux_agn(pivot_energy, flux_density, spectral_slope, curvature, min_energy=100.0, max_energy=100000.0):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        energy = quad(agn_photon_flux, min_energy, max_energy, args=(pivot_energy, flux_density, spectral_slope, curvature))[0]

    return energy


def integral_photon_flux_pulsar(pivot_energy, flux_density, spectral_slope, exponential_index, exponential_factor,
                       min_energy=100.0, max_energy=100000.0):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Integrate over 0.1 - 100 GeV (100 - 100000 MeV) by default
        energy = quad(pulsar_photon_flux, min_energy, max_energy, args=(flux_density, pivot_energy, spectral_slope,
                                                                        exponential_factor, exponential_index))[0]

    return energy


def pulsar_photon_flux(E, F_0, E_0, Gamma, a, b):

    division = E / E_0

    exponent = a * (np.power(E_0, b) - np.power(E, b))

    dF_dE = F_0 * np.power(division, -Gamma) * np.exp(exponent)

    return dF_dE
