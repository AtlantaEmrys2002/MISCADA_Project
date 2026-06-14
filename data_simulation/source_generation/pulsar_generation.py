from astropy import units as u
from . pulsar_spectral_parameters import energy_flux_pulsar
import numpy as np
from scipy.stats import Mixture, Normal


def pulsar_generator(pulsar_stats, energy_flux_low, energy_flux_high, sigma_1, sigma_2):

    # sigma_1 and sigma_2 are the fitted standard deviations of Gaussian distribution

    (mean_log_Gamma_pulsars, std_log_Gamma_pulsars, b_values, len_pulsars, mean_log_a_pulsars, std_log_a_pulsars,
     mean_log_flux_density_pulsars, std_log_flux_density_pulsars, mean_log_pivot_energy_pulsars,
     std_log_pivot_energy_pulsars, latitudes_pulsars) = pulsar_stats

    unique_values = np.unique_all(b_values)
    choice_values = unique_values.values
    new_weights = unique_values.counts / len_pulsars

    while True:

        # SPECTRAL PARAMETERS

        # Generate new pivot energies - in ID8, they randomly select pivot energies from a Gaussian distribution.
        # However, the distribution of pivot energies in the 4FGL follows log-normal more precise (CHECK THIS IS TRUE
        # FOR PULSARS_
        pivot_energy = np.random.lognormal(mean=mean_log_pivot_energy_pulsars, sigma=std_log_pivot_energy_pulsars,
                                             size=1)[0]

        # Generate new flux densities - log-normal for flux densities
        flux_density = np.random.lognormal(mean=mean_log_flux_density_pulsars, sigma=std_log_flux_density_pulsars,
                                           size=1)[0]

        # Gaussian recommended in ID8 - AT THE MOMENT - may change to log-normal

        # Generate new spectral slopes (Gammas) - CHANGED FROM GAUSSIAN TO LOG-NORMAL BASED ON CHI SQUARED GOODNESS OF FIT
        # ANALYSIS (THIS IS A CHANGE FROM ID8, WHICH RECOMMENDED GAUSSIAN)
        # spectral_slope = np.random.normal(loc=mean_Gamma_pulsars, scale=std_Gamma_pulsars, size=1)[0]
        spectral_slope = np.random.lognormal(mean=mean_log_Gamma_pulsars, sigma=std_log_Gamma_pulsars, size=1)[0]

        # Generate new exponential factors (as)
        exponential_factor = np.random.lognormal(mean=mean_log_a_pulsars, sigma=std_log_a_pulsars, size=1)[0]

        # Generate new exponential indices (bs) - changed to random choice instead of Gaussian (which was recommended in
        # ID8)
        exponential_index = np.random.choice(choice_values, p=new_weights)

        # Generate energy fluxes
        energy_flux = energy_flux_pulsar(pivot_energy, flux_density, spectral_slope, exponential_index,
                                         exponential_factor)

        if (energy_flux >= energy_flux_low) and (energy_flux < energy_flux_high) and (~np.isnan(energy_flux)):

            # SPATIAL PARAMETERS

            # l - uniform distribution assumed
            longitude = np.random.uniform(low=0, high=2 * np.pi, size=1)[0]

            # b - double Gaussian - two overlapping sampled as one

            # Randomly sample pulsar latitudes from distribution created above
            X1 = Normal(mu=0, sigma=sigma_1)
            X2 = Normal(mu=0, sigma=sigma_2)

            # CHANGE WEIGHTS HERE TO REFLECT MSP VS YNG
            mixture = Mixture([X1, X2])

            latitude = mixture.sample(shape=(1, 1)).flatten()[0]

            return np.asarray([pivot_energy, flux_density, spectral_slope, exponential_index, exponential_factor,
                               energy_flux, longitude, latitude])


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

    # Log-normal for Gamma even though says Gaussian in ID8
    log_Gammas = np.log(Gammas)

    mean_log_Gamma, std_log_Gamma = np.nanmean(log_Gammas), np.nanstd(log_Gammas, ddof=1)

    mean_b, std_b = np.nanmean(b_values), np.nanstd(b_values, ddof=1)

    log_a_values = np.log(a_values)

    # Log-normal for a even if ID8 says Gaussian
    mean_log_a, std_log_a = np.nanmean(log_a_values), np.nanstd(log_a_values, ddof=1)

    mean_log_flux_density, std_log_flux_density = np.nanmean(log_flux_densities), np.nanstd(log_flux_densities)

    # Log-normal for pivot energy even though ID8 says Gaussian
    log_pivot_energies = np.log(pivot_energies)

    mean_log_pivot_energy, std_log_pivot_energy = np.nanmean(log_pivot_energies), np.nanstd(log_pivot_energies, ddof=1)

    pulsar_latitudes = pulsars['GLAT']

    return (mean_log_Gamma, std_log_Gamma, b_values, len(pulsars), mean_log_a, std_log_a, mean_log_flux_density,
            std_log_flux_density, mean_log_pivot_energy, std_log_pivot_energy, pulsar_latitudes)