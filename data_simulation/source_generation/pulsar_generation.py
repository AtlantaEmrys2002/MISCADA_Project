from astropy import units as u
from . pulsar_spectral_parameters import energy_flux_pulsar
import numpy as np
from . utils import split_normal
from scipy.optimize import curve_fit
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

    log_a_values = np.log(a_values)

    # Log-normal for values of a even if ID8 says Gaussian
    mean_log_a, std_log_a = np.nanmean(log_a_values), np.nanstd(log_a_values, ddof=1)

    mean_log_flux_density, std_log_flux_density = np.nanmean(log_flux_densities), np.nanstd(log_flux_densities)

    # Log-normal for pivot energy even though ID8 says Gaussian
    log_pivot_energies = np.log(pivot_energies)

    mean_log_pivot_energy, std_log_pivot_energy = np.nanmean(log_pivot_energies), np.nanstd(log_pivot_energies, ddof=1)

    pulsar_latitudes = pulsars['GLAT']

    return (mean_log_Gamma, std_log_Gamma, b_values, len(pulsars), mean_log_a, std_log_a, mean_log_flux_density,
            std_log_flux_density, mean_log_pivot_energy, std_log_pivot_energy, pulsar_latitudes)


def generate_mock_pulsar_catalog(pulsar_stats, num_pulsars=350):

    (mean_log_Gamma_pulsars, std_log_Gamma_pulsars, b_values, len_pulsars, mean_log_a_pulsars, std_log_a_pulsars,
     mean_log_flux_density_pulsars, std_log_flux_density_pulsars, mean_log_pivot_energy_pulsars,
     std_log_pivot_energy_pulsars, latitudes_pulsars) = pulsar_stats
    #
    # # SPECTRAL PARAMETERS
    #
    # # Generate new pivot energies - in ID8, they randomly select pivot energies from a Gaussian distribution. However,
    # # the distribution of pivot energies in the 4FGL follows log-normal more precise (CHECK THIS IS TRUE FOR PULSARS_
    # pivot_energies = np.random.lognormal(mean=mean_log_pivot_energy_pulsars, sigma=std_log_pivot_energy_pulsars,
    #                                      size=num_pulsars)
    #
    # # Generate new flux densities - log-normal for flux densities
    # flux_densities = np.random.lognormal(mean=mean_log_flux_density_pulsars, sigma=std_log_flux_density_pulsars,
    #                                      size=num_pulsars)
    #
    # # Generate new spectral slopes (Gammas) - changed to log-normal for ID8 gaussian
    # spectral_slopes = np.random.lognormal(mean=mean_log_Gamma_pulsars, sigma=std_log_Gamma_pulsars, size=num_pulsars)
    #
    # # Generate new exponential factors (as) - log-normal (CHANGED FROM ID8, which used normal/Gaussian BASED ON RESULTS
    # # FOUND IN analysing_pulsar_parameters() function)
    # exponential_factors = np.random.lognormal(mean=mean_log_a_pulsars, sigma=std_log_a_pulsars, size=num_pulsars)
    #
    # # Generate new exponential indices (bs) by selecting from values available (instead of Gaussian recommended by ID8)
    # unique_values = np.unique_all(b_values)
    # choice_values = unique_values.values
    #
    # # Divide by number of pulsars in 4FGL to get probabilities
    # new_weights = unique_values.counts / len_pulsars
    #
    # exponential_indices = np.random.choice(choice_values, p=new_weights, size=num_pulsars)
    #
    # # Combine into one array
    # parameters = np.stack((pivot_energies, flux_densities, spectral_slopes, exponential_indices, exponential_factors), axis=-1)
    #
    # # Energy fluxes
    # energy_fluxes = np.fromiter((energy_flux_pulsar(x[0], x[1], x[2], x[3], x[4]) for x in parameters), np.float64)
    #
    # # Select rows with valid energy fluxes
    # mask = ~np.isnan(energy_fluxes)
    # energy_fluxes = energy_fluxes[mask]
    # parameters = parameters[mask]
    #
    # # Combine two arrays to create mock catalog's spectral parameters
    # parameters = np.concatenate((parameters, np.array([energy_fluxes]).T), axis=1)
    #
    # # SPATIAL PARAMETERS
    #
    # # l - uniform distribution assumed
    # galactic_longitudes = np.random.uniform(low=0, high=2 * np.pi, size=len(parameters))

    # b - double Gaussian - two overlapping sampled as one
    counts, bins = np.histogram(latitudes_pulsars.value, bins=100, density=True)

    bin_width = np.abs(bins[1] - bins[0])

    start_value = bins[0] + (bin_width / 2)

    x_values = [start_value + (k * bin_width) for k in range(len(bins) - 1)]

    popt, _ = curve_fit(f=split_normal, xdata=np.asarray(x_values), ydata=np.asarray(counts), bounds=([0, 0], [90, 90]))

    # # Randomly sample pulsar latitudes from distribution created above
    # X1 = Normal(mu=0, sigma=popt[0])
    # X2 = Normal(mu=0, sigma=popt[1])
    #
    # # CHANGE WEIGHTS HERE TO REFLECT MSP VS YNG
    #
    # mixture = Mixture([X1, X2])
    #
    # galactic_latitudes = mixture.sample(shape=(num_pulsars, 1)).flatten()
    #
    # # Combine two arrays to create mock catalog's spatial parameters
    # parameters = np.concatenate((parameters, np.array([galactic_longitudes]).T), axis=1)
    #
    # # Combine two arrays to create mock catalog's spatial parameters
    # parameters = np.concatenate((parameters, np.array([galactic_latitudes]).T), axis=1)


    parameters = []

    for x in range(num_pulsars):
        new_source = pulsar_generator(pulsar_stats, energy_flux_low=0, energy_flux_high=1000, sigma_1=popt[0],
                                      sigma_2=popt[1])
        parameters.append(np.array(new_source))

    parameters = np.array(parameters)



    # CUTOFF THRESHOLD AND LUMINOSITY FUNCTION CHECK

    # Cut at detection threshold

    # CHECK THIS AND ADD BACK IN - NEED CUT OFF THRESHOLD!!!!!!!!!!!!!!!!!!!
    #
    # parameters = parameters[~(parameters[:, 5] <= np.float64(1.0 * 10 ** (-12))), :]
    #
    # # ADD MORE SOURCES - FOLLOW LUMINOSITY FUNCTION OF 4FGL - ENSURE ENOUGH
    #
    # extra_sources = []
    #
    # for x in range(extra):
    #
    #     print(x + 1)
    #
    #     # CHANGED TO 1.0 * 10 ** -12 as recommended by 4FGL DR4 (ID22) paper for outside galactic plane
    #
    #     new_source = pulsar_generation(pulsar_stats, 1.0 * 10 ** (-12), 1000, sigma_1=popt[0], sigma_2=popt[1])
    #
    #     extra_sources.append(new_source)
    #
    # parameters = np.vstack((parameters, np.asarray(extra_sources)))
    #
    # # FAINT SOURCE FLAT EXTRAPOLATION
    #
    # # Flat extrapolation of AGN - assume constant below given threshold (not Gaussian)
    #
    # # Bin data and take average of first three
    # bin_edges = 10 ** np.linspace(-14, -9, 50)
    # counts, _ = np.histogram(parameters[:, 5], bins=bin_edges)
    #
    # # Take average number of sources of first five bins for flat extrapolation
    # first_non_empty_bin = np.nonzero(counts)[0][0]
    # mean_counts_per_bin, std_counts_per_bin = (np.mean(counts[first_non_empty_bin: first_non_empty_bin + 10]),
    #                                            np.std(counts[first_non_empty_bin: first_non_empty_bin + 10], ddof=1))
    #
    # faint_sources = []
    #
    # our_threshold = np.argwhere(bin_edges >= 3.4 * 10 ** (-13))[0][0]
    #
    # for x in range(our_threshold, first_non_empty_bin):
    #
    #     # Generate number of sources in bin - approximately flat/same as bins at peak
    #     counts_per_bin_flat_extrapolation = round(np.random.normal(loc=mean_counts_per_bin, scale=std_counts_per_bin,
    #                                                                size=1)[0])
    #
    #     for k in range(counts_per_bin_flat_extrapolation):
    #         new_source = pulsar_generation(pulsar_stats, bin_edges[x], bin_edges[x + 1], sigma_1=popt[0], sigma_2=popt[1])
    #         faint_sources.append(new_source)
    #
    # faint_sources = np.asarray(faint_sources)
    #
    # parameters = np.vstack((parameters, faint_sources))

    # CHECK LUMIN FUNCTION - NOT FINISHED!!!!!!

    return parameters
