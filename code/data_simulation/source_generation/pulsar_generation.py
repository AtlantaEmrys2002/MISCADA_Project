# N.B. Cauchy distribution was suggested in ID25, but for exponential index b AND
# not latitude (which definitely does not make sense given our plots).
# I came to very different conclusions than ID25 and do not examine correlation at
# all in ID25. They do suggest a Gumbel for beta - look into fitting that.
# Basically, can compare ID25's suggestions with ID8s, but there is no justification
# for their choice and their choices do not make sense given my analysis


from astropy.table import QTable
from math import floor
import numpy as np
from . pulsar_spectral_parameters import energy_flux_pulsar, pulsar_flux_density
from scipy.stats import cauchy


def pulsar_generator(pulsar_stats, energy_flux_low, energy_flux_high):

    # sigma_1 and sigma_2 are the fitted standard deviations of Gaussian distribution

    (mean_log_Gamma_pulsars, std_log_Gamma_pulsars, mean_log_a_pulsars, std_log_a_pulsars,
     mean_log_flux_density_pulsars, std_log_flux_density_pulsars, mean_log_pivot_energy_pulsars,
     std_log_pivot_energy_pulsars, choice_values, new_weights) = pulsar_stats

    # # Generate new exponential indices (bs) by selecting from values available (instead of Gaussian recommended by ID8)
    # len_pulsars = len(b_values)
    #
    # unique_values = np.unique_all(b_values)
    # choice_values = unique_values.values
    #
    # # Divide by number of pulsars in 4FGL to get probabilities
    # new_weights = unique_values.counts / len_pulsars

    while True:

        # SPECTRAL PARAMETERS

        # Generate new pivot energies - in ID8, they randomly select pivot energies from a Gaussian distribution.
        # However, the distribution of pivot energies in the 4FGL follows log-normal more precise
        pivot_energy = np.random.lognormal(mean=mean_log_pivot_energy_pulsars, sigma=std_log_pivot_energy_pulsars)

        # Generate new flux densities - log-normal for flux densities was initially used then used cubic correlation
        # of log pivot energy and log flux density
        # flux_density = np.random.lognormal(mean=mean_log_flux_density_pulsars, sigma=std_log_flux_density_pulsars)

        flux_density = pulsar_flux_density(pivot_energy)[0]

        # Gaussian recommended in ID8 - AT THE MOMENT - may change to log-normal

        # Generate new spectral slopes (Gammas) - CHANGED FROM GAUSSIAN TO LOG-NORMAL BASED ON CHI SQUARED GOODNESS OF FIT
        # ANALYSIS (THIS IS A CHANGE FROM ID8, WHICH RECOMMENDED GAUSSIAN)
        # spectral_slope = np.random.normal(loc=mean_Gamma_pulsars, scale=std_Gamma_pulsars, size=1)[0]
        spectral_slope = np.random.lognormal(mean=mean_log_Gamma_pulsars, sigma=std_log_Gamma_pulsars)

        # Generate new exponential factors (as)
        exponential_factor = np.random.lognormal(mean=mean_log_a_pulsars, sigma=std_log_a_pulsars)

        # Generate new exponential indices (bs) - changed to random choice instead of Gaussian (which was recommended in
        # ID8)
        exponential_index = np.random.choice(choice_values, p=new_weights)

        # Generate energy fluxes
        energy_flux = energy_flux_pulsar(pivot_energy, flux_density, spectral_slope, exponential_index,
                                         exponential_factor)

        # Check energy flux in given range AND that the energy flux is valid
        if (energy_flux >= energy_flux_low) and (energy_flux < energy_flux_high) and (~np.isnan(energy_flux)):

            # SPATIAL PARAMETERS

            # l - uniform distribution assumed
            longitude = np.random.uniform(low=-(2 * np.pi), high=2 * np.pi)

            # b - double Gaussian - two overlapping sampled as one

            # Randomly sample pulsar latitudes from distribution created above
            latitude = cauchy(loc=0, scale=np.float64(0.018077045649988577)).rvs()

            # Clip to correct range
            latitude = np.clip(latitude, a_min=-np.pi / 2, a_max=np.pi / 2)

            return np.asarray([pivot_energy, flux_density, spectral_slope, exponential_index, exponential_factor,
                               energy_flux, longitude, latitude])


def luminosity_function_pulsar(catalog: str, detection_threshold):

    # Used to build luminosity function of the simulated AGNs

    # Read 4FGL Catalog
    catalog = QTable.read(catalog, format='fits', hdu=1)['CLASS1', 'Energy_Flux100']

    # Reformat columns
    catalog['CLASS1'] = np.asarray([k.decode('utf-8').strip().lower() for k in catalog['CLASS1'].value.filled('-')])

    # Select all rows that describe pulsars
    pulsar_mask = (catalog["CLASS1"] == "psr")

    pulsars = catalog[pulsar_mask]

    energy_fluxes_4fgl = pulsars['Energy_Flux100'].value

    # Number of sources in the lowest energy flux bin - different values for pulsars
    n_min = np.random.uniform(low=15, high=25)

    # The minimum energy flux of our generated sources is an order of magnitude less than the 4FGL
    our_threshold = detection_threshold / 10

    # Following method detailed in ID8

    # Bin 4FGL data
    min_bin_val = np.log10(np.min(energy_fluxes_4fgl))
    max_bin_val = np.log10(np.max(energy_fluxes_4fgl))

    # Changed number of bins here as only so many pulsars
    log_linspace = np.linspace(min_bin_val, max_bin_val, 10)

    bin_edges = 10 ** log_linspace
    counts, bin_intervals = np.histogram(energy_fluxes_4fgl, bins=bin_edges)

    # Calculate width of bins
    bin_width = log_linspace[1] - log_linspace[0]

    # FLAT EXTRAPOLATION TO FAINTER DETECTION THRESHOLD

    # Extend to one order of magnitude less than the detection threshold of the 4FGL (similar premise to ID8) -
    # assume constant below given threshold (not Gaussian)

    # Number of bins between current lowest energy bin and our faint source threshold
    num_extra_bins = floor((np.log10(bin_intervals[0]) - np.log10(our_threshold)) / bin_width)

    extra_intervals = [10 ** (np.log10(bin_intervals[0]) - (bin_width * x)) for x in range(num_extra_bins, 0, -1)]

    # Create extra bin intervals

    # Calculate number of random

    # Find bin with the most AGNs
    peak = np.argmax(counts)

    # For any bin to the right of the peak that has 0 or 1 expected counts, set to 2
    for k in range(peak, len(counts)):
        if counts[k] == 1 or counts[k] == 0:
            counts[k] = 2

    # Generate random numbers for number of energy flux bins to the right of the peak
    n_noise = list(np.random.uniform(low=0.8, high=1.3, size=len(bin_intervals) - 1 - peak))

    # Create some noise in energy bins greater than peak
    for k in range(peak, len(n_noise)):
        counts[k + peak] = counts[k + peak] * n_noise[k]

    bin_intervals = np.array(extra_intervals + list(bin_intervals))

    # Set number of counts equal to peak for original 4FGL bins to the left of the peak
    for k in range(0, peak):
        counts[k] = n_min

    counts = [n_min for _ in range(num_extra_bins)] + list(counts)

    # Convert counts to int
    counts = [int(k) for k in counts]

    peak = peak + num_extra_bins

    return np.array([counts])[0], np.array(bin_intervals), peak


def generate_mock_pulsar_catalog(catalog: str, pulsars, detection_threshold):

    # Select pivot energy values
    pivot_energies = pulsars['Pivot_Energy'].value

    # Select Gamma values and convert from masked to ordinary numpy array
    Gammas = pulsars['PLEC_IndexS'].data.filled(np.nan)

    # Select exponential indices
    b_values = pulsars['PLEC_Exp_Index'].data.filled(np.nan)

    # Select exponential factors - CHECK (SAYS IN UNITS OF MeV^-b BUT NEED with GeV)
    # Generate new exponential factors (as) - log-normal (CHANGED FROM ID8, which used normal/Gaussian BASED ON RESULTS
    # FOUND IN analysing_pulsar_parameters() function)
    a_values = pulsars['PLEC_ExpfactorS'].data

    # Select flux density values and convert from ph / (cm2 MeV s) to ph / (cm2 GeV s)
    flux_densities = pulsars['PLEC_Flux_Density'].value  # .to(u.ph / (u.cm * u.cm * u.GeV * u.s)).value
    log_flux_densities = np.log(flux_densities)

    # Log-normal for Gamma even though says Gaussian in ID8
    log_Gammas = np.log(Gammas)

    mean_log_Gamma_pulsars, std_log_Gamma_pulsars = np.nanmean(log_Gammas), np.nanstd(log_Gammas, ddof=1)

    log_a_values = np.log(a_values)

    # Log-normal for values of a even if ID8 says Gaussian
    mean_log_a_pulsars, std_log_a_pulsars = np.nanmean(log_a_values), np.nanstd(log_a_values, ddof=1)

    mean_log_flux_density_pulsars, std_log_flux_density_pulsars = (np.nanmean(log_flux_densities),
                                                                   np.nanstd(log_flux_densities))

    # Log-normal for pivot energy even though ID8 says Gaussian
    log_pivot_energies = np.log(pivot_energies)

    mean_log_pivot_energy_pulsars, std_log_pivot_energy_pulsars = (np.nanmean(log_pivot_energies),
                                                                   np.nanstd(log_pivot_energies, ddof=1))

    # Generate new exponential indices (bs) by selecting from values available (instead of Gaussian recommended by ID8)

    unique_values = np.unique_all(b_values)
    choice_values = unique_values.values

    # Divide by number of pulsars in 4FGL to get probabilities
    new_weights = unique_values.counts / len(b_values)

    cauchy_params = cauchy.fit(pulsars['GLAT'].value, floc=0)

    pulsar_stats = (mean_log_Gamma_pulsars, std_log_Gamma_pulsars, mean_log_a_pulsars, std_log_a_pulsars,
     mean_log_flux_density_pulsars, std_log_flux_density_pulsars, mean_log_pivot_energy_pulsars,
     std_log_pivot_energy_pulsars, choice_values, new_weights)

    # Fit split-normal distribution that describes pulsar latitudes and pass the appropriate parameters to
    # pulsar_generator()

    # CREATE NEW SOURCES

    # Determine how many AGNs to generate based on 4FGL luminosity function
    target_counts, target_bin_intervals, target_peak = luminosity_function_pulsar(catalog=catalog,
                                                                                  detection_threshold=detection_threshold)

    actual_counts = np.zeros_like(target_counts)

    parameters = []

    while (actual_counts[0] < target_counts[0]) and np.any(np.less(actual_counts[target_peak:],
                                                                   target_counts[target_peak:])):


        # Due to uncomplimentary functionality - need max of intervals[:-1] - see reference to Digitize Error
        new_source = pulsar_generator(pulsar_stats, energy_flux_low=np.min(target_bin_intervals),
                                      energy_flux_high=np.max(target_bin_intervals[:-1]))

        idx = np.digitize(new_source[5], target_bin_intervals)

        if 0 < idx < len(target_bin_intervals):

            if actual_counts[idx] < target_counts[idx]:
                parameters.append(np.array(new_source))
                actual_counts[idx] += 1

                print("PULSAR: {}".format(len(parameters)))

        else:

            parameters.append(np.array(new_source))
            actual_counts[idx] += 1

            print("PULSAR: {}".format(len(parameters)))

    parameters = np.array(parameters)

    return parameters

# REFERENCES

# Exclude NaNs - https://stackoverflow.com/questions/17126543/numpy-array-get-the-subset-slice-of-an-array-which-is-not
# -nan
