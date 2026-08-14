"""
Main function for creating all-sky maps. Placed in separate file for ease of parallelization.
"""

from map_generation.healpix_maps import create_count_map, create_diffuse_source_map
from pathlib import Path
from read_write_functions import save_count_maps, xml_parser
from verification.visualisation import plot_all_sky_map


def create_all_count_maps(params: list) -> None:
    """
    Creates num_maps_per_catalog count maps for catalog catalog_id.
    """
    (catalog_id, num_maps_per_catalog, energy_bins, nside, exposure_maps, binned_point_source_psf_parameters,
     binned_diffuse_source_psf, infinite_statistics_galactic_diffuse_backgrounds,
     infinite_statistics_isotropic_background) = params

    catalog_file_location = "./simulated_data/catalogs/catalog_{}".format(catalog_id + 1)
    infinite_statistics_directory = "./simulated_data/infinite_count_maps/infinite_count_map_{}".format(
        catalog_id + 1)

    # Calculate coordinates and binned fluxes from each mock simulated catalog
    agn_coordinates, agn_binned_fluxes = xml_parser(energy_bins, xml_file=catalog_file_location + "/agns.xml")
    pulsar_coordinates, pulsar_binned_fluxes = (
        xml_parser(energy_bins, xml_file=catalog_file_location + "/pulsars.xml"))

    # CREATE COUNT MAPS

    for m in range(num_maps_per_catalog):

        print("Creating Count Map {}".format((catalog_id * num_maps_per_catalog) + m + 1))

        # i.e. sample infinite counts map and convolve with PSFs

        # Create background - convolve infinite statistics maps of isotropic and galactic backgrounds with diffuse
        # PSF, then scale with randomly-generated normalisation constant, and Poisson sample to create unique
        # background count map
        diffuse_source_background = create_diffuse_source_map(
            expected_counts_diffuse_background=infinite_statistics_galactic_diffuse_backgrounds,
            expected_counts_isotropic_background=infinite_statistics_isotropic_background,
            psfs=binned_diffuse_source_psf)

        # Create AGN count maps (with PSF convolution and Poisson sampling
        agn_point_source_map = (
            create_count_map(coordinates=agn_coordinates, exposure_maps=exposure_maps,
                             psf_parameters=binned_point_source_psf_parameters, fluxes=agn_binned_fluxes,
                             nside=nside,
                             infinite_stats_file=infinite_statistics_directory + "/agn_infinite_counts_{}.fits"))

        # Create pulsar count maps (with PSF convolution and Poisson sampling)
        pulsar_point_source_map = (
            create_count_map(coordinates=pulsar_coordinates, exposure_maps=exposure_maps,
                             psf_parameters=binned_point_source_psf_parameters, fluxes=pulsar_binned_fluxes,
                             nside=nside,
                             infinite_stats_file=infinite_statistics_directory + "/pulsar_infinite_counts_{}.fits"))

        # SAVE COUNT MAPS

        save_location = "./simulated_data/count_maps/skymap_{}".format((catalog_id * num_maps_per_catalog) + m + 1)

        # Create directory to store simulated count maps in if it does not already exist
        Path(save_location).mkdir(parents=True, exist_ok=True)

        save_count_maps(diffuse_source_background, save_file=save_location + "/background_{}.fits")
        save_count_maps(agn_point_source_map, save_file=save_location + "/agns_{}.fits")
        save_count_maps(pulsar_point_source_map, save_file=save_location + "/pulsars_{}.fits")

        # Save background
        if catalog_id == 0 and m == 0:
            plot_all_sky_map(healpix_maps=pulsar_point_source_map, energy_bins=energy_bins, title="Pulsar Count",
                             directory="./plots/all_sky_maps/", logarithmic=True)
            plot_all_sky_map(healpix_maps=agn_point_source_map, energy_bins=energy_bins, title="AGN Count",
                             directory="./plots/all_sky_maps/", logarithmic=True)
            plot_all_sky_map(healpix_maps=diffuse_source_background, energy_bins=energy_bins,
                             title="Background Count", directory="./plots/all_sky_maps/", logarithmic=True)

            actual_count_map = pulsar_point_source_map + agn_point_source_map + diffuse_source_background
            plot_all_sky_map(healpix_maps=actual_count_map, energy_bins=energy_bins,
                             title="Count", directory="./plots/all_sky_maps/", logarithmic=True)
