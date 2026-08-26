import argparse
from novel import novel_source_extraction_algorithms
from pathlib import Path
from read_write_functions import read_patches, read_real_data

if __name__ == "__main__":
    # PROCESS USER INPUT

    parser = argparse.ArgumentParser(description="Reads in a collection of specified patches and formats them to be "
                                                 "passed to benchmark algorithms. These algorithms are then applied and"
                                                 "the parameters of the 'best' versions of them are saved.")

    parser.add_argument("--patch_location", required=True, type=str, help="The directory in which the"
                                                                          "patch data is stored.")

    parser.add_argument("--num_patches", required=True, type=int, help="The number of patches to read from"
                                                                       "the specified patch directory.")

    parser.add_argument("--save_directory", required=True, type=str, help="Location in which to save the "
                                                                          "predictions and results for each source"
                                                                          "extraction method.")

    args = parser.parse_args()

    patches_directory = args.patch_location
    num_patches = args.num_patches
    save_directory = args.save_directory

    real_data_save_directory = save_directory + "/real"

    Path(real_data_save_directory).mkdir(parents=True, exist_ok=True)

    # READ IN PATCHES CORRECTLY

    print("Reading in and formatting patches...")

    # train, valid, test = read_patches(num_patches=num_patches, directory=patches_directory)

    # CHANGE THIS BACK AT THE END

    train, valid, test = read_patches(num_patches=2000, directory=patches_directory)

    real_data = read_real_data(num_patches=768, directory="./real_data/real_patches/patches")

    novel_source_extraction_algorithms(training_data=train, validation_data=valid, testing_data=test,
                                       real_data=real_data, save_directory=save_directory)

    print("Complete")

# REFERENCES

# Argparse Errors - https://stackoverflow.com/questions/10900617/getting-syntax-error-near-unexpected-token-in-python
# Interpolation in Imshow - https://stackoverflow.com/questions/55121294/imshow-plot-with-no-data-values-excluded-from-interpolation
