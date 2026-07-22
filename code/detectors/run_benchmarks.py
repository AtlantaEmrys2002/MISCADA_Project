import argparse
from benchmarks.uneb import uneb_algorithm
from benchmarks.unek import unek_algorithm
from read_write_functions import read_patches
from utils import random_data


# UNEK Algorithm

# train_batches, validation_batches, test_batches = random_data(n=256)

# print(unek_algorithm(train_batches, validation_batches, test_batches))

# print(uneb_algorithm(train_batches, validation_batches, test_batches))

# from benchmarks.components.clustering_algorithms import blob_detection
# import numpy as np
# import torch
#
# test_data = []
#
# for i, data in enumerate(test_batches):
#
#     inputs, labels = data[0], data[1]
#
#     test_data.append(data[1])
#
# test_data = torch.from_numpy(np.array(test_data))
#
# import matplotlib.pyplot as plt
#
# test_loc = blob_detection(test_data[0])
#
# for k in range(len(test_data[0])):
#
#     # WHY IS IT SOMETIMES CRASHING IN tERMS OF SIZE
#
#     if len(test_loc[k]) != 0:
#
#         D = test_data[0][k][0].detach().numpy().astype(np.uint8)
#
#         # Highlight areas
#         for x in range(64):
#             for y in range(64):
#                 if D[x, y] == 1:
#                     D[x, y] = 128
#
#         # centres = np.round((np.array([p.pt for p in test_loc[k]]))).astype(np.uint8)
#
#         centres = test_loc[k]
#
#         for c in centres:
#
#             D[c[0], c[1]] = 255
#
#         plt.imshow(D, interpolation='none')
#
#         plt.show()

if __name__ == "__main__":

    # PROCESS USER INPUT

    parser = argparse.ArgumentParser(description="Reads in a collection of specified patches and formats them to be "
                                                 "passed to benchmark algorithms. These benchmarks are then applied and"
                                                 "the parameters of the 'best' versions of them are saved.")

    parser.add_argument("--patch_location", required=True, type=str, help="The directory in which the"
                                                                          "patch data is stored.")

    parser.add_argument("--num_patches", required=True, type=int, help="The number of patches to read from"
                                                                       "the specified patch directory.")

    args = parser.parse_args()

    patches_directory = args.patch_location
    num_patches = args.num_patches

    # READ IN PATCHES CORRECTLY

    print("Reading in and formatting patches...")

    # WARNING - REMEMBER THAT PREDICTED LOCATIONS ARE NOT IN THE SAME ORDER AS ACTUAL LOCATIONS - HAVE TO FIND WHICH ONES ARE CLOSEST
    # REMEMBER ACTUAL LOCATIONS ARE GIVEN AS y,x AND NOT x, y - READ tHE METADATA CAREFULLY!!!!!!! LOOK BACK AT NOTES IN read_write_functions

    train, valid, test, metadata = read_patches(num_patches=num_patches, directory=patches_directory)

    print("Complete")

    # train, valid, test = random_data(n=256)

    unek_predicted_segmentations, unek_predicted_locations = unek_algorithm(train, valid, test)


    # Save predictions

    # print(predicted_locations)

    print(metadata)




# REFERENCES

# Interpolation in Imshow - https://stackoverflow.com/questions/55121294/imshow-plot-with-no-data-values-excluded-from-interpolation
