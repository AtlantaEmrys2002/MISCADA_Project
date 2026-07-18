import argparse
from benchmarks.uneb import uneb_algorithm
from benchmarks.unek import unek_algorithm
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

    parser = argparse.ArgumentParser(description="Generates a series of catalogs of simulated gamma-ray sources (AGNs"
                                                 "and pulsars) with spectral and spatial parameter distributions "
                                                 "identical to that of a specified catalog (e.g. 4FGL) and stores them "
                                                 "in a fermitools-compatible XML format.")

    # READ IN PATCHES CORRECTLY

    train_batches, validation_batches, test_batches = random_data(n=256)



# REFERENCES

# Interpolation in Imshow - https://stackoverflow.com/questions/55121294/imshow-plot-with-no-data-values-excluded-from-interpolation
