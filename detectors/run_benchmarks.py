from benchmarks.uneb import uneb_algorithm
from benchmarks.unek import unek_algorithm
from utils import random_data


# UNEK Algorithm

train_batches, validation_batches, test_batches = random_data(n=256)

# print(unek_algorithm(train_batches, validation_batches, test_batches))

print(uneb_algorithm(train_batches, validation_batches, test_batches))
