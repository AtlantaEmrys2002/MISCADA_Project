import math
import numpy as np
from pathlib import Path
import pickle


def read_patches(num_patches=int, directory=str, split=True):

    # Read in patches
    patches = (np.array([np.load("{}/patch_{}/patch.npy".format(directory, n)) for n in range(num_patches)])
               .astype(np.float32))

    # Read in masks
    masks = (np.array([[np.load("{}/patch_{}/mask.npy".format(directory, n))] for n in range(num_patches)])
             .astype(np.float32))

    # Size of each split
    train_set_size = math.floor(float(num_patches) * 0.7)
    validation_set_size = math.floor(float(num_patches) * 0.2)
    test_set_size = int(num_patches) - train_set_size - validation_set_size

    # Selects same patches each time - only important here, as may train segmentation models and then work on improving
    # classifier, but still want the same set-up.
    rng = np.random.default_rng(42)

    if split:

        # RANDOM SPLIT OF INDICES - 70% vs 20% v 10%

        possible_indices = np.arange(num_patches)

        train_patch_ids = rng.choice(possible_indices, size=train_set_size, replace=False)

        validation_patch_ids = rng.choice(np.setdiff1d(possible_indices, train_patch_ids), size=validation_set_size,
                                              replace=False)

        test_patch_ids = rng.choice(np.setdiff1d(possible_indices, np.concatenate((train_patch_ids, validation_patch_ids))),
                                        size=test_set_size, replace=False)

        training_maps = np.array([patches[k] for k in train_patch_ids])
        validation_maps = np.array([patches[k] for k in validation_patch_ids])
        testing_maps = np.array([patches[k] for k in test_patch_ids])

        training_masks = np.array([masks[k][0] for k in train_patch_ids])
        validation_masks = np.array([masks[k][0] for k in validation_patch_ids])
        testing_masks = np.array([masks[k][0] for k in test_patch_ids])

        return (train_patch_ids, training_maps, training_masks, validation_patch_ids, validation_maps, validation_masks,
                test_patch_ids, testing_maps, testing_masks)

    else:

        return np.array([k for k in range(num_patches)]), np.array([patches[k] for k in range(num_patches)]), np.array([masks[k][0] for k in range(num_patches)])


def save_predictions(patch_ids, predicted_segmentations, predicted_locations, predicted_classes, actual_classes,
                     directory: str, method: str, real=False):
    # Create directory to store results

    save_location = directory + "/{}/".format(method)

    Path(save_location).mkdir(parents=True, exist_ok=True)

    if real:

        patches_location = "./real_data/real_patches/"

    else:

        patches_location = "./../data_simulation/simulated_data/patches/"

    # SAVE ACTUAL AND PREDICTED SEGMENTATIONS OF EACH PATCH

    actual_segmentations = [np.load(patches_location + "patch_{}/mask.npy".format(p)) for p in patch_ids]

    predicted_segmentations = [predicted_segmentations[p][0].astype(np.float64) for p in range(len(patch_ids))]

    segmentations = np.array(list(zip(actual_segmentations, predicted_segmentations)))

    np.save(save_location + "segmentations.npy", segmentations)

    # SAVE ACTUAL AND PREDICTED LOCATIONS OF EACH PATCH

    with open(save_location + "predicted_locations.data", 'wb') as f:

        pickle.dump(predicted_locations, f)

    # Save patch IDs to ensure the actual locations of the sources can be retrieved
    np.save(save_location + "patch_ids.npy", patch_ids)

    # SAVE ACTUAL AND PREDICTED CLASSIFICATIONS OF EACH DETECTED SOURCE

    classifications = [[actual_classes[k], predicted_classes[k]] for k in range(len(actual_classes))]

    np.save(save_location + "classifications", classifications)

# REFERENCES

# Order with Test Data - https://discuss.pytorch.org/t/maintaining-order-of-data-while-using-dataloader/32297/2
# Pickle - https://stackoverflow.com/questions/20996267/how-to-save-2d-arrays-lists-in-python
# Pickle - https://stackoverflow.com/questions/17225287/write-and-read-a-list-from-file
