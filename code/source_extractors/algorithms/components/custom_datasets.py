# import torch

from torch import is_tensor, from_numpy

from torch.utils.data import Dataset


class FermiCountMapDataset(Dataset):
    """Custom dataset of fermi patches for training only (not testing)."""

    # def __init__(self, patches_directory, masks_directory, num_patches):

    def __init__(self, patches, masks, num_patches):

        self.patches = patches

        self.masks = masks

        self.num_patches = num_patches

    def __len__(self):

        return self.num_patches

    def __getitem__(self, idx):

        if is_tensor(idx):
            idx = idx.tolist()

        selected_patches = self.patches[idx]
        selected_masks = self.masks[idx]

        sample = {"patch": from_numpy(selected_patches), "mask": from_numpy(selected_masks)}

        return sample


class ClassifierSubPatchesDataset(Dataset):
    """Custom dataset of fermi sub-patches for training (not testing) deep learning classifier."""

    def __init__(self, data, num_sub_patches):

        self.sub_patches = [k[0] for k in data]

        self.labels = [k[1] for k in data]

        self.num_sub_patches = num_sub_patches

    def __len__(self):

        return self.num_sub_patches

    def __getitem__(self, idx):

        if is_tensor(idx):
            idx = idx.tolist()

        selected_sub_patches = self.sub_patches[idx]
        selected_labels = self.labels[idx]

        sample = {"subpatch": from_numpy(selected_sub_patches), "label": from_numpy(selected_labels)}

        return sample
