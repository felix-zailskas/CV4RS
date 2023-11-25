import csv
import pickle as pk
from pathlib import Path

import lmdb
import numpy as np
import pyarrow as pa
import torch
import torchvision.transforms as transforms
from skimage.transform import resize
from torch.utils.data import Dataset


def interp_band(bands, img10_shape=[120, 120]):
    bands_interp = np.zeros([bands.shape[0]] + img10_shape).astype(np.float32)
    for i in range(bands.shape[0]):
        bands_interp[i] = resize(bands[i] / 30000, img10_shape, mode="reflect") * 30000

    return bands_interp


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pk.loads(buf)


class Ben19Dataset(Dataset):
    def __init__(self, lmdb_path, csv_path, img_transform="default"):
        """
        Parameter
        ---------
        lmdb_path:      path to the LMDB file for efficiently loading the patches.
        csv_path:       path to a csv file containing files that will make up the dataset
                        (e.g. train, ireland, val etc.)
        img_transform:  default does BEN19 specific interpolation and transformation
                        s.t. a PyTorch model can be trained.
        """
        lmdb_path = str(lmdb_path)
        self.env = lmdb.open(
            lmdb_path, readonly=True, lock=False, readahead=False, meminit=False
        )
        self.patch_names = self.read_csv(csv_path)
        self.transform = self.init_transform(img_transform)

    def read_csv(self, csv_data):
        patch_names = []
        with open(csv_data, "r") as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                patch_names.append(row[0])
        return patch_names

    def init_transform(self, img_transform):
        if img_transform == "default":
            return transforms.Compose([ToTensorBEN19(), NormalizeBEN19()])
        else:
            return img_transform

    def __getitem__(self, idx):
        """Get item at position idx of Dataset."""
        patch_name = self.patch_names[idx]

        with self.env.begin(write=False) as txn:
            byteflow = txn.get(patch_name.encode())

        bands10, bands20, _, multi_hot_label = loads_pyarrow(byteflow)
        bands20 = interp_band(bands20)
        bands10 = bands10.astype(np.float32)
        bands20 = bands20.astype(np.float32)
        label = multi_hot_label.astype(np.float32)

        sample = dict(bands10=bands10, bands20=bands20, label=label, index=idx)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        """Get length of Dataset."""
        return len(self.patch_names)


class NormalizeBEN19(object):
    """BEN19 specific normalization."""

    def __init__(self):
        self.bands10_mean = [429.9430203, 614.21682446, 590.23569706, 2218.94553375]
        self.bands10_std = [572.41639287, 582.87945694, 675.88746967, 1365.45589904]

        self.bands20_mean = [
            950.68368468,
            1792.46290469,
            2075.46795189,
            2266.46036911,
            1594.42694882,
            1009.32729131,
        ]
        self.bands20_std = [
            729.89827633,
            1096.01480586,
            1273.45393088,
            1356.13789355,
            1079.19066363,
            818.86747235,
        ]

        self.bands60_mean = [340.76769064, 2246.0605464]
        self.bands60_std = [554.81258967, 1302.3292881]

    def __call__(self, sample):
        bands10, bands20 = sample["bands10"], sample["bands20"]
        label, index = sample["label"], sample["index"]

        for t, m, s in zip(bands10, self.bands10_mean, self.bands10_std):
            t.sub_(m).div_(s)

        for t, m, s in zip(bands20, self.bands20_mean, self.bands20_std):
            t.sub_(m).div_(s)

        data = torch.cat((bands10, bands20), dim=0)

        return dict(data=data, label=label, index=index)


class ToTensorBEN19(object):
    """Ben19 specific conversion of ndarrays in sample to Tensors."""

    def __call__(self, sample):
        bands10, bands20 = sample["bands10"], sample["bands20"]
        label, index = sample["label"], sample["index"]
        return dict(
            bands10=torch.tensor(bands10),
            bands20=torch.tensor(bands20),
            label=label,
            index=index,
        )
