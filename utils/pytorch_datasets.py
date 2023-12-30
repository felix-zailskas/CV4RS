import csv
import lmdb
import numpy as np
import pyarrow as pa
import pickle as pk
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage.transform import resize

from bigearthnet_common.base import ben_19_labels_to_multi_hot

def interp_band(bands, img10_shape=[120, 120]):
    """
    Iterpolate bands.
    https://github.com/lanha/DSen2/blob/master/utils/patches.py.
    """
    bands_interp = np.zeros([bands.shape[0]] + img10_shape).astype(np.float32)
    for i in range(bands.shape[0]):
        bands_interp[i] = resize(bands[i] / 30000, img10_shape, mode='reflect') * 30000

    return bands_interp




def loads_pyarrow(buf):

 # return pa.deserialize(buf)
    return pk.loads(buf)

class Ben19Dataset(Dataset):
    def __init__(self, lmdb_path, csv_path, img_transform='default'):
        """
        Parameter
        ---------
        lmdb_path:      path to the LMDB file for efficiently loading the patches.
        csv_path:       path to a csv file containing files that will make up the dataset
                        (e.g. train, ireland, val etc.)
        img_transform:  default does BEN19 specific interpolation and transformation
                        s.t. a PyTorch model can be trained.
        """
        self.lmdb_path = lmdb_path
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.env = None
        self.patch_names = self.read_csv(csv_path)
        self.transform = self.init_transform(img_transform)

    def read_csv(self, csv_data):
        patch_names = []
        with open(csv_data, 'r') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                patch_names.append(row[0])
        return patch_names

    def init_transform(self, img_transform):
        if img_transform == 'default':
            return transforms.Compose([ToTensorBEN19(),
                                       NormalizeBEN19()])
        else:
            return img_transform
    
    def _init_db(self):
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=True, meminit=False)
    def __getitem__(self, idx):
        """Get item at position idx of Dataset."""
        if self.env is None:
            self._init_db()
        patch_name = self.patch_names[idx]

        with self.env.begin(write=False) as txn:
            byteflow = txn.get(patch_name.encode())

        patch_data = loads_pyarrow(byteflow)
        
        
        
        s2_patch = patch_data.s2_patch

        s2_bands10 = np.float32(s2_patch.get_stacked_10m_bands())
        s2_bands20 = np.float32(s2_patch.get_stacked_20m_bands())
        s2_bands20 = interp_band(s2_bands20)                                
        label = ben_19_labels_to_multi_hot(patch_data.new_labels)
        
      #  label = patch_data.new_labels
        
        sample = dict(s2_bands10=s2_bands10, s2_bands20=s2_bands20, label=label, index=idx)
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

        self.bands20_mean = [950.68368468, 1792.46290469, 2075.46795189,
                             2266.46036911, 1594.42694882, 1009.32729131]
        self.bands20_std = [729.89827633, 1096.01480586, 1273.45393088,
                            1356.13789355, 1079.19066363, 818.86747235]

        self.bands60_mean = [340.76769064, 2246.0605464]
        self.bands60_std = [554.81258967, 1302.3292881]

    def __call__(self, sample):

        bands10, bands20 = sample['bands10'], sample['bands20']
        label, index = sample['label'], sample['index']

        for t, m, s in zip(bands10, self.bands10_mean, self.bands10_std):
            t.sub_(m).div_(s)

        for t, m, s in zip(bands20, self.bands20_mean, self.bands20_std):
            t.sub_(m).div_(s)

        data = torch.cat((bands10, bands20), dim=0)

        return dict(data=data, label=label, index=index)


class ToTensorBEN19(object):
    """Ben19 specific conversion of ndarrays in sample to Tensors."""

    def __call__(self, sample):
        bands10, bands20 = sample['s2_bands10'], sample['s2_bands20']
        label, index = sample['label'], sample['index']
        return dict(bands10=torch.tensor(bands10),
                    bands20=torch.tensor(bands20),
                    label=label,
                    index=index)
