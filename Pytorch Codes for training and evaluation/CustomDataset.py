##################################################
# Code to create the Numpy/HDR dataset (Only compatible with pytorch):
'''
MIT License

Copyright (c) 2022 Pranesh6767

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''
##################################################

from torch.utils.data import Dataset
import torch
import numpy as np
from spectral import open_image
import pickle


class ICONESDataset_patched(Dataset):
    """ returns patches directly."""

    def __init__(self, file_paths, std_scaler_path):
        """
        Args:
            file_paths (list): list of filepaths of HDR files .
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file_IDs = file_paths # HDR file path
        dbfile = open(std_scaler_path, 'rb')
        self.standardisation_param = pickle.load(dbfile)

    def __len__(self):
        return len(self.file_IDs)

    def __getitem__(self, index: int):
        img_path = self.file_IDs[index].split('?')[0]
        img_part_R = int(self.file_IDs[index].split('?')[1])
        img_part_C = int(self.file_IDs[index].split('?')[2])

        img = open_image(img_path)
        img_open = img.open_memmap(writeable=True).astype('float32')
        img_np = np.array(img_open)
        if (img_part_R==3):
            x0 = img_np.shape[0]-64
            x1 = img_np.shape[0]
        else:
            x0 = 64* img_part_R
            x1 = x0 + 64

        if (img_part_C==3):
            y0 = img_np.shape[1]-64
            y1 = img_np.shape[1]
        else:
            y0 = 64* img_part_R
            y1 = y0 + 64

        img_np = img_np[x0:x1, y0:y1, :]
        img_std = img_np.copy()
        for channel_i in range(224):
            img_std[:, :, channel_i] = (img_np[:, :, channel_i] - self.standardisation_param['mean'][channel_i]) / \
                                       self.standardisation_param['std'][channel_i]

        img_tensor = torch.from_numpy(img_std)

        sample = (img_tensor, 0)

        return sample

class ICONESDataset(Dataset):
    """returns bigger image you need to split them while training."""

    def __init__(self, file_paths, std_scaler_path):
        """
        Args:
            file_paths (list): list of filepaths of HDR files .
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file_IDs = file_paths # HDR file path
        dbfile = open(std_scaler_path, 'rb')
        self.standardisation_param = pickle.load(dbfile)

    def __len__(self):
        return len(self.file_IDs)

    def __getitem__(self, index: int):
        img_path = self.file_IDs[index]

        img = open_image(img_path)
        img_open = img.open_memmap(writeable=True).astype('float32')
        img_np = np.array(img_open)

        img_np = img_np[0:256, 0:256, :]
        img_std = img_np.copy()
        for channel_i in range(224):
            img_std[:, :, channel_i] = (img_np[:, :, channel_i] - self.standardisation_param['mean'][channel_i]) / \
                                       self.standardisation_param['std'][channel_i]

        img_tensor = torch.from_numpy(img_std)

        sample = (img_tensor, 0)

        return sample

class numpyDataset(Dataset):
    """returns bigger image you need to split them while training."""

    def __init__(self, file_paths, std_scaler_path=None,normalization_par = None):
        """
        Args:
            file_paths (list): list of filepaths of HDR files .
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file_IDs = file_paths # HDR file path
        self.normalization_par = normalization_par
        if normalization_par == None:
            dbfile = open(std_scaler_path, 'rb')
            self.standardisation_param = pickle.load(dbfile)

    def __len__(self):
        return len(self.file_IDs)

    def __getitem__(self, index: int):
        img_path = self.file_IDs[index]
        img_np = np.load(img_path)

        if self.normalization_par==None:
            img_std = img_np.copy()
            for channel_i in range(img_np.shape[2]):
                img_std[:, :, channel_i] = (img_np[:, :, channel_i] - self.standardisation_param['mean'][channel_i]) / \
                                           self.standardisation_param['std'][channel_i]
        else:
            img_std = img_np / self.normalization_par

        img_tensor = torch.from_numpy(img_std)

        sample = (img_tensor, 0)

        return sample
