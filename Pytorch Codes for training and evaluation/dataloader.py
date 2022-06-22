##################################################
# Code for generative train dataloader and test dataloader:
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

import os
from sklearn.model_selection import train_test_split
import CustomDataset
from torch.utils.data.dataloader import DataLoader


class customdataloader:
    def __init__(self,global_data_dir):
        self.global_data_dir = global_data_dir
        self.torch_dataloader = None
        self.random_seed_1 = 34
        self.batch_size = 64

    def load_ICONESDataset_pached(self, std_scaler_path):
        subfolders = os.listdir(self.global_data_dir)
        subfolders = [os.path.join(self.global_data_dir, i) for i in subfolders]
        list_of_files = []
        for j in subfolders:
            list_of_files_temp = os.listdir(j)
            list_of_files_temp = [os.path.join(j, i) for i in list_of_files_temp if (i[-3:] == 'hdr')]
            list_of_files.extend(list_of_files_temp)

        list_of_files_mod = []
        for i in range(4):
            for j in range(4):
                list_of_files_mod_temp = [(k + '?' + str(i) + '?' + str(j)) for k in list_of_files]
                list_of_files_mod.extend(list_of_files_mod_temp)

        train_file_paths, test_file_paths = train_test_split(list_of_files_mod, test_size=0.20,
                                                             random_state=self.random_seed_1)

        ICONES_train_set = CustomDataset.ICONESDataset_patched(train_file_paths, std_scaler_path)
        ICONES_test_set = CustomDataset.ICONESDataset_patched(test_file_paths, std_scaler_path)

        train_dl = DataLoader(ICONES_train_set, self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_dl = DataLoader(ICONES_test_set, self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        return train_dl,test_dl

    def load_ICONESDataset(self, std_scaler_path):
        subfolders = os.listdir(self.global_data_dir)
        subfolders = [os.path.join(self.global_data_dir, i) for i in subfolders]
        list_of_files = []
        for j in subfolders:
            list_of_files_temp = os.listdir(j)
            list_of_files_temp = [os.path.join(j, i) for i in list_of_files_temp if (i[-3:] == 'hdr')]
            list_of_files.extend(list_of_files_temp)

        train_file_paths, test_file_paths = train_test_split(list_of_files, test_size=0.20,
                                                             random_state=self.random_seed_1)

        ICONES_train_set = CustomDataset.ICONESDataset(train_file_paths, std_scaler_path)
        ICONES_test_set = CustomDataset.ICONESDataset(test_file_paths, std_scaler_path)

        batch_size = 4 # batchsize becomes 16X of original after splitting

        train_dl = DataLoader(ICONES_train_set, batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_dl = DataLoader(ICONES_test_set, batch_size, shuffle=True, num_workers=4, pin_memory=True)

        return train_dl, test_dl

    def load_numpyDataset(self, std_scaler_path=None,normalization_par=None):
        list_of_files = os.listdir(self.global_data_dir)
        for filename_i in range(len(list_of_files)):
            list_of_files[filename_i] = os.path.join(self.global_data_dir,list_of_files[filename_i])

        train_file_paths, test_file_paths = train_test_split(list_of_files, test_size=0.20,
                                                             random_state=self.random_seed_1)
        if normalization_par==None:
            train_set = CustomDataset.numpyDataset(train_file_paths, std_scaler_path=std_scaler_path)
            test_set = CustomDataset.numpyDataset(test_file_paths, std_scaler_path=std_scaler_path)
        else:
            train_set = CustomDataset.numpyDataset(train_file_paths,normalization_par = normalization_par)
            test_set = CustomDataset.numpyDataset(test_file_paths,normalization_par= normalization_par)


        train_dl = DataLoader(train_set, self.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        test_dl = DataLoader(test_set, self.batch_size, shuffle=True, num_workers=2, pin_memory=True)

        return train_dl, test_dl


    def update_random_seed(self,up_ran_seed):
        self.random_seed_1 = up_ran_seed
