##################################################
# Example: to use the training codes:
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
# Importing required libraries
import torch
import shutil
import os

# importing API files to wd
import_paths = '../input/cae-codes'
modules = os.listdir(import_paths)
for file_i in modules:
    shutil.copy(os.path.join(import_paths,file_i),file_i)

# importing API scripts in runtime
import dataloader
import logger
import trainer

# defining device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# creating instance of dataloader
CusDataLoader = dataloader.customdataloader('../input/processed-eurosat/EuroSat Output')
trainDL, testDL = CusDataLoader.load_numpyDataset(normalization_par=5500)

# Defining model architecture
model = None
'''
-----------
'''
model = model.to(device)

# creating instance of Logger
Cuslogger = logger.logger_regression('../input/cae-codes/test_tensor_vis.pt',device,'visuals')

# Maximum pixel value in whole dataset
max_pixel_value = 1

# creating instance of trainer
Custrainer = trainer.trainAutoencoder(model,trainDL,testDL,Cuslogger,device,max_pixel_value,epochs = 30)

# Calling train_model method of class trainer
Custrainer.train_model()
