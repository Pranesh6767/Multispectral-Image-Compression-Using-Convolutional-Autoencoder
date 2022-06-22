##################################################
# code to generate, print and save training/validation results :
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

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import torch

class logger_regression:
    def __init__(self,test_tensor_path,device,savefolder):
        self.losslistinstance_train = []
        self.losslistinstance_train_cont = []
        self.epochwiselosstrain = []

        self.losslistinstance_val = []
        self.losslistinstance_val_cont = []
        self.epochwiselossval = []

        self.mselistinstance_val = []
        self.epochwisemseval = []

        self.psnrlistinstance_val = []
        self.epochwisepsnrval = []

        self.test_tensor = torch.load(test_tensor_path)
        self.device = device

        self.savefolder = savefolder

        self.LRS = []
        self.imagenumber = 0
        if os.path.exists(self.savefolder):
            pass
        else:
            os.mkdir(self.savefolder)

    def updateLossListTrain(self,loss,LRC):
        self.losslistinstance_train.append(loss)
        self.LRS.append(LRC)

    def updateLossListval(self,lossdict):
        loss = lossdict['loss']
        mse = lossdict['mse']
        psnr = lossdict['psnr']
        self.losslistinstance_val.append(loss)
        self.mselistinstance_val.append(mse)
        self.psnrlistinstance_val.append(psnr)

    def displayLossTrain(self):
        avg_loss = np.array(self.losslistinstance_train).mean()
        print("Train: Average Squared Recunstruction Loss:",avg_loss)

    def displayLossval(self):
        avg_loss = np.array(self.losslistinstance_val).mean()
        avg_MSE = np.array(self.mselistinstance_val).mean()
        avg_PSNR = np.array(self.psnrlistinstance_val).mean()
        print("Val: Average Squared Recunstruction Loss:",avg_loss)
        print("Val: Average MSE between original images and Recunstructed images:",avg_MSE)
        print("Val: Average PSNR between original images and Recunstructed images:", avg_PSNR)
        return avg_loss

    def resetTrainLoss(self):
        avg_loss = np.array(self.losslistinstance_train).mean()
        self.epochwiselosstrain.append(avg_loss)
        self.losslistinstance_train_cont.extend(self.losslistinstance_train)
        self.losslistinstance_train = []

    def resetvalLoss(self):
        avg_loss = np.array(self.losslistinstance_val).mean()
        avg_MSE = np.array(self.mselistinstance_val).mean()
        avg_PSNR = np.array(self.psnrlistinstance_val).mean()

        self.epochwiselossval.append(avg_loss)
        self.epochwisemseval.append(avg_MSE)
        self.epochwisepsnrval.append(avg_PSNR)

        self.losslistinstance_val_cont.extend(self.losslistinstance_val)

        self.losslistinstance_val = []
        self.mselistinstance_val = []
        self.psnrlistinstance_val = []


    def plotLoss(self):
        plt.plot(self.epochwiselosstrain, '-bx')
        plt.plot(self.epochwiselossval, '-rx')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend(['Training', 'Validation'])
        plt.title('Loss vs. No. of epochs')
        plt.savefig(os.path.join(self.savefolder,'0_lossplot.png'))
        plt.show()

        plt.plot(self.epochwisemseval, '-bx')
        plt.xlabel('epochs')
        plt.ylabel('MSE')
        plt.title('Val: MSE vs. No. of Epochs')
        plt.savefig(os.path.join(self.savefolder,'0_mse.png'))
        plt.show()

        plt.plot(self.epochwisepsnrval, '-bx')
        plt.xlabel('epochs')
        plt.ylabel('PSNR (in dB)')
        plt.title('Val: PSNR vs. No. of Epochs')
        plt.savefig(os.path.join(self.savefolder,'0_psnr.png'))
        plt.show()

        plt.plot(self.losslistinstance_train_cont, '-bx')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Train: Loss vs. No. of Steps')
        plt.savefig(os.path.join(self.savefolder,'0_train_cont_loss.png'))
        plt.show()

        plt.plot(self.losslistinstance_val_cont, '-bx')
        plt.xlabel('Validation Steps')
        plt.ylabel('Loss')
        plt.title('Val: Loss vs. No. of Steps')
        plt.savefig(os.path.join(self.savefolder,'0_val_cont_loss.png'))
        plt.show()

    def plotLR(self):
        plt.plot(self.LRS)
        plt.xlabel('Batch no.')
        plt.ylabel('Learning rate')
        plt.title('Learning Rate vs. Batch no.')
        plt.savefig(os.path.join(self.savefolder,'0_LRplot.png'))
        plt.show()

    def visualize_io(self,model):
        o = model(self.test_tensor.to(self.device))

        im_arr_ip = []
        for i in range(8):
            im_arr_ip.append(self.test_tensor.cpu().numpy()[i,2, :, :])

        for i in range(8):
            im_arr_ip.append(o.cpu().detach().numpy()[i,2, :, :])

        fig = plt.figure(figsize=(20., 5.))

        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(2, 8),  # creates 2x2 grid of axes
                         axes_pad=0.1,  # pad between axes in inch.
                         )

        for ax, im in zip(grid, im_arr_ip):
            # Iterating over the grid returns the Axes.
            ax.imshow(im, cmap='gray')

        plt.savefig(os.path.join(self.savefolder,'io_' + str(self.imagenumber) + '.png'))
        self.imagenumber = self.imagenumber + 1
        plt.show()

    def savemodel(self,model,modelpath):
        return None
