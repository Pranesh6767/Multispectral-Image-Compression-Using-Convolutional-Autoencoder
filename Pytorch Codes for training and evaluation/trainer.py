##################################################
# Code to train and validate the pytorch model (CAE):
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

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from math import log10, sqrt
import pickle
import os


class trainAutoencoder:
    def __init__(self,model,train_dataset,test_dataset,logger,device,max_pixel_value,loss_fun = 'l1',
                 epochs = 30,max_lr = 0.003,grad_clip = 0.1, weight_decay = 1e-4,pct_start=0.2,
                 LRS = None, LRS_UP_per_step = True, LRS_UP_per_epoch = False):

        # Input class params
        self.curloss_max = 9999999

        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.logger = logger
        self.savefolder = logger.savefolder
        self.modelsavePATH = os.path.join(self.savefolder,'model_weights.pt')
        self.epochs = epochs
        self.max_lr = max_lr
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        self.device = device

        self.max_pixel_value = max_pixel_value

        # derived class params
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.max_lr, weight_decay=self.weight_decay)
        if LRS == None:
            self.sched = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, self.max_lr, epochs=self.epochs,
                                                             pct_start = pct_start, steps_per_epoch=len(self.train_dataset))
        else:
            self.sched = LRS

        self.LRS_UP_per_step = LRS_UP_per_step
        self.LRS_UP_per_epoch = LRS_UP_per_epoch
        # You can also choose  CyclicLR - triangular2 or CyclicLR - exp_range... Just make sure that you have stepped
        # the Schedular at right point it might be after epoch or after step

        if loss_fun=='mse':
            self.lossfun = nn.MSELoss()

        if loss_fun=='bce':
            self.lossfun = nn.BCELoss()

        if loss_fun=='l1':
            self.lossfun = nn.L1Loss()


    def get_lr(self,opt):
        for param_group in opt.param_groups:
            return param_group['lr']

    def get_PSNR(self,original, compressed):
        '''

        :param original: Input image
        :param compressed: reconstructed image
        :return: touple : (mse,psnr)
        '''
        mse = np.mean((original - compressed) ** 2)
        if (mse == 0):
            return (0,100)
        max_pixel = self.max_pixel_value
        psnr = 20 * log10(max_pixel / sqrt(mse))
        return (mse,psnr)

    def training_step(self, batch):
        cur_LR = self.get_lr(self.optimizer)
        '''
        for input_img in range(batch[0].shape[0]):
            cur_img = batch[0][input_img, :, :, :].permute(2, 0, 1)

            for row in range(4):
                for col in range(4):
                    if (col == 3):
                        x0 = cur_img.shape[1] - 64
                        x1 = cur_img.shape[1]
                    else:
                        x0 = 64 * col
                        x1 = x0 + 64

                    if (row == 3):
                        y0 = cur_img.shape[2] - 64
                        y1 = cur_img.shape[2]
                    else:
                        y0 = 64 * row
                        y1 = y0 + 64
                    patched_tensor.append(cur_img[:, x0:x1, y0:y1])
        patched_tensor = torch.stack(patched_tensor).to(self.device)'''
        self.optimizer.zero_grad()
        patched_tensor = batch[0].permute(0, 3, 1, 2).to(self.device)

        cunstucted_out = self.model(patched_tensor)
        loss = self.lossfun(cunstucted_out,patched_tensor)
        loss.backward()
        if self.grad_clip:
            nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()

        if self.LRS_UP_per_step:
            self.sched.step()

        return (float(loss.detach()),cur_LR)

    def validation_step(self, batch):
        '''
        patched_tensor = []
        for input_img in range(batch[0].shape[0]):
            cur_img = batch[0][input_img, :, :, :].permute(2, 0, 1)

            for row in range(4):
                for col in range(4):
                    if (col == 3):
                        x0 = cur_img.shape[1] - 64
                        x1 = cur_img.shape[1]
                    else:
                        x0 = 64 * col
                        x1 = x0 + 64

                    if (row == 3):
                        y0 = cur_img.shape[2] - 64
                        y1 = cur_img.shape[2]
                    else:
                        y0 = 64 * row
                        y1 = y0 + 64
                    patched_tensor.append(cur_img[:, x0:x1, y0:y1])
        patched_tensor = torch.stack(patched_tensor).to(self.device)'''
        patched_tensor = batch[0].permute(0, 3, 1, 2).to(self.device)

        cunstucted_out = self.model(patched_tensor)
        loss = self.lossfun(cunstucted_out,patched_tensor)
        mse_val,psnr_val = self.get_PSNR(patched_tensor.cpu().detach().numpy(),cunstucted_out.cpu().detach().numpy())

        return {'loss':float(loss.detach()),'mse':mse_val,'psnr':psnr_val}

    def train_model(self):
        torch.cuda.empty_cache()
        self.logger.visualize_io(self.model)
        for n_epoch in range(self.epochs):
            print("Training Started")
            for i,batch in enumerate(self.train_dataset):
                curloss,curLR = self.training_step(batch)
                self.logger.updateLossListTrain(curloss,curLR)
                '''
                if i%10 == 0:
                    self.logger.displayLossTrain()'''
            print("")

            print("Validation Stated")
            for i, batch in enumerate(self.test_dataset):
                curlossdict = self.validation_step(batch)
                self.logger.updateLossListval(curlossdict)
                '''
                if i % 10 == 0:
                    self.logger.displayLossval()'''
            print("")
            print("Epoch ", str(n_epoch + 1), "Completed")
            print("Dispaying the Results")

            self.logger.displayLossTrain()
            self.logger.resetTrainLoss()

            curloss_now_temp = self.logger.displayLossval()
            self.logger.resetvalLoss()
            self.logger.visualize_io(self.model)

            if self.LRS_UP_per_epoch:
                self.sched.step()

            if curloss_now_temp < self.curloss_max:
                self.curloss_max = curloss_now_temp
                torch.save(self.model.state_dict(), self.modelsavePATH)

            print("")
            print("____________________________________________")

        print("Training Completed")
        print("Displaying the Final results")
        self.logger.plotLoss()
        self.logger.plotLR()
        try:
            file_pi = open(os.path.join(self.savefolder,'logger.obj'), 'wb')
            pickle.dump(self.logger, file_pi)
            # load that sh*t with this code:
            '''import pickle 
            filehandler = open('logger.obj', 'rb') 
            logger = pickle.load(filehandler)'''
        except:
            print("Couldn't save logger")

        return None
