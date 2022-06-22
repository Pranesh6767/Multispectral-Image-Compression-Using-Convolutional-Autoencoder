##################################################
# Baseline architecture for CAE:
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

import torch.nn as nn
import torch.nn.functional as F


# Encoder V3 Dataset V2

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.actfun = nn.LeakyReLU()

        # encoder

        self.conv1_1 = nn.Conv2d(in_channels=13, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(16)

        self.conv2_1 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(8)

        self.conv3_1 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(4)

        self.conv4_1 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=2, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(2)

        # Decoder

        self.convT1_1 = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=3, stride=2, output_padding=1,
                                           padding=1)
        self.convT1_2 = nn.ConvTranspose2d(in_channels=2, out_channels=4, kernel_size=3, stride=1, output_padding=0,
                                           padding=1)
        self.batch_norm5 = nn.BatchNorm2d(4)

        self.convT2_1 = nn.ConvTranspose2d(in_channels=4, out_channels=4, kernel_size=3, stride=2, output_padding=1,
                                           padding=1)
        self.convT2_2 = nn.ConvTranspose2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, output_padding=0,
                                           padding=1)
        self.batch_norm6 = nn.BatchNorm2d(8)

        self.convT3_1 = nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=2, output_padding=1,
                                           padding=1)
        self.convT3_2 = nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, output_padding=0,
                                           padding=1)
        self.batch_norm7 = nn.BatchNorm2d(16)

        self.convT4_1 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, output_padding=1,
                                           padding=1)
        self.convT4_2 = nn.ConvTranspose2d(in_channels=16, out_channels=13, kernel_size=3, stride=1, output_padding=0,
                                           padding=1)

    def forward(self, x):
        x = self.actfun(self.batch_norm1(self.conv1_2(self.conv1_1(x))))
        x = self.actfun(self.batch_norm2(self.conv2_2(self.conv2_1(x))))
        # x = self.actfun(self.batch_norm3(self.conv3_2(self.conv3_1(x))))
        # x = self.actfun(self.batch_norm4(self.conv4_2(self.conv4_1(x))))

        # x = self.actfun(self.batch_norm5(self.convT1_2(self.convT1_1(x))))
        # x = self.actfun(self.batch_norm6(self.convT2_2(self.convT2_1(x))))
        x = self.actfun(self.batch_norm7(self.convT3_2(self.convT3_1(x))))
        x = self.sigmoid(self.convT4_2(self.convT4_1(x)))
        return x


model = Net()