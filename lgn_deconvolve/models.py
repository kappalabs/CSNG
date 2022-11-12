import os
import argparse
import os
import random

import timm.models.resnet
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch.nn.functional as F

from params import *


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        print("initing zeros")
        nn.init.zeros_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        # self.extractor = timm.models.resnet.resnet18(pretrained=False, num_classes=nz, in_chans=4)

        # self.main = nn.Sequential(
        #     # input is Z, going into a convolution
        #     nn.ConvTranspose2d(1, ngf * 8, 7, 1, 0, bias=False),
        #     # nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
        #     nn.BatchNorm2d(ngf * 8),
        #     # nn.BatchNorm2d(ngf),
        #     nn.ReLU(True),
        #     # state size. (ngf*8) x 4 x 4
        #     nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        #     # nn.ConvTranspose2d(ngf, ngf * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 4),
        #     nn.ReLU(True),
        #     # state size. (ngf*4) x 8 x 8
        #     nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 2),
        #     nn.ReLU(True),
        #     # state size. (ngf*2) x 16 x 16
        #     nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf),
        #     nn.ReLU(True),
        #     # state size. (ngf) x 32 x 32
        #     nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
        #     nn.Tanh()
        #     # state size. (nc) x 64 x 64
        # )

        # self.deconv = nn.ConvTranspose2d(1, 1, 60, 1, 0, bias=False)

        # self.main = nn.Sequential(
        #     nn.Linear(51*51, 110*110, bias=True),
        #     nn.Tanh()
        # )

        # hidden_size = 1024 * 6
        # self.fc1 = nn.Linear(51*51, hidden_size, bias=True)
        # self.fc2 = nn.Linear(hidden_size, 110*110, bias=True)

        self.fc1 = nn.Linear(51*51, 110*110, bias=True)
        # self.fc1 = nn.Linear(51*51, 110*110, bias=False)

        # self.fc1_bn = nn.BatchNorm2d(1)
        # self.conv1 = nn.ConvTranspose2d(1, 8, 3, 1, 1, bias=True)
        # self.conv1_bn = nn.BatchNorm2d(8)
        # self.conv2 = nn.ConvTranspose2d(8, 8, 3, 1, 1, bias=True)
        # self.conv2_bn = nn.BatchNorm2d(8)
        # self.conv3 = nn.ConvTranspose2d(8, 8, 3, 1, 1, bias=True)
        # self.conv3_bn = nn.BatchNorm2d(8)
        # self.conv4 = nn.ConvTranspose2d(8, 4, 3, 1, 1, bias=True)
        # self.conv4_bn = nn.BatchNorm2d(4)
        # self.conv5 = nn.ConvTranspose2d(4, 1, 3, 1, 1, bias=True)

        # self.main = nn.Sequential(
        #     nn.ConvTranspose2d(1, 1, 60, 1, 0, bias=True),
        #     # nn.Tanh()
        # )

        # self.main = nn.Sequential(
        #     # nn.Conv2d(1, 1, (36, 36), 1, (47, 47), bias=True, padding_mode='zeros'),
        #     nn.Conv2d(1, 1, (10, 10), 1, (34, 34), bias=True, padding_mode='zeros'),
        #     # nn.Tanh()
        # )

        # self.main = nn.Sequential(
        #     nn.ConvTranspose2d(1, 1, 60, 1, 0, bias=True),
        #     nn.Tanh()
        # )

        # self.main = nn.Sequential(
        #     nn.ConvTranspose2d(1, ngf, 60, 1, 0, bias=False),
        #     nn.BatchNorm2d(ngf),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(ngf, nc, 1, 1, 0, bias=False),
        #     nn.Tanh()
        # )

        # self.main = nn.Sequential(
        #     # input is Z, going into a convolution
        #     nn.ConvTranspose2d(1, ngf * 8, 7, 1, 0, bias=False),
        #     # nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
        #     nn.BatchNorm2d(ngf * 8),
        #     # nn.BatchNorm2d(ngf),
        #     nn.ReLU(True),
        #     # state size. (ngf*8) x 4 x 4
        #     nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 2, 1, bias=False),
        #     # nn.ConvTranspose2d(ngf, ngf * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 4),
        #     nn.ReLU(True),
        #     # state size. (ngf*4) x 8 x 8
        #     nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 2),
        #     nn.ReLU(True),
        #     # state size. (ngf*2) x 16 x 16
        #     nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf),
        #     nn.ReLU(True),
        #     # state size. (ngf) x 32 x 32
        #     nn.ConvTranspose2d(ngf, nc, 3, 2, 1, bias=False),
        #     nn.Tanh()
        #     # state size. (nc) x 64 x 64
        # )

    def forward(self, x):
        # features = self.extractor(x).view(-1, nz, 1, 1)
        # out_img = self.deconv(x)
        # out_img = self.main(x)

        # out_img = self.main(input.view(x.size(0), -1)).view(x.size(0), nc, 110, 110)

        x = nn.Flatten()(x)
        # out_img = F.relu(self.fc1(x))
        # out_img = self.fc2(out_img).view(x.size(0), 1, 110, 110)

        out_img = self.fc1(x).view(x.size(0), 1, 110, 110)

        # out_img = F.relu(self.fc1_bn(out_img))
        # out_img = self.conv1(out_img)
        # out_img = F.relu(self.conv1_bn(out_img))
        # out_img = self.conv2(out_img)
        # out_img = F.relu(self.conv2_bn(out_img))
        # out_img = self.conv3(out_img)
        # out_img = F.relu(self.conv3_bn(out_img))
        # out_img = self.conv4(out_img)
        # out_img = F.relu(self.conv4_bn(out_img))
        # out_img = self.conv5(out_img)
        # print("size", out_img.size())
        # exit()

        # out_img = self.main(x)
        # print("size", out_img.size())

        return out_img


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        out = self.main(input)

        return out
