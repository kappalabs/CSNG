import os
import copy
from collections import defaultdict

import wandb
import torch
import kornia.losses
import torch.utils.data

import numpy as np
import torch.nn as nn
import torch.optim as optim

from typing import Tuple

from lgn_deconvolve.model_evaluator import ModelEvaluator
from ml_models.model_base import ModelBase
from ml_models.conv_models.conv_model_base import CNNModelBase


class CNNModel(CNNModelBase):

    def __init__(self, stimuli_shape, response_shape, num_filters=1):
        super().__init__(stimuli_shape, response_shape)

        ngf = 32
        self.max_pool_kernel_size = 1
        # num_filters = int(np.prod(response_shape))
        number_inputs = int(np.prod(response_shape) // self.max_pool_kernel_size)
        number_outputs = int(np.prod(stimuli_shape))
        number_output_channels = 1

        # self.compressed_side_size = 16
        # self.compressed_channels = 8
        # self.compressed_length = self.compressed_side_size ** 2 * self.compressed_channels
        #
        # self.compress = nn.Sequential(
        #     nn.Linear(in_features=number_inputs, out_features=self.compressed_length, bias=False),
        #     nn.BatchNorm1d(num_features=self.compressed_length),
        #     nn.ReLU(True),
        # )

        # self.compressed_side_size = 5
        # self.compressed_channels = 2_400
        self.compressed_side_size = 2
        self.compressed_channels = 15_000
        # self.compressed_side_size = 1
        # self.compressed_channels = 60_000

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels=self.compressed_channels, out_channels=ngf * 16, kernel_size=3, stride=1,
                               padding=0),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 3, 2, 1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 4, 3, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, number_output_channels, 3, 2, 1),
            nn.Sigmoid()
            # state size. (number_outputs) x 64 x 64
        )

    def forward(self, x):
        # out_img = self.deconv(x)
        # out_img = self.conv(out_img)
        x = x.view(x.shape[0], -1)
        x = nn.functional.max_pool1d(x, kernel_size=self.max_pool_kernel_size, stride=self.max_pool_kernel_size)
        # print("x.shape orig", x.shape)
        # x = self.compress(x)
        # print("x.shape compressed", x.shape)
        x = x.view(x.shape[0], self.compressed_channels, self.compressed_side_size, self.compressed_side_size)
        # print("x.shape compressed resized", x.shape)
        out_img = self.main(x)
        # print("out_img.shape", out_img.shape)
        out_img = nn.functional.interpolate(out_img, size=self.stimuli_shape, mode='bilinear', align_corners=False)
        # print("out_img.shape resized", out_img.shape)

        return out_img
