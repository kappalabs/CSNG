import numpy as np
import torch.nn as nn

from ml_models.conv_models.conv_model_base import CNNModelBase


class CNNModel(CNNModelBase):

    def __init__(self, stimuli_shape, response_shape, dropout: float):
        super().__init__(stimuli_shape, response_shape, dropout)

        self.number_inputs = int(np.prod(response_shape))

        self.compression_fc_size = 512
        self.output_fc_channels = 1
        self.output_fc_side_size = 110

        self.intermediate = nn.Sequential(
            nn.Linear(self.number_inputs, self.compression_fc_size),
            nn.BatchNorm1d(self.compression_fc_size),
            nn.ReLU(True),
            nn.Dropout(self.dropout),

            nn.Linear(self.compression_fc_size, self.output_fc_channels * self.output_fc_side_size ** 2),
            nn.BatchNorm1d(self.output_fc_channels * self.output_fc_side_size ** 2),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
        )

        ngf = 32
        number_output_channels = 1

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.output_fc_channels, out_channels=ngf * 16, kernel_size=3, stride=1,
                               padding=1),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 3, 1, 1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 1, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 1, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 3, 1, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, number_output_channels, 3, 1, 1),
            nn.Sigmoid()
            # state size. (number_outputs) x 64 x 64
        )

    def forward(self, x):
        x = self.intermediate(x)
        x = x.view(-1, self.output_fc_channels, self.output_fc_side_size, self.output_fc_side_size)

        x = self.decode(x)

        x = nn.functional.interpolate(x, size=self.stimuli_shape[-2:], mode='bilinear', align_corners=False)

        return x
