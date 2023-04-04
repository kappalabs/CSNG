import numpy as np
import torch.nn as nn

from ml_models.conv_models.conv_model_base import CNNModelBase


class CNNModel(CNNModelBase):
    """
    This model is based on the model from the paper:
    Reconstruction of natural visual scenes from neural spikes with deep
    neural networks
    Yichen Zhang, Shanshan Jia, Yajing Zheng, Zhaofei Yu, Yonghong Tian,
    Siwei Ma, Tiejun Huang, Jian K. Liu

    implemented as in https://github.com/jiankliu/Spike-Image-Decoder/blob/main/SID.py
    """

    def __init__(self, stimuli_shape, response_shape, dropout: float):
        super().__init__(stimuli_shape, response_shape, dropout)

        self.number_inputs = int(np.prod(response_shape))

        print("self.number_inputs: ", self.number_inputs)

        self.compression_fc_size = 512
        self.output_fc_channels = 1
        self.output_fc_side_size = 110
        number_output_channels = 1

        self.intermediate = nn.Sequential(
            nn.Linear(self.number_inputs, self.compression_fc_size),
            nn.BatchNorm1d(self.compression_fc_size),
            nn.ReLU(True),
            nn.Dropout(self.dropout),

            nn.Linear(self.compression_fc_size, self.output_fc_channels * self.output_fc_side_size ** 2),
            # nn.BatchNorm1d(self.output_fc_channels * self.output_fc_side_size ** 2),
            # nn.ReLU(True),
            # nn.Dropout(self.dropout),
            nn.Sigmoid(),
        )

        self.encode = nn.Sequential(
            nn.Conv2d(self.output_fc_channels, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 5, 2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(.25),

            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout(.25),

            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout(.25),
        )

        self.decode = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout(.25),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(.25),

            nn.Upsample(size=55),
            nn.Conv2d(128, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout(.25),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, number_output_channels, 7, 1, 3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.intermediate(x)
        x = x.view(-1, self.output_fc_channels, self.output_fc_side_size, self.output_fc_side_size)

        x = self.encode(x)
        x = self.decode(x)

        x = nn.functional.interpolate(x, size=self.stimuli_shape[-2:], mode='bilinear', align_corners=False)

        return x
