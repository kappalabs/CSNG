import numpy as np
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, num_channels=1):
        super(Discriminator, self).__init__()

        ngf = 32

        self.encode = nn.Sequential(
            nn.Conv2d(num_channels, ngf, 3, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(True),

            nn.Conv2d(ngf, ngf * 2, 3, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(True),

            nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(True),

            nn.Conv2d(ngf * 4, ngf * 8, 3, 2, 1),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(True),

            nn.Conv2d(ngf * 8, ngf * 16, 3, 2, 1),
            nn.BatchNorm2d(ngf * 16),
            nn.LeakyReLU(True),

            nn.Conv2d(ngf * 16, ngf * 32, 3, 2, 1),
            nn.BatchNorm2d(ngf * 32),
            nn.LeakyReLU(True),

            nn.Conv2d(ngf * 32, 1, 3, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encode(x)

        return x
