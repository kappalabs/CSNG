import torch.nn as nn


class CNNModelBase(nn.Module):

    def __init__(self, stimuli_shape, response_shape):
        super().__init__()

        self.stimuli_shape = stimuli_shape
        self.response_shape = response_shape
