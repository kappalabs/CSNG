import os
import torch
import pickle
import torch.utils.data

import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class LGNData:

    def __init__(self, train_part=0.8, datanorm=None):
        # Load the dataset from pickle file
        data_dir = os.path.join("..", "datasets")
        neuron_position_file = \
            os.path.join(data_dir, 'lgn_convolved_imagenet_val_greyscale110x110_resize110_cropped.pickle')
        with open(neuron_position_file, 'rb') as f:
            self.dataset = pickle.load(f)

        # Number of all available samples
        self.num_all_data = len(self.dataset['stim'])

        # Calculate training set size
        self.num_train_data = int(self.num_all_data * train_part)

        # Training data
        self.stimuli_dataset_train = self.dataset['stim'][:self.num_train_data]  # 110 x 110
        self.response_dataset_train = self.dataset['resp'][:self.num_train_data]  # 51 x 51

        # Stimuli: compute offsets & standard deviations
        self.stimuli_offsets = np.mean(self.stimuli_dataset_train, axis=0)
        self.stimuli_stds = np.std(self.stimuli_dataset_train, axis=0)

        # Responses: compute offsets & standard deviations
        self.response_offsets = np.mean(self.response_dataset_train, axis=0)
        self.response_stds = np.std(self.response_dataset_train, axis=0)

        # Testing data
        self.stimuli_dataset_test = self.dataset['stim'][self.num_train_data:]
        self.response_dataset_test = self.dataset['resp'][self.num_train_data:]

        if datanorm == 'mean0_std1':
            self.stimuli_dataset_train -= self.stimuli_offsets
            self.stimuli_dataset_train /= self.stimuli_stds
            self.response_dataset_train -= self.response_offsets
            self.response_dataset_train /= self.response_stds

            # Testing data
            self.stimuli_dataset_test -= self.stimuli_offsets
            self.stimuli_dataset_test /= self.stimuli_stds
            self.response_dataset_test -= self.response_offsets
            self.response_dataset_test /= self.response_stds

        self.stimuli_shape = self.stimuli_dataset_train[0].shape
        self.response_shape = self.response_dataset_train[0].shape

        self.num_test_data = len(self.stimuli_dataset_test)

    def get_train(self):
        return self.stimuli_dataset_train, self.response_dataset_train

    def get_test(self):
        return self.stimuli_dataset_test, self.response_dataset_test


class LGNDataset(torch.utils.data.Dataset):

    def __init__(self, response, stimuli, datanorm):
        self.num_samples = response.shape[0]

        self.stimulus_dataset = stimuli
        self.response_dataset = response

        self.response_offsets = 0
        self.response_stds = 1
        self.stimulus_offsets = 0
        self.stimulus_stds = 1

        self.datanorm = datanorm

        self.load_data()

    def load_data(self):
        if self.response_dataset is not None:
            shmi, shma, shme, shstd = self.response_dataset.min(), self.response_dataset.max(), \
                                      self.response_dataset.mean(), self.response_dataset.std()
            print(" - responses raw description:", shmi, shma, shme, shstd)

            self.response_offsets = self.response_dataset.mean(axis=0)
            self.response_stds = self.response_dataset.std(axis=0)

            self.num_samples = np.shape(self.response_dataset)[0]

        if self.stimulus_dataset is not None:
            shmi, shma, shme, shstd = self.stimulus_dataset.min(), self.stimulus_dataset.max(), \
                                      self.stimulus_dataset.mean(), self.stimulus_dataset.std()
            print(" - stimuli raw description:", shmi, shma, shme, shstd)

            self.stimulus_offsets = self.stimulus_dataset.mean(axis=0)
            self.stimulus_stds = self.stimulus_dataset.std(axis=0)

            self.num_samples = np.shape(self.stimulus_dataset)[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        response = torch.zeros((1, ))
        response_raw = torch.zeros((1, ))
        if self.response_dataset is not None:
            response_raw = self.response_dataset[idx]

            if self.datanorm is None:
                response = transforms.ToTensor()(response_raw)
            elif self.datanorm == "mean0":
                response = transforms.ToTensor()(response_raw - self.response_offsets)
            elif self.datanorm == "mean0_std1":
                response = transforms.ToTensor()(response_raw - self.response_offsets) / self.response_stds

        stimulus = torch.zeros((1, ))
        stimulus_raw = torch.zeros((1, ))
        if self.stimulus_dataset is not None:
            stimulus_raw = self.stimulus_dataset[idx]

            stimulus = transforms.ToTensor()(stimulus_raw)
            # stimulus = transforms.ToTensor()(stimulus_raw - self.stimulus_offsets)
            # stimulus = transforms.ToTensor()(stimulus_raw - self.stimulus_offsets) / self.stimulus_stds

        return {"stimulus": stimulus, "response": response,
                "response_raw": response_raw, "stimulus_raw": stimulus_raw}


class CentralPxCropTransform:
    def __call__(self, x):
        x = TF.crop(x, 55, 55, 1, 1)
        return x
