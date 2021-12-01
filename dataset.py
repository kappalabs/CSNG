import os
import pickle
import argparse
import random
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

from params import *


class StimuliDataset(torch.utils.data.Dataset):

    def __init__(self, training, transform=None, transform_sheets=None):
        self.training = training
        self.transform = transform
        self.transform_sheets = transform_sheets

        self.num_samples = 0
        # self.response_data = None
        # self.image_data = None

        self.neuron_positions_dataframe = None
        self.stimuli_dataframe = None
        self.response_dataframe = None

        self.load_data()

    @staticmethod
    def reconstruct_sheets(neuron_positions, response_neurons):
        sheets = neuron_positions['sheet']
        xs = neuron_positions['x']
        ys = neuron_positions['y']
        num_neurons = len(sheets)

        sheet_to_idx = {'V1_Exc_L2/3': 0, 'V1_Exc_L4': 1, 'V1_Inh_L4': 2, 'V1_Inh_L2/3': 3}

        num_sheets = len(set(sheets))
        xmi, xma = np.min(np.array(xs)), np.max(np.array(xs))
        ymi, yma = np.min(np.array(ys)), np.max(np.array(ys))

        sheets_size = (64, 64)
        sheet_images = np.zeros((*sheets_size, num_sheets))

        neuron_idxs = np.random.choice(range(num_neurons), int(num_neurons * 0.1), replace=False)
        # for neuron_idx in range(num_neurons):
        for neuron_idx in neuron_idxs:
            x, y, sheet = xs[neuron_idx], ys[neuron_idx], sheets[neuron_idx]
            x_pos = np.round((x - xmi) / (xma - xmi) * (sheets_size[0] - 1))
            x_pos = int(np.clip(x_pos, 0, sheets_size[0] - 1))
            y_pos = np.round((y - ymi) / (yma - ymi) * (sheets_size[1] - 1))
            y_pos = int(np.clip(y_pos, 0, sheets_size[1] - 1))
            sheet_idx = sheet_to_idx[sheet]

            sheet_images[y_pos, x_pos, sheet_idx] += response_neurons[neuron_idx]

        return sheet_images

    def load_data(self):
        data_dir = os.path.join("DataForDavid")
        neuron_position_file = os.path.join(data_dir, 'neuron_position_pandas_dataframe.pickle')
        with open(neuron_position_file, 'rb') as f:
            self.neuron_positions_dataframe = pickle.load(f)

        time_window_dir = os.path.join(data_dir, "time_window_25-350")
        stimuli_file = os.path.join(time_window_dir, "stim_single_trial.pickle")
        with open(stimuli_file, 'rb') as f:
            self.stimuli_dataframe = pickle.load(f)
        response_file = os.path.join(time_window_dir, "resp_single_trial.pickle")
        with open(response_file, 'rb') as f:
            self.response_dataframe = pickle.load(f)

        # sheet_images = self.reconstruct_sheets(neuron_positions_dataframe, response_dataframe)

        # self.response_data = sheet_images
        # self.response_data = np.array(response_dataframe)
        # self.image_data = np.array(stimuli_dataframe)
        # print("image_data:", np.min(self.image_data), np.max(self.image_data), np.mean(self.image_data), np.std(self.image_data))
        # exit()

        train_num = int(np.shape(self.response_dataframe)[0] * 0.8)

        if self.training:
            self.stimuli_dataframe = self.stimuli_dataframe[:train_num]
            self.response_dataframe = self.response_dataframe[:train_num]
            # self.response_data = self.response_data[:train_num]
            # self.image_data = self.image_data[:train_num]
        else:
            # self.response_data = self.response_data[train_num:]
            # self.image_data = self.image_data[train_num:]
            self.stimuli_dataframe = self.stimuli_dataframe[train_num:]
            self.response_dataframe = self.response_dataframe[train_num:]

        self.num_samples = np.shape(self.stimuli_dataframe)[0]

    def __len__(self):
        return self.num_samples * 100

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx %= self.num_samples

        # img = self.image_data[idx][55-32:55+32, 55-32:55+32]
        # label = self.response_data[idx]
        img = self.stimuli_dataframe[idx][55-32:55+32, 55-32:55+32]
        sheet_images = self.reconstruct_sheets(self.neuron_positions_dataframe, self.response_dataframe[idx])
        # shmi, shma, shme, shstd = sheet_images.min(), sheet_images.max(), sheet_images.mean(), sheet_images.std()
        # print(shmi, shma, shme, shstd, sheet_images.shape)

        if self.transform:
            img = self.transform(img)
        if self.transform_sheets:
            sheet_images = self.transform_sheets(sheet_images)
            # print("size after transform", sheet_images.size())

        return {"image": img, "sheets": sheet_images}


def get_dataloader(training=True):
    subsample = 16
    transform = transforms.Compose([
        # transforms.Resize(image_size),
        # transforms.CenterCrop((64, 64)),
        transforms.ToTensor(),
        # transforms.Resize((subsample, subsample)),
        # transforms.Resize((64, 64)),
        transforms.Normalize((50, ), (33, )),
    ])
    transform_sheets = transforms.Compose([
        # transforms.Resize(image_size),
        # transforms.CenterCrop((64, 64)),
        transforms.ToTensor(),
        # transforms.Resize((subsample, subsample)),
        # transforms.Resize((64, 64)),
        transforms.Normalize((0.2, ), (0.7, )),
    ])

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(StimuliDataset(training=training, transform=transform,
                                                            transform_sheets=transform_sheets),
                                             batch_size=batch_size, shuffle=True, num_workers=workers)

    return dataloader


if __name__ == '__main__':
    dataloader = get_dataloader()

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    print(device)

    # Plot some training images
    real_batch = next(iter(dataloader))
    sheet_images = real_batch['sheets'].to(device)
    shmi, shma, shme, shstd = sheet_images.min(), sheet_images.max(), sheet_images.mean(), sheet_images.std()
    print("over all")
    print(shmi, shma, shme, shstd)

    plt.figure(figsize=(64, 32))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Stimuli")
    plt.imshow(np.transpose(vutils.make_grid(real_batch['image'].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Sheets")
    plt.imshow(np.transpose(vutils.make_grid(real_batch['sheets'].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0))[:, :, :3])
    plt.savefig("samples.png")
