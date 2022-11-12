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



class LGNDataset(torch.utils.data.Dataset):

    def __init__(self, training, transform=None):
        self.training = training
        self.transform = transform

        self.num_samples = 0

        self.stimulus_dataset = None
        self.response_dataset = None

        self.dataset = None
        self.response_offsets = 0
        self.response_stds = 1
        self.stimulus_offsets = 0
        self.stimulus_stds = 1

        self.load_data()

    def load_data(self):
        data_dir = os.path.join("..", "datasets")
        neuron_position_file = os.path.join(data_dir, 'lgn_convolved_imagenet_val_greyscale110x110_resize110_cropped.pickle')
        with open(neuron_position_file, 'rb') as f:
            self.dataset = pickle.load(f)

        num_all_data = np.shape(self.dataset['stim'])[0]

        num_train_data = int(num_all_data * 0.8)

        responses = self.dataset['resp']
        shmi, shma, shme, shstd = responses.min(), responses.max(), responses.mean(), responses.std()
        print("responses: over all")
        print(shmi, shma, shme, shstd)

        stimuli = self.dataset['stim']
        shmi, shma, shme, shstd = stimuli.min(), stimuli.max(), stimuli.mean(), stimuli.std()
        print("stimuli: over all")
        print(shmi, shma, shme, shstd)

        stimuli_dataset_train = self.dataset['stim'][:num_train_data]
        response_dataset_train = self.dataset['resp'][:num_train_data]
        stimuli_dataset_test = self.dataset['stim'][num_train_data:]
        response_dataset_test = self.dataset['resp'][num_train_data:]

        if self.training:
            self.stimulus_dataset = stimuli_dataset_train
            self.response_dataset = response_dataset_train
        else:
            self.stimulus_dataset = stimuli_dataset_test
            self.response_dataset = response_dataset_test
        self.stimulus_offsets = -self.stimulus_dataset.mean(axis=0)
        self.response_offsets = -self.response_dataset.mean(axis=0)
        self.stimulus_stds = self.stimulus_dataset.std(axis=0)
        self.response_stds = self.response_dataset.std(axis=0)
        # self.response_stds = self.response_dataset.std(axis=0) # use L2

        self.num_samples = np.shape(self.stimulus_dataset)[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx %= self.num_samples

        response_raw = self.response_dataset[idx]
        stimulus_raw = self.stimulus_dataset[idx]
        # img = self.stimuli_dataset[idx][55 - 32:55 + 32, 55 - 32:55 + 32]

        # response = transforms.ToTensor()(response_raw)
        response = transforms.ToTensor()(response_raw + self.response_offsets)
        # response = transforms.ToTensor()(response_raw + self.response_offsets) / self.response_stds
        # response = transforms.ToTensor()((response_raw + 41.36009) / (51.05438 + 41.36009))
        # response = (response_raw + 41.36009) / (51.05438 + 41.36009)
        # response = transforms.Normalize((3.1100, ), (7.5376, ))(response)
        # response = transforms.Resize((image_size, image_size))(response)

        stimulus = transforms.ToTensor()(stimulus_raw)
        # stimulus = transforms.ToTensor()(stimulus_raw + self.stimulus_offsets)
        # stimulus = transforms.ToTensor()(stimulus_raw + self.stimulus_offsets) / self.stimulus_stds
        # if self.transform:
        #     # stimulus = self.transform(stimulus)
        #     # Normalize the image [0, 1] -> [-1, 1]
        #     stimulus = transforms.ToTensor()(stimulus_raw)
        #     # stimulus *= 2
        #     # stimulus -= 1

        return {"stimulus": stimulus, "response": response, "response_raw": response_raw, "stimulus_raw": stimulus_raw}

    @staticmethod
    def get_dataloader(batch_size, training=True):
        transform = transforms.Compose([
            # transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # transforms.CenterCrop((51, 51)),
            # transforms.CenterCrop((15, 15)),
            # transforms.Normalize((0.4301, ), (0.2298, )),
        ])

        # Create the dataloader
        dataloader = torch.utils.data.DataLoader(LGNDataset(training=training, transform=transform),
                                                 batch_size=batch_size, shuffle=training, num_workers=workers)
        print("Returning loader with", len(dataloader.dataset), "samples")

        return dataloader


def main():
    dataloader = LGNDataset.get_dataloader(batch_size, training=False)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    print(device)

    # Plot some training images
    real_batch = next(iter(dataloader))
    responses = real_batch['response'].to(device)
    shmi, shma, shme, shstd = responses.min(), responses.max(), responses.mean(), responses.std()
    print("responses: over all")
    print(shmi, shma, shme, shstd)

    stimuli = real_batch['stimuli'].to(device)
    shmi, shma, shme, shstd = stimuli.min(), stimuli.max(), stimuli.mean(), stimuli.std()
    print("stimuli: over all")
    print(shmi, shma, shme, shstd)

    plt.figure(figsize=(64, 32))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Stimuli")
    plt.imshow(np.transpose(vutils.make_grid(real_batch['stimulus'].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))

    # plt.subplot(1, 2, 2)
    # plt.axis("off")
    # plt.title("Sheets")
    # plt.imshow(np.transpose(vutils.make_grid(real_batch['stimulus'].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0))[:, :, :3])
    # plt.savefig("samples.png")


if __name__ == '__main__':
    main()

