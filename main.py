import os
import pandas
import pickle
import argparse
import os
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
from torch.nn import Parameter

from ml_models import *
from dataset import *


# Set random seed for reproducibility
manualSeed = 999
manualSeed = random.randint(1, 10000)  # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print('device', device)


def load_state(checkpoint_file_path, device, model=None):
    if model is None:
        # Create the generator
        netG = Generator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    netG.apply(weights_init)

    best_loss = 123456789

    if os.path.isfile(checkpoint_file_path):
        checkpoint = torch.load(checkpoint_file_path)
        loaded_state = checkpoint['model_netG']
        curr_state = netG.state_dict()
        for name, param in loaded_state.items():
            if name not in curr_state:
                continue
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            if param.shape == curr_state[name].shape:
                curr_state[name].copy_(param)
            else:
                print("shapes mismatch, skipping... (", param.shape, curr_state[name].shape, ")")
        netG.load_state_dict(curr_state)

        best_loss = checkpoint['best_loss_netG']

        print("Load Model Accuracy: ", best_loss)
    else:
        print("init model load ...")

    return netG, best_loss


def main():
    # data_dir = os.path.join("DataForDavid")
    # neuron_position_file = os.path.join(data_dir, 'neuron_position_pandas_dataframe.pickle')
    # with open(neuron_position_file, 'rb') as f:
    #     dataframe = pickle.load(f)

    data_dir = os.path.join("datasets")
    lgn_imagenet_dataset = os.path.join(data_dir, 'lgn_convolved_imagenet_val_greyscale110x110_resize110_cropped.pickle')
    dataset = pickle.load(open(lgn_imagenet_dataset, 'rb'))

    print()

    #
    # ###################
    # CREATING
    # ###################
    #


    # # Print the model
    # print(netG)

    # sh = dataframe['sheet']
    # xs = dataframe['x']
    # ys = dataframe['y']
    # xmi, xma = np.min(np.array(xs)), np.max(np.array(xs))
    # ymi, yma = np.min(np.array(ys)), np.max(np.array(ys))
    # sheets = set(sh)

    # plt.figure(figsize=(10, 10))
    # plt.title("Neuron positions")
    # plt.plot(xs, ys, '.', label="positions")
    # # plt.plot(ys, '.', label="Y")
    # plt.xlabel("Index")
    # plt.ylabel("Value")
    # plt.legend()
    # plt.savefig("neuron_positions.png")
    # exit()

    dataloader = get_dataloader()
    dataloader_val = get_dataloader(training=False)

    best_loss = 123456789
    version = 'v{}'.format(0)
    checkpoint_filename = "best_model"
    checkpoint_file_path = './checkpoint/' + checkpoint_filename + '_ckpt.' + version
    netG, best_loss = load_state(checkpoint_file_path, device)

    #
    # ###################
    # TRAINING
    # ###################
    #

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    iters = 0

    criterion_mse = nn.MSELoss()
    criterion = nn.L1Loss()
    # Setup Adam optimizers for both G and D
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            data_img = data['image'].to(device, dtype=torch.float)
            # data_labels = data['label'].to(device, dtype=torch.float).view(data_img.size(0), -1, 1, 1)
            data_labels = data['sheets'].to(device, dtype=torch.float)

            # print("tensor data", torch.min(data_img), torch.max(data_img), torch.mean(data_img), torch.std(data_img))

            netG.zero_grad()
            pred_gen = netG(data_labels)
            lossG = criterion(data_img, pred_gen)
            loss_mse = criterion_mse(data_img, pred_gen)
            # loss = lossG + 0.1 * loss_mse
            loss = lossG
            loss.backward()
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_G: %.4f\tLoss: %.4f' % (epoch, num_epochs, i, len(dataloader), lossG.item(), loss.item()))
                plt.figure(figsize=(10, 5))
                plt.title("Generator and Discriminator Loss During Training")
                plt.plot(G_losses, label="G")
                plt.xlabel("iterations")
                plt.ylabel("Loss")
                plt.legend()
                # plt.show()
                plt.savefig("training.png")
                plt.close()

            # Save Losses for plotting later
            curr_loss = float(lossG.item())
            G_losses.append(curr_loss)

            if curr_loss < best_loss:
                best_loss = curr_loss

                print('saving...', flush=True)
                state = {
                    'model_netG': netG.state_dict(),
                    'best_loss_netG': best_loss,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, checkpoint_file_path)

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                real_batch = next(iter(dataloader_val))
                data_img = real_batch['image'].to(device, dtype=torch.float)
                # data_labels = real_batch['label'].to(device, dtype=torch.float).view(data_img.size(0), -1, 1, 1)
                data_labels = real_batch['sheets'].to(device, dtype=torch.float)

                with torch.no_grad():
                    fake = netG(data_labels).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                # Plot the fake images from the last epoch
                if len(img_list) > 0:
                    # Plot the real images
                    plt.figure(figsize=(15, 15))
                    plt.subplot(1, 2, 1)
                    plt.axis("off")
                    plt.title("GT Images")
                    plt.imshow(np.transpose(vutils.make_grid(data_img[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

                    plt.subplot(1, 2, 2)
                    plt.axis("off")
                    plt.title("Predicted Images")
                    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
                    # plt.show()
                    plt.savefig("validation_sample.png")
                    plt.close()

            iters += 1


if __name__ == '__main__':
    main()
