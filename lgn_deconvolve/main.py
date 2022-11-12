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

from models import *
from lgn_dataset import *

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


# Set random seed for reproducibility
manualSeed = 999
manualSeed = random.randint(1, 10000)  # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


def load_state(checkpoint_file_path, device):
    # Create the generator
    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)

    best_loss_netG = best_loss_netD = float("inf")

    data_dir = os.path.join("..", "datasets")
    neuron_position_file = os.path.join(data_dir, 'lgn_convolved_imagenet_val_greyscale110x110_resize110_cropped.pickle')
    with open(neuron_position_file, 'rb') as f:
        dataset = pickle.load(f)
    plt.imshow(dataset['filter'])
    plt.savefig("filter_60x60")
    conv = dataset['filter']
    filter = torch.from_numpy(conv)
    plt.close()

    if os.path.isfile(checkpoint_file_path):
        checkpoint = torch.load(checkpoint_file_path)
        loaded_state_netG = checkpoint['model_netG']
        if 'model_netD' in checkpoint:
            loaded_state_netD = checkpoint['model_netD']

            curr_state = netD.state_dict()
            for name, param in loaded_state_netD.items():
                if name not in curr_state:
                    continue
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                if param.shape == curr_state[name].shape:
                    # param = filter.view(param.shape)
                    curr_state[name].copy_(param)
                else:
                    print("shapes mismatch, skipping... (", param.shape, curr_state[name].shape, ")")
            netD.load_state_dict(curr_state)

        curr_state = netG.state_dict()
        for name, param in loaded_state_netG.items():
            print("loaded param name", name, "shape", param.size(), 'min', param.min(), 'max', param.max())
            if name not in curr_state:
                continue
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            if param.shape == curr_state[name].shape:
                # max_p = np.max(conv)
                # param = filter.view(param.shape) / max_p * torch.max(param).detach().cpu().numpy()

                curr_state[name].copy_(param)

                print("Got param with size", param.size())

                # if len(param.size()) == 1:
                #     deconv_filter = param.view(-1, param.size()[0]).detach().cpu().numpy()
                #     plt.figure(figsize=(5, 5))
                #     # plt.axis("off")
                #     plt.title("Bias")
                #     fil = np.reshape(deconv_filter, (110, 110))
                #     plt.imshow(fil)
                #     plt.colorbar()
                #     plt.savefig("deconv_filter_bias_{}x1".format(deconv_filter.shape[0]))
                #     plt.close()


                if len(param.size()) > 1:
                    deconv_filter = param.view(-1, param.size()[-2], param.size()[-1]).detach().cpu().numpy()
                    # plt.imshow(deconv_filter + np.min(deconv_filter))
                    # plt.savefig("deconv_filter_{}x{}".format(deconv_filter.shape[0], deconv_filter.shape[1]))
                    # plt.close()

                    # deconv_filter =

                    # print("deconv")
                    # num_convs = deconv_filter.shape[0]
                    # plt.figure(figsize=(num_convs*5, 5))
                    # for i in range(num_convs):
                    #     plt.subplot(1, num_convs, i + 1)
                    #     # plt.axis("off")
                    #     plt.title("Weights")
                    #     # plt.imshow(deconv_filter[i] + np.min(deconv_filter[i]))
                    #     # fil = deconv_filter[i] + np.min(deconv_filter[i])
                    #     fil = deconv_filter[i]
                    #     print("filter shape", fil.shape)
                    #     # if fil.shape[0] % 110 != 0:
                    #     #     continue
                    #     # fil = fil[:, int(51*51/2)]
                    #     # fil = fil[:, 0]
                    #     # fil = fil[int(110*110/2), :]
                    #     # fil = np.reshape(fil, (51, 51))
                    #     plt.imshow(fil)
                    #     plt.colorbar()
                    # plt.savefig("deconv_filters_CN_{}x{}".format(deconv_filter.shape[-2], deconv_filter.shape[-1]))
                    # plt.close()

                    # print("deconv")
                    # for ind in range(110*110):
                    # # for ind in range(51*51):
                    # #     if ind != 110*55+55:
                    # #         continue
                    #     plt.figure(figsize=(5, 5))
                    #     plt.subplot(1, 1, 1)
                    #     plt.title("Weights")
                    #     fil = deconv_filter[0]
                    #     fil = fil[ind, :]
                    #     # fil = fil[:, ind]
                    #     fil = np.reshape(fil, (51, 51))
                    #     # fil = np.reshape(fil, (110, 110))
                    #     plt.imshow(fil)
                    #     plt.colorbar()
                    #     plt.savefig("plots_ln_4/deconv_filter_{:04d}".format(ind))
                    #     # plt.savefig("deconv_filters_LN_{}x{}".format(deconv_filter.shape[-2], deconv_filter.shape[-1]))
                    #     plt.close()

                    # exit()

            else:
                print("shapes mismatch, skipping... (", param.shape, curr_state[name].shape, ")")
        netG.load_state_dict(curr_state)

        best_loss_netG = checkpoint['best_loss_netG']
        if 'best_loss_netD' in checkpoint:
            best_loss_netD = checkpoint['best_loss_netD']

        print("Load Model Accuracy: netG:", best_loss_netG, ", netD:", best_loss_netD)
    else:
        print("init model load ...")
    netG.train()

    return netG, netD, best_loss_netG, best_loss_netD


def main():
    import numpy as np
    from sklearn.linear_model import LinearRegression

    data_dir = os.path.join("..", "datasets")
    neuron_position_file = os.path.join(data_dir,
                                        'lgn_convolved_imagenet_val_greyscale110x110_resize110_cropped.pickle')
    with open(neuron_position_file, 'rb') as f:
        dataset = pickle.load(f)

    num_all_data = np.shape(dataset['stim'])[0]

    percent_part = 0.2
    num_train_data = int(num_all_data * percent_part)

    stimuli_dataset_train = dataset['stim'][:num_train_data]
    response_dataset_train = dataset['resp'][:num_train_data]
    stimuli_dataset_test = dataset['stim'][num_train_data:]
    response_dataset_test = dataset['resp'][num_train_data:]

    num_test_data = len(stimuli_dataset_test)

    stimuli_dataset_train = stimuli_dataset_train.reshape((num_train_data, 110*110))
    # stimuli_dataset_test = stimuli_dataset_test.reshape((num_test_data, 110*110))

    response_dataset_train = response_dataset_train.reshape((num_train_data, 51*51))
    response_dataset_test = response_dataset_test.reshape((num_test_data, 51*51))

    lr_model = LinearRegression(fit_intercept=False)
    model_filename = "linear_regression_model_{}.pkl".format(int(percent_part * 10))
    if not os.path.exists(model_filename):
        lr_model.fit(response_dataset_train, stimuli_dataset_train)
        # X, y = response_dataset_train, stimuli_dataset_train
        # X, y, X_offset, y_offset, X_scale = lr_model._preprocess_data(
        #     X,
        #     y,
        #     fit_intercept=True,
        #     normalize=False,
        #     copy=True,
        #     sample_weight=None,
        #     return_mean=True,
        # )
        with open(model_filename, 'wb') as f:
            pickle.dump(lr_model, f)
    with open(model_filename, 'rb') as f:
        lr_model = pickle.load(f)
    # pred = lr_model.predict(response_dataset_test)
    # pred = pred.reshape((num_test_data, 110, 110))
    # plt.figure(figsize=(10, 10))
    # plt.imshow(pred[0])
    # plt.savefig("lr_result.png")
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plt.imshow(stimuli_dataset_test[0])
    # plt.savefig("lr_result_orig.png")
    # plt.close()
    # pass
    #
    # exit()

    weights = lr_model.coef_
    for ind in range(110*110):
    # for ind in range(51*51):
        if ind != 110*55+55:
            continue
        plt.figure(figsize=(5, 5))
        plt.subplot(1, 1, 1)
        plt.title("Weights")
        fil = weights
        fil = fil[ind, :]
        # fil = fil[:, ind]
        fil = np.reshape(fil, (51, 51))
        # fil = np.reshape(fil, (110, 110))
        plt.imshow(fil)
        plt.colorbar()
        # plt.savefig("plots_lr_2/deconv_filter_{:04d}".format(ind))
        plt.savefig("deconv_filter_LR_{:04d}".format(ind))
        plt.close()
    # exit()

    #
    # ###################
    # CREATING
    # ###################
    #

    dataloader_trn = LGNDataset.get_dataloader(batch_size=batch_size, training=True)
    dataloader_val = LGNDataset.get_dataloader(batch_size=64, training=False)

    version = 'v{}'.format(0)
    checkpoint_filename = "best_model"
    checkpoint_file_path = './checkpoint/' + checkpoint_filename + '_ckpt.' + version
    netG, netD, best_loss_netG, best_loss_netD = load_state(checkpoint_file_path, device)

    #
    # ###################
    # TRAINING
    # ###################
    #

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_L1_losses = []
    G_losses = []
    D_losses = []
    iters = 0

    # Initialize BCELoss function
    criterion_bce = nn.BCELoss()

    # # Create batch of latent vectors that we will use to visualize
    # #  the progression of the generator
    # fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    criterion_mse = nn.MSELoss(reduction='none')
    criterion_l1 = nn.L1Loss()
    ssim_loss = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=False, channel=1)
    # Setup Adam optimizers for both G and D
    # optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=0, amsgrad=True)
    # optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

    optimizerG = optim.SGD(netG.parameters(), lr=lr, weight_decay=0, momentum=0.8)

    # CROP_SIZE = 15
    # CROP_SIZE = 51
    CROP_SIZE = 64
    # CROP_SIZE = 110

    MASK_SIZE = 110

    center_x, center_y = MASK_SIZE / 2, MASK_SIZE / 2
    reduction_mask = np.ones((MASK_SIZE, MASK_SIZE))
    # max_val = float("-inf")
    # for y in range(MASK_SIZE):
    #     for x in range(MASK_SIZE):
    #         dist = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
    #         reduction_mask[x, y] = dist
    #         if dist > max_val:
    #             max_val = dist
    # reduction_mask /= max_val
    # reduction_mask = 1 - reduction_mask
    reduction_mask = torch.from_numpy(reduction_mask).to(device)

    resize_110 = transforms.Resize((110, 110))
    transform_crop = transforms.Compose([
        transforms.Resize((110, 110)),
        transforms.CenterCrop((CROP_SIZE, CROP_SIZE)),
    ])

    transform_discriminator = transforms.Compose([
        transforms.Resize((110, 110)),
        transforms.CenterCrop((CROP_SIZE, CROP_SIZE)),
        transforms.Resize((64, 64)),
    ])

    # # scheduler = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=30, gamma=0.1)

    real_val_batch = next(iter(dataloader_val))

    #
    # NOTE: mean over loss
    # 
    # num_samples = 0
    # loss_sum = torch.zeros((110, 110)).to(device, dtype=torch.float)
    # for i, data in enumerate(dataloader_val, 0):
    #     data_stimulus = data['stimulus'].to(device, dtype=torch.float)
    #     data_response = data['response'].to(device, dtype=torch.float)
    #     data_response_raw = data['response_raw'].to(device, dtype=torch.float)
    #
    #     # pred_gen = netG(data_response)
    #     # pred_gen = resize_110(pred_gen)
    #
    #     # NOTE: scikit model prediction
    #     data_response_raw = data_response_raw.detach().cpu().numpy()
    #     data_response_raw = data_response_raw.reshape((data_stimulus.size(0), 51*51))
    #     fake = lr_model.predict(data_response_raw)
    #     fake = fake.reshape((data_stimulus.size(0), 1, 110, 110))
    #     pred_gen = torch.from_numpy(fake).to(device, dtype=torch.float)
    #
    #     pred_gen_crop = transform_crop(pred_gen)
    #
    #     criterion_mse_ = nn.MSELoss(reduction='none')
    #     loss_mse = criterion_mse_(data_stimulus, pred_gen_crop)
    #
    #     print("loss_mse", loss_mse.size())
    #     loss_sum += loss_mse.sum(axis=0).view((110, 110))
    #     print("loss_sum", loss_sum.size())
    #     num_samples += data_stimulus.size(0)
    #
    # plt.figure(figsize=(5, 5))
    # plt.title("Loss MSE mean")
    # fil = np.reshape(loss_sum.detach().cpu() / num_samples, (110, 110))
    # plt.imshow(fil)
    # plt.colorbar()
    # plt.savefig("loss_mse_mean")
    # plt.close()
    #
    # exit()

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader_trn, 0):
            data_stimulus = data['stimulus'].to(device, dtype=torch.float)
            data_response = data['response'].to(device, dtype=torch.float)
            data_response_raw = data['response_raw'].to(device, dtype=torch.float)

            data_stimulus_crop = transform_crop(data_stimulus)
            data_stimulus_disc = transform_discriminator(data_stimulus)

            # ############################
            # # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            # ###########################
            # ## Train with all-real batch
            # netD.zero_grad()
            # # Format batch
            # b_size = data_stimulus_crop.size(0)
            # label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # # Forward pass real batch through D
            # output = netD(data_stimulus_disc).view(-1)
            # # Calculate loss on all-real batch
            # errD_real = criterion_bce(output, label)
            # # Calculate gradients for D in backward pass
            # errD_real.backward()
            # D_x = output.mean().item()
            #
            # ## Train with all-fake batch
            # # # Generate batch of latent vectors
            # # noise = torch.randn(b_size, nz, 1, 1, device=device)
            # # Generate fake image batch with G
            # fake = netG(data_response)
            # label.fill_(fake_label)
            # fake_for_discriminator = transform_discriminator(fake)
            # # Classify all fake batch with D
            # output = netD(fake_for_discriminator.detach()).view(-1)
            # # Calculate D's loss on the all-fake batch
            # errD_fake = criterion_bce(output, label)
            # # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            # errD_fake.backward()
            # D_G_z1 = output.mean().item()
            # # Compute error of D as sum over the fake and the real batches
            # errD = errD_real + errD_fake
            # # Update D
            # optimizerD.step()
            #
            # ############################
            # # (2) Update G network: maximize log(D(G(z)))
            # ###########################
            # netG.zero_grad()
            # label.fill_(real_label)  # fake labels are real for generator cost
            # # Since we just updated D, perform another forward pass of all-fake batch through D
            # output = netD(fake_for_discriminator).view(-1)
            # # Calculate G's loss based on this output
            # errG = criterion_bce(output, label)
            # # Calculate gradients for G
            # errG.backward()
            # D_G_z2 = output.mean().item()
            # # Update G
            # optimizerG.step()
            #
            # fake_crop = transform_crop(fake)
            # lossG_l1 = criterion_l1(data_stimulus_crop, fake_crop)

            optimizerG.zero_grad()
            netG.zero_grad()
            pred_gen = netG(data_response)
            # pred_gen = netG(data_response_raw)
            # pred_gen = resize_110(pred_gen)

            # # NOTE: scikit model prediction
            # data_response_raw = data_response_raw.detach().cpu().numpy()
            # data_response_raw = data_response_raw.reshape((data_stimulus.size(0), 51 * 51))
            # pred_gen = lr_model.predict(data_response_raw)
            # pred_gen = pred_gen.reshape((data_stimulus.size(0), 1, 110, 110))
            # pred_gen = torch.from_numpy(pred_gen).to(device)

            pred_gen_crop = transform_crop(pred_gen)
            # lossG_l1 = criterion_l1(data_stimulus_crop, pred_gen_crop)
            lossG_l1 = criterion_l1(data_stimulus, pred_gen)
            # data_stimulus_masked = data_stimulus * reduction_mask
            # pred_gen_masked = pred_gen * reduction_mask
            # lossG_l1_center = criterion_l1(data_stimulus_masked, pred_gen_masked)
            lossG_l1_center = criterion_l1(data_stimulus_crop, pred_gen_crop)
            loss_mse = criterion_mse(data_stimulus_crop, pred_gen_crop).mean()
            # # NOTE: no transform -> better loss calculation?
            # loss_mse = criterion_mse(data_stimulus, pred_gen).mean()
            # _ssim_loss = 1 - ssim_loss(data_stimulus, pred_gen).mean()
            # loss = lossG_l1 + 0.1 * loss_mse
            # loss = lossG_l1_center
            # loss = _ssim_loss
            loss = loss_mse
            # loss = lossG_l1
            loss.backward()
            optimizerG.step()
            # scheduler.step()

            # Save Losses for plotting later
            # G_losses.append(errG.item())
            # D_losses.append(errD.item())
            # G_L1_losses.append(float(lossG_l1.item()))
            G_L1_losses.append(float(lossG_l1_center.item()))

            # Output training stats
            if i % 500 == 0:
                print('[%d/%d][%d/%d]\tLoss_G: %.4f\tLoss: %.4f\tLoss L1 %.4f' %
                      (epoch, num_epochs, i, len(dataloader_trn), float(lossG_l1_center.item()),
                       float(loss.item()), float(lossG_l1.item())))
                # print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                #       % (epoch, num_epochs, i, len(dataloader_trn), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                plt.figure(figsize=(10, 5))
                plt.title("Generator and Discriminator Loss During Training")
                plt.plot(G_losses, label="G")
                plt.plot(G_L1_losses, label="G L1")
                plt.plot(D_losses, label="D")
                plt.xlabel("iterations")
                plt.ylabel("Loss")
                plt.legend()
                plt.savefig("training.png")
                plt.close()

                # Compute moving loss
                moving_loss = float("inf")
                if len(G_L1_losses) > 10:
                    moving_loss = np.mean(G_L1_losses[-10:])
                # Save if the loss is better
                if moving_loss < best_loss_netG:
                    print('saving because {} < {}...'.format(moving_loss, best_loss_netG), flush=True)
                    best_loss_netG = moving_loss

                    state = {
                        'model_netG': netG.state_dict(),
                        'model_netD': netD.state_dict(),
                        'best_loss_netG': best_loss_netG,
                        'best_loss_netD': best_loss_netD,
                        'epoch': epoch,
                    }
                    if not os.path.isdir('checkpoint'):
                        os.mkdir('checkpoint')
                    torch.save(state, checkpoint_file_path)

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or (i % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader_trn) - 1)):
                data_stimulus = real_val_batch['stimulus'].to(device, dtype=torch.float)
                data_response = real_val_batch['response'].to(device, dtype=torch.float)
                data_response_raw = real_val_batch['response_raw'].to(device, dtype=torch.float)

                data_stimulus_crop = transform_crop(data_stimulus).detach().cpu()

                with torch.no_grad():
                    netG.eval()
                    fake = netG(data_response).detach().cpu()
                    netG.train()
                    # fake = netG(data_response_raw).detach().cpu()
                    # fake = resize_110(fake)
                    # fake = transform_crop(fake)
                print("PyTorch fake output", fake.size())

                # # NOTE: scikit model prediction
                # data_response_raw = data_response_raw.detach().cpu().numpy()
                # data_response_raw = data_response_raw.reshape((data_stimulus.size(0), 51*51))
                # fake = lr_model.predict(data_response_raw)
                # fake = fake.reshape((data_stimulus.size(0), 1, 110, 110))
                # fake = torch.from_numpy(fake)

                fake_crop = transform_crop(fake)

                # Save the center-pixel filter
                netG.eval()
                net_params = netG.state_dict()
                for name, param in net_params.items():
                    print("loaded param name", name, "shape", param.size(), 'min', param.min(), 'max', param.max())
                    if isinstance(param, Parameter):
                        # backwards compatibility for serialized parameters
                        param = param.data

                    print("Got param with size", param.size())

                    if len(param.size()) > 1:
                        deconv_filter = param.view(-1, param.size()[-2], param.size()[-1]).detach().cpu().numpy()

                        # num_convs = deconv_filter.shape[0]
                        # plt.figure(figsize=(num_convs * 5, 5))
                        # for i in range(num_convs):
                        #     plt.subplot(1, num_convs, i + 1)
                        #     plt.title("Weights")
                        #     fil = deconv_filter[i]
                        #     print("filter shape", fil.shape)
                        #     plt.imshow(fil)
                        #     plt.colorbar()
                        # plt.savefig("deconv_filters_CN_{}x{}".format(deconv_filter.shape[-2], deconv_filter.shape[-1]))
                        # plt.close()

                        for ind in range(110*110):
                            if ind != 110*55+55:
                                continue
                            plt.figure(figsize=(5, 5))
                            plt.subplot(1, 1, 1)
                            plt.title("Weights")
                            fil = deconv_filter[0]
                            fil = fil[ind, :]
                            # fil = fil[:, ind]
                            fil = np.reshape(fil, (51, 51))
                            # fil = np.reshape(fil, (110, 110))
                            plt.imshow(fil)
                            plt.colorbar()
                            # plt.savefig("plots_ln_3/deconv_filter_{:04d}".format(ind))
                            plt.savefig("deconv_filters_LN_{}x{}".format(deconv_filter.shape[-2], deconv_filter.shape[-1]))
                            plt.close()
                netG.train()


                # fake = transform_crop(fake)
                # data_stimulus = transform_crop(data_stimulus)
                img_list.append(vutils.make_grid(fake_crop[:64], padding=2, normalize=True))

                # Plot the fake images from the last epoch
                if len(img_list) > 0:
                    # Plot the real images
                    plt.figure(figsize=(20, 5))
                    plt.subplot(1, 4, 1)
                    plt.axis("off")
                    plt.title("Stimuli")
                    plt.imshow(np.transpose(vutils.make_grid(data_stimulus_crop[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

                    plt.subplot(1, 4, 2)
                    plt.axis("off")
                    plt.title("Predicted Images")
                    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))

                    # plt.subplot(1, 3, 3)
                    # plt.axis("off")
                    # plt.title("Response")
                    # plt.imshow(np.transpose(vutils.make_grid(data_response[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

                    criterion_mse_ = nn.MSELoss(reduction='none')
                    loss_mse = criterion_mse_(data_stimulus_crop, fake_crop)
                    plt.subplot(1, 4, 3)
                    plt.axis("off")
                    plt.title("Loss MSE")
                    plt.imshow(np.transpose(vutils.make_grid(loss_mse[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))
                    print("MSE TST:", float(loss_mse.mean().item()))

                    criterion_l1_ = nn.L1Loss(reduction='none')
                    loss_l1 = criterion_l1_(data_stimulus_crop, fake_crop)
                    plt.subplot(1, 4, 4)
                    plt.axis("off")
                    plt.title("Loss L1")
                    plt.imshow(np.transpose(vutils.make_grid(loss_l1[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))
                    print("L1 TST:", float(loss_l1.mean().item()))

                    _ssim_loss = 1 - ssim_loss(data_stimulus_crop, fake_crop)
                    print("SSIM TST:", float(_ssim_loss.mean().item()))

                    # plt.show()
                    plt.savefig("validation_sample.png")
                    plt.close()

            iters += 1


if __name__ == '__main__':
    main()
