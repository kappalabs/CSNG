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


class ConvolutionalNetworkModel(ModelBase):

    class NNModel(nn.Module):

        def __init__(self, stimuli_shape, response_shape, num_filters=1):
            super().__init__()

            self.stimuli_shape = stimuli_shape
            self.response_shape = response_shape

            ngf = 32
            self.max_pool_kernel_size = 5
            # num_filters = int(np.prod(response_shape))
            number_inputs = int(np.prod(response_shape) // self.max_pool_kernel_size)
            number_outputs = int(np.prod(stimuli_shape))
            number_output_channels = 1

            self.compressed_side_size = 16
            self.compressed_channels = 8
            self.compressed_length = self.compressed_side_size ** 2 * self.compressed_channels

            self.compress = nn.Sequential(
                nn.Linear(in_features=number_inputs, out_features=self.compressed_length, bias=False),
                nn.BatchNorm1d(num_features=self.compressed_length),
                nn.ReLU(True),
            )

            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(in_channels=self.compressed_channels, out_channels=ngf * 8, kernel_size=3, stride=1,
                                   padding=0),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 2, 1),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(ngf, number_output_channels, 3, 2, 1),
                nn.Tanh()
                # state size. (number_outputs) x 64 x 64
            )

        def forward(self, x):
            # out_img = self.deconv(x)
            # out_img = self.conv(out_img)
            x = x.view(x.shape[0], -1)
            x = nn.functional.max_pool1d(x, kernel_size=self.max_pool_kernel_size, stride=self.max_pool_kernel_size)
            # print("x.shape orig", x.shape)
            x = self.compress(x)
            # print("x.shape compressed", x.shape)
            x = x.view(x.shape[0], self.compressed_channels, self.compressed_side_size, self.compressed_side_size)
            # print("x.shape compressed resized", x.shape)
            out_img = self.main(x)
            # print("out_img.shape", out_img.shape)
            out_img = nn.functional.interpolate(out_img, size=self.stimuli_shape, mode='bilinear', align_corners=False)
            # print("out_img.shape resized", out_img.shape)

            return out_img

    def __init__(self, checkpoint_filepath: str, device: torch.device, config: dict,
                 data_stimuli_shape: tuple, data_response_shape: tuple):
        super().__init__(checkpoint_filepath, device, config, data_stimuli_shape, data_response_shape)

        self.model_loss = config['model_loss']
        self.best_loss = float("inf")
        self.epoch = 0

        self.criterion = None
        if self.model_loss == 'L1':
            self.criterion = nn.L1Loss(reduction='none')
        elif self.model_loss == 'L2' or self.model_loss == 'MSE':
            self.criterion = nn.MSELoss(reduction='none')
        elif self.model_loss == 'SSIM':
            self.criterion = kornia.losses.SSIMLoss(window_size=3, reduction='none')
        else:
            raise Exception("Unknown loss function: " + self.model_loss)

        self.load_model()

        if self.config['clear_progress']:
            self.best_loss = float("inf")
            self.epoch = 0
            self.num_epochs_curr = 0

    def save_model(self):
        data = {
            'network': self.model.state_dict(),
            'best_loss': self.best_loss,
            'epoch': self.epoch,
            'wandb_run_id': self.wandb_run_id,
            'num_epochs': self.num_epochs_curr,
            'stimuli_shape': self.stimuli_shape,
            'response_shape': self.response_shape,
        }
        super().save_model_data(data)

    def load_model(self):
        data = super().load_model_data()

        # Define the network
        self.model = ConvolutionalNetworkModel.NNModel(self.stimuli_shape, self.response_shape)
        self.model.to(self.device)

        if data is not None:
            self.model.load_state_dict(data['network'])
            self.best_loss = data['best_loss']
            self.epoch = data['epoch']
            self.wandb_run_id = data['wandb_run_id']
            self.num_epochs_curr = data['num_epochs']
            self.stimuli_shape = data['stimuli_shape']
            self.response_shape = data['response_shape']

        print("printing the model summary (for debugging purposes)...")
        from torchinfo import summary
        # print(self.model)
        summary(self.model, input_size=(1, 60_000))

    def fit(self, dataloader_trn: torch.utils.data.DataLoader, dataloader_val: torch.utils.data.DataLoader):
        self.model.train()

        model, best_loss, best_epoch = copy.deepcopy(self.model), float("inf"), 0

        num_samples = len(dataloader_trn.dataset)
        num_batches_in_epoch = num_samples / self.batch_size

        # criterion_mse = nn.MSELoss(reduction='none')
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, threshold=0.0002, patience=3,
        #                                                  cooldown=4, verbose=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)

        transform_crop = None
        # transform_crop = ModelEvaluator.get_central_crop_transform()

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(self.num_epochs_curr + 1, self.num_epochs + 1):
            epoch_loss = torch.zeros((1,)).to(self.device, dtype=torch.float)
            dict_losses = defaultdict(float)

            # For each batch in the dataloader
            for i, data in enumerate(dataloader_trn, 0):
                data_stimulus = data['stimulus'].to(self.device, dtype=torch.float)  # type: torch.Tensor
                data_stimulus.unsqueeze_(1)
                data_response = data['response'].to(self.device, dtype=torch.float)

                # Prepare the network
                optimizer.zero_grad()
                # Compute the predictions
                predictions = model(data_response)
                if transform_crop is not None:
                    data_stimulus = transform_crop(data_stimulus)
                    predictions = transform_crop(predictions)
                # Compute the loss
                loss = self.criterion(data_stimulus, predictions)
                epoch_loss += loss.mean(dim=1).mean(dim=1).mean(dim=1).sum()
                loss = loss.mean()
                # Back-propagate the loss
                loss.backward()
                optimizer.step()

                # Compute all the losses
                dict_losses_ = ModelEvaluator.compute_losses(data_stimulus, predictions)
                for key, value in dict_losses_.items():
                    dict_losses[key] += value

                # Current state info
                if i < num_batches_in_epoch - 1:
                    print("   - epoch {}/{}, batch {}/{:.1f}: {} loss {}"
                          .format(epoch, self.num_epochs, i + 1, num_batches_in_epoch, self.model_loss, loss.item()))
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_epoch = epoch

            # Save the model
            if best_loss < self.best_loss:
                print(" - new loss {} is better than previous {} -> saving the new model..."
                      .format(best_loss, self.best_loss))
                self.model = model
                self.best_loss = best_loss
                self.save_model()

            # Log the progress
            epoch_loss = epoch_loss / num_samples
            dict_losses = {"train.{}".format(key): value / num_samples for key, value in dict_losses.items()}
            print(" * epoch {}/{}: {} loss {}, LR {}"
                  .format(epoch, self.num_epochs, self.model_loss, epoch_loss.item(), scheduler.state_dict()))
            wandb.log({
                "epoch": epoch,
                "train.loss": epoch_loss.item(),
                "train.best_loss": best_loss,
                "train.best_epoch": best_epoch,
                "train.learning_rate": scheduler.get_last_lr(),
            }, commit=False)
            wandb.log(dict_losses, commit=True)
            ModelEvaluator.evaluate(dataloader_val, self)
            ModelEvaluator.log_outputs(dataloader_val, self)

            # Adjust the learning rate
            scheduler.step()

            self.num_epochs_curr = epoch

        return model, best_loss, best_epoch

    def train(self, dataloader_trn: torch.utils.data.DataLoader, dataloader_val: torch.utils.data.DataLoader,
              continue_training=False):

        # Train the network if not available
        if not os.path.exists(self.checkpoint_filepath) or continue_training:
            print("Training new network...")
            model, loss, epoch = self.fit(dataloader_trn, dataloader_val)

            if loss < self.best_loss:
                print(" - new loss {} is better than previous {} -> saving the new model...".format(loss, self.best_loss))
                self.model = model
                self.best_loss = loss
                self.save_model()

    def predict(self, dataloader_tst: torch.utils.data.DataLoader):
        super().predict(dataloader_tst)

        self.model.eval()

        # Create the dataloader
        # dataloader_tst = torch.utils.data.DataLoader(
        #     LGNDataset(response_np, None, self.datanorm),
        #     batch_size=512, shuffle=False, num_workers=self.num_workers)
        print("Returning loader with", len(dataloader_tst.dataset), "samples")

        # For each batch in the dataloader
        predictions = None
        for i, data in enumerate(dataloader_tst, 0):
            data_response = data['response'].to(self.device, dtype=torch.float)

            prediction = self.model(data_response).detach().cpu().numpy()
            if predictions is None:
                predictions = prediction
            else:
                predictions = np.concatenate([predictions, prediction], axis=0)

        prediction = predictions

        return prediction

    def get_kernel(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise Exception("Model not trained")

        weights = self.model.deconv.weight
        weights = weights.view((1, weights.size()[2], weights.size()[3]))
        weights = weights.repeat(110 * 110, 1, 1)
        weights = weights.detach().cpu().numpy()

        if self.model.deconv.bias is not None:
            biases = self.model.deconv.bias.detach().cpu().numpy()
        else:
            biases = np.zeros((110, 110))

        print("weights CNN", weights.shape)

        # weights = np.reshape(weights, (-1, 51, 51))
        # biases = np.reshape(biases, (110, 110))

        return weights, biases