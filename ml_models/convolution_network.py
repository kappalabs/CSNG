import os
import copy
import wandb
import torch
import importlib
import kornia.losses
import torch.utils.data

import numpy as np
import torch.nn as nn
import torch.optim as optim
import kornia.augmentation as K

from tqdm import tqdm
from typing import Tuple
from torchinfo import summary
from collections import defaultdict
from ml_models.model_base import ModelBase
from lgn_deconvolve.model_evaluator import ModelEvaluator


class ConvolutionalNetworkModel(ModelBase):

    def __init__(self, checkpoint_filepath: str, device: torch.device, config: dict,
                 data_stimuli_shape: tuple, data_response_shape: tuple):
        super().__init__(checkpoint_filepath, device, config, data_stimuli_shape, data_response_shape)

        self.model_loss = config['model_loss']
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.epoch = 0

        # Prepare the criterion
        self.criterion = None
        self.criterion_msssim = kornia.losses.MS_SSIMLoss(reduction='none').to(self.device)
        if self.model_loss == 'L1':
            self.criterion = nn.L1Loss(reduction='none').to(self.device)
        elif self.model_loss == 'MSE':
            self.criterion = nn.MSELoss(reduction='none').to(self.device)
        elif self.model_loss == 'SSIM':
            self.criterion = kornia.losses.SSIMLoss(window_size=3, reduction='none').to(self.device)
        elif self.model_loss == 'MSSSIM':
            self.criterion = kornia.losses.MS_SSIMLoss(reduction='none').to(self.device)
        elif self.model_loss == 'MIX':
            self.criterion_l1 = nn.L1Loss(reduction='none').to(self.device)
            self.criterion_msssim = kornia.losses.MS_SSIMLoss(reduction='none').to(self.device)
            alpha = 0.84
            self.criterion = lambda x, y: alpha * self.criterion_msssim(x, y) + (1 - alpha) * self.criterion_l1(x, y)
        elif self.model_loss == 'D1':
            from ml_models.losses.discriminator_v1 import Discriminator
            self.criterion = Discriminator().to(self.device)

            print("printing the Discriminator model summary (for debugging purposes)...")
            summary(self.criterion, input_size=(1, *self.stimuli_shape))
        else:
            raise Exception("Unknown loss function: " + self.model_loss)

        # Prepare the model transformations
        self.train_stimuli_transform = nn.Sequential()
        self.random_erasing = config['random_erasing']
        if self.random_erasing:
            transform_ = K.RandomErasing(p=0.1, value=0.5)
            self.train_stimuli_transform.add_module('RandomErasing', transform_)
        self.random_gaussian_noise = config['random_gaussian_noise']
        if self.random_gaussian_noise:
            transform_ = K.RandomGaussianNoise(p=0.1, mean=0.0, std=0.1)
            self.train_stimuli_transform.add_module('RandomGaussianNoise', transform_)
        self.train_stimuli_transform.to(self.device)

        self.load_model()

        if self.config['clear_progress']:
            self.best_loss = float("inf")
            self.epoch = 0
            self.num_epochs_curr = 0

    def save_model(self):
        data = {
            'network': self.model.state_dict(),
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'epoch': self.epoch,
            'wandb_run_id': self.wandb_run_id,
            'num_epochs': self.num_epochs_curr,
            'stimuli_shape': self.stimuli_shape,
            'response_shape': self.response_shape,
            'version': self.version,
        }
        super().save_model_data(data)

    def load_model(self):
        data = super().load_model_data()

        # Define the network
        module_model = importlib.import_module("ml_models.conv_models.model_v{}".format(self.version))
        CNNModel = getattr(module_model, "CNNModel")
        self.model = CNNModel(self.stimuli_shape, self.response_shape, self.dropout)
        self.model.to(self.device)

        if data is not None:
            self.model.load_state_dict(data['network'])
            self.best_loss = data['best_loss']
            self.best_epoch = data['best_epoch']
            self.epoch = data['epoch']
            self.wandb_run_id = data['wandb_run_id']
            self.num_epochs_curr = data['num_epochs']
            self.stimuli_shape = data['stimuli_shape']
            self.response_shape = data['response_shape']
            saved_version = data['version']

            if saved_version != self.version:
                raise Exception("Saved model version ({}) does not match current model version ({})".format(
                    saved_version, self.version))

        print("printing the model summary (for debugging purposes)...")
        summary(self.model, input_size=(1, np.prod(self.response_shape)))

    def fit(self, dataloader_trn: torch.utils.data.DataLoader, dataloader_val: torch.utils.data.DataLoader):
        self.model.train()

        model, best_loss, best_epoch = copy.deepcopy(self.model), float("inf"), 0

        num_samples = len(dataloader_trn.dataset)
        num_batches_in_epoch = num_samples / self.batch_size

        # criterion_mse = nn.MSELoss(reduction='none')
        if self.optimizer.lower() == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-4)
        elif self.optimizer.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        else:
            raise Exception("Unknown optimizer: " + self.optimizer)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, threshold=0.0002, patience=3,
        #                                                  cooldown=4, verbose=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)

        transform_crop = None
        # transform_crop = ModelEvaluator.get_central_crop_transform()

        eval_criteria = ModelEvaluator.get_criteria(self.device)

        criterion_bce = nn.BCELoss()

        # Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.
        if self.model_loss == 'D1':
            optimizerD = optim.Adam(self.criterion.parameters(), lr=self.learning_rate, weight_decay=1e-4)

            # eval_criteria.append(('D1', lambda x, y: criterion_bce(self.criterion(y),
            #                                                        torch.ones_like(self.criterion(y)))))
            eval_criteria.append(('D1', lambda x, y: self.criterion(y)))

        print("Starting Training Loop...")
        # For each epoch
        pbar_epoch = tqdm(total=self.num_epochs - self.num_epochs_curr, unit='epoch', colour="green", ascii=True)
        for epoch in range(self.num_epochs_curr + 1, self.num_epochs + 1):
            epoch_loss = torch.zeros((1,)).to(self.device, dtype=torch.float)
            dict_losses = defaultdict(float)

            # For each batch in the dataloader
            for i, data in enumerate(dataloader_trn, 0):
                stimuli = data['stimulus'].to(self.device, dtype=torch.float)
                responses = data['response'].to(self.device, dtype=torch.float)

                # Perform the transforms
                stimuli = self.train_stimuli_transform(stimuli)

                if self.model_loss == 'D1':
                    ############################
                    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    ###########################
                    ## Train with all-real batch
                    self.criterion.zero_grad()
                    b_size = stimuli.size(0)
                    label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)
                    # Forward pass real batch through D
                    output = self.criterion(stimuli).view(-1)
                    # Calculate loss on all-real batch
                    errD_real = criterion_bce(output, label)
                    # Calculate gradients for D in backward pass
                    errD_real.backward()
                    D_x = output.mean().item()
                    dict_losses['D_x'] += D_x

                    ## Train with all-fake batch
                    # # Generate fake image batch with G
                    predictions = model(responses)
                    label.fill_(fake_label)
                    # Classify all fake batch with D
                    output = self.criterion(predictions.detach()).view(-1)
                    # Calculate D's loss on the all-fake batch
                    errD_fake = criterion_bce(output, label)
                    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                    errD_fake.backward()
                    D_G_z1 = output.mean().item()
                    dict_losses['D_G_z1'] += D_G_z1
                    # Compute error of D as sum over the fake and the real batches
                    errD = errD_real + errD_fake
                    dict_losses['errD'] += errD.item()
                    # Update D
                    optimizerD.step()

                    ############################
                    # (2) Update G network: maximize log(D(G(z)))
                    ###########################
                    model.zero_grad()
                    label.fill_(real_label)  # fake labels are real for generator cost
                    # Since we just updated D, perform another forward pass of all-fake batch through D
                    output = self.criterion(predictions).view(-1)
                    # Calculate G's loss based on this output
                    alpha = 0.8
                    errG = alpha * criterion_bce(output, label) + \
                           (1-alpha) * self.criterion_msssim(stimuli, predictions).mean(dim=1).mean(dim=1).sum()
                    # Calculate gradients for G
                    errG.backward()
                    D_G_z2 = output.mean().item()
                    dict_losses['D_G_z2'] += D_G_z2
                    # Update G
                    optimizer.step()

                    loss = errG
                else:
                    # Prepare the network
                    optimizer.zero_grad()
                    # Compute the predictions
                    output = model(responses)
                    if type(output) is tuple:
                        predictions = output[0]
                    else:
                        predictions = output

                    if transform_crop is not None:
                        stimuli = transform_crop(stimuli)
                        predictions = transform_crop(predictions)
                    # Compute the loss
                    loss = self.criterion(stimuli, predictions)
                    if len(loss.shape) == 4:
                        epoch_loss += loss.detach().mean(dim=1).mean(dim=1).mean(dim=1).sum()
                    if len(loss.shape) == 3:
                        epoch_loss += loss.detach().mean(dim=1).mean(dim=1).sum()
                    loss = loss.mean()
                    # Back-propagate the loss
                    loss.backward()
                    optimizer.step()

                # Compute all the losses
                self.criterion.eval()
                dict_losses_ = ModelEvaluator.compute_losses(stimuli, predictions, self.device, eval_criteria)
                for key, value in dict_losses_.items():
                    dict_losses[key] += value
                self.criterion.train()

                # Current state info
                if i < num_batches_in_epoch - 1:
                    print("   - epoch {}/{}, batch {}/{:.1f}: {} loss {}"
                          .format(epoch, self.num_epochs, i + 1, num_batches_in_epoch, self.model_loss, loss.item()))

            # Make evaluation on the validation set of the updated model
            # NOTE: deepcopy is probably not necessary, but it's safer for now
            prev_model = copy.deepcopy(self.model)
            self.model = model
            self.criterion.eval()
            evaluate_dict = ModelEvaluator.evaluate(dataloader_val, self,
                                                    criteria=eval_criteria, log_dict_prefix='val.')
            outputs_dict = ModelEvaluator.log_outputs(dataloader_val, self, log_dict_prefix='val.')
            self.criterion.train()

            # Update the best loss & model
            validation_loss = evaluate_dict['val.{}'.format(self.model_loss)]

            # Save the model based on the validation loss
            if validation_loss < self.best_loss:
                print(" - new loss {} is better than previous {} -> saving the new model..."
                      .format(validation_loss, self.best_loss))
                self.model = copy.deepcopy(model)
                self.best_loss = validation_loss
                self.best_epoch = epoch
                self.save_model()
            else:
                print(" - new loss {} is worse than previous {} -> keeping the previous model..."
                      .format(validation_loss, self.best_loss))
                self.model = prev_model

            # Log the progress
            epoch_loss = epoch_loss / num_samples
            dict_losses = {"train.{}".format(key): value / num_samples for key, value in dict_losses.items()}
            print(" * epoch {}/{}: {} loss {}, LR {}"
                  .format(epoch, self.num_epochs, self.model_loss, epoch_loss.item(), scheduler.state_dict()))
            wandb.log({
                "epoch": epoch,
                "train.loss": epoch_loss.item(),
                "train.best_loss": self.best_loss,
                "train.best_epoch": self.best_epoch,
                "train.learning_rate": scheduler.get_last_lr(),
            }, commit=False)
            wandb.log(dict_losses, commit=False)
            wandb.log(evaluate_dict, commit=False)
            wandb.log(outputs_dict, commit=True)

            # Adjust the learning rate
            scheduler.step()

            self.num_epochs_curr = epoch

            pbar_epoch.update(1)
        pbar_epoch.close()

    def train(self, dataloader_trn: torch.utils.data.DataLoader, dataloader_val: torch.utils.data.DataLoader,
              continue_training=False):

        # Train the network if not available
        if not os.path.exists(self.checkpoint_filepath) or continue_training:
            print("Training new network...")
            self.fit(dataloader_trn, dataloader_val)

    def predict_batch(self, batch: torch.FloatTensor) -> torch.FloatTensor:
        super().predict_batch(batch)

        self.model.eval()

        data_response = batch.to(self.device, dtype=torch.float)

        output = self.model(data_response)
        if type(output) is tuple:
            prediction = output[0].detach()
        else:
            prediction = output.detach()

        return prediction

    def predict_batch_with_intermediate(self, batch: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        super().predict_batch_with_intermediate(batch)

        self.model.eval()

        data_response = batch.to(self.device, dtype=torch.float)

        output = self.model(data_response)
        if type(output) is tuple:
            prediction = output[0].detach()
            intermediate = output[1].detach()
        else:
            prediction = output.detach()
            intermediate = None

        return prediction, intermediate

    def predict(self, dataloader: torch.utils.data.DataLoader) -> torch.FloatTensor:
        super().predict(dataloader)

        self.model.eval()

        print("Received loader with", len(dataloader.dataset), "samples")

        # For each batch in the dataloader
        predictions = None
        for i, data in enumerate(dataloader, 0):
            data_response = data['response']

            prediction = self.predict_batch(data_response)

            # Concatenate the predictions
            if predictions is None:
                predictions = prediction
            else:
                predictions = torch.cat((predictions, prediction), dim=0)

        return predictions

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
