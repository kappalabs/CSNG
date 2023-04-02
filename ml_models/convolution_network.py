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

from typing import Tuple
from collections import defaultdict
from ml_models.model_base import ModelBase
from lgn_deconvolve.model_evaluator import ModelEvaluator


class ConvolutionalNetworkModel(ModelBase):

    def __init__(self, checkpoint_filepath: str, device: torch.device, config: dict,
                 data_stimuli_shape: tuple, data_response_shape: tuple, version: int = 1):
        super().__init__(checkpoint_filepath, device, config, data_stimuli_shape, data_response_shape)

        self.version = version

        self.model_loss = config['model_loss']
        self.best_loss = float("inf")
        self.epoch = 0

        self.criterion = None
        if self.model_loss == 'L1':
            self.criterion = nn.L1Loss(reduction='none')
        elif self.model_loss == 'MSE':
            self.criterion = nn.MSELoss(reduction='none')
        elif self.model_loss == 'SSIM':
            self.criterion = kornia.losses.SSIMLoss(window_size=3, reduction='none')
            self.criterion.to(self.device)
        elif self.model_loss == 'MSSSIM':
            self.criterion = kornia.losses.MS_SSIMLoss(reduction='none')
            self.criterion.to(self.device)
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
            'version': self.version,
        }
        super().save_model_data(data)

    def load_model(self):
        data = super().load_model_data()

        # Define the network
        module_model = importlib.import_module("ml_models.conv_models.model_v{}".format(self.version))
        CNNModel = getattr(module_model, "CNNModel")
        self.model = CNNModel(self.stimuli_shape, self.response_shape)
        self.model.to(self.device)

        if data is not None:
            self.model.load_state_dict(data['network'])
            self.best_loss = data['best_loss']
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

        eval_criteria = ModelEvaluator.get_criteria(self.device)

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(self.num_epochs_curr + 1, self.num_epochs + 1):
            epoch_loss = torch.zeros((1,)).to(self.device, dtype=torch.float)
            dict_losses = defaultdict(float)

            # For each batch in the dataloader
            for i, data in enumerate(dataloader_trn, 0):
                stimuli = data['stimulus'].to(self.device, dtype=torch.float)
                responses = data['response'].to(self.device, dtype=torch.float)

                # Prepare the network
                optimizer.zero_grad()
                # Compute the predictions
                predictions = model(responses)
                if transform_crop is not None:
                    stimuli = transform_crop(stimuli)
                    predictions = transform_crop(predictions)
                # Compute the loss
                loss = self.criterion(stimuli, predictions)
                if len(loss.shape) == 4:
                    epoch_loss += loss.mean(dim=1).mean(dim=1).mean(dim=1).sum()
                if len(loss.shape) == 3:
                    epoch_loss += loss.mean(dim=1).mean(dim=1).sum()
                loss = loss.mean()
                # Back-propagate the loss
                loss.backward()
                optimizer.step()

                # Compute all the losses
                dict_losses_ = ModelEvaluator.compute_losses(stimuli, predictions, self.device, eval_criteria)
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
            wandb.log(dict_losses, commit=False)
            evaluate_dict = ModelEvaluator.evaluate(dataloader_val, self, log_dict_prefix='val.')
            wandb.log(evaluate_dict, commit=False)
            outputs_dict = ModelEvaluator.log_outputs(dataloader_val, self, log_dict_prefix='val.')
            wandb.log(outputs_dict, commit=True)

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

    def predict_batch(self, batch: torch.FloatTensor) -> torch.FloatTensor:
        super().predict_batch(batch)

        self.model.eval()

        data_response = batch.to(self.device, dtype=torch.float)

        prediction = self.model(data_response).detach()

        return prediction

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
