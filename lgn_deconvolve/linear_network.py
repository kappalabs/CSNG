import os
from typing import Tuple

import torch
import torch.utils.data

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

# Decide which device we want to run on
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print('device', device)


class LinearNetworkModel:

    class LGNDataset(torch.utils.data.Dataset):

        def __init__(self, response, stimuli):
            self.num_samples = response.shape[0]

            self.stimulus_dataset = stimuli
            self.response_dataset = response

            self.response_offsets = 0
            self.response_stds = 1
            self.stimulus_offsets = 0
            self.stimulus_stds = 1

            self.load_data()

        def load_data(self):
            if self.response_dataset is not None:
                shmi, shma, shme, shstd = self.response_dataset.min(), self.response_dataset.max(), \
                                          self.response_dataset.mean(), self.response_dataset.std()
                print(" - responses raw description:", shmi, shma, shme, shstd)

                self.response_offsets = -self.response_dataset.mean(axis=0)
                self.response_stds = self.response_dataset.std(axis=0)

                self.num_samples = np.shape(self.response_dataset)[0]

            if self.stimulus_dataset is not None:
                shmi, shma, shme, shstd = self.stimulus_dataset.min(), self.stimulus_dataset.max(), \
                                          self.stimulus_dataset.mean(), self.stimulus_dataset.std()
                print(" - stimuli raw description:", shmi, shma, shme, shstd)

                self.stimulus_offsets = -self.stimulus_dataset.mean(axis=0)
                self.stimulus_stds = self.stimulus_dataset.std(axis=0)

                self.num_samples = np.shape(self.stimulus_dataset)[0]

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            response = torch.zeros((1, ))
            response_raw = torch.zeros((1, ))
            if self.response_dataset is not None:
                response_raw = self.response_dataset[idx]

                response = transforms.ToTensor()(response_raw)
                # response = transforms.ToTensor()(response_raw + self.response_offsets)
                # response = transforms.ToTensor()(response_raw + self.response_offsets) / self.response_stds

            stimulus = torch.zeros((1, ))
            stimulus_raw = torch.zeros((1, ))
            if self.stimulus_dataset is not None:
                stimulus_raw = self.stimulus_dataset[idx]

                stimulus = transforms.ToTensor()(stimulus_raw)
                # stimulus = transforms.ToTensor()(stimulus_raw + self.stimulus_offsets)
                # stimulus = transforms.ToTensor()(stimulus_raw + self.stimulus_offsets) / self.stimulus_stds

            return {"stimulus": stimulus, "response": response,
                    "response_raw": response_raw, "stimulus_raw": stimulus_raw}

    class NNModel(nn.Module):

        def __init__(self, stimuli_shape, response_shape, use_bias):
            super(LinearNetworkModel.NNModel, self).__init__()

            self.stimuli_shape = stimuli_shape
            self.response_shape = response_shape

            self.fc1 = nn.Linear(response_shape[0] * response_shape[1], stimuli_shape[0] * stimuli_shape[1],
                                 bias=use_bias)

        def forward(self, x):
            x = nn.Flatten()(x)

            out_img = self.fc1(x).view(x.size(0), 1, *self.stimuli_shape)

            return out_img

    def __init__(self, subfolder: str, init_zeros=False, use_crop=False, init_kernel=None, use_bias=False, datanorm=None):
        self.stimuli_shape = None
        self.response_shape = None
        self.init_zeros = init_zeros
        self.use_crop = use_crop
        self.init_kernel = init_kernel
        self.use_bias = use_bias
        self.datanorm = datanorm

        self.learning_rate = 0.1
        self.num_epochs = 50
        self.batch_size = 512 * 12  # Fits into the GPU (<4GB)
        self.batch_size = 35000
        self.num_workers = 12

        self.model = None
        self.model_name = self.get_name()
        self.model_path = os.path.join(self.get_name(), subfolder)
        self.model_filepath = os.path.join(self.model_path, 'network.weights')

    def get_name(self):
        name = "linear_network_model"
        if self.use_bias:
            name += "_bias"
        else:
            name += "_nobias"

        if self.datanorm is None:
            name += "_nodatanorm"
        else:
            raise Exception("Unimplemented normalization")

        if self.use_crop:
            name += "_crop64x64"

        if self.init_zeros:
            name += "_init0"
        else:
            name += "_initrnd"

        return name

    def _init_zeros(self):
        nn.init.zeros_(self.model.fc1.weight.data)
        if self.model.fc1.bias is not None:
            nn.init.zeros_(self.model.fc1.bias.data)

    def _init_kernel(self, kernel):
        kernel = np.reshape(kernel, (110*110, 51*51))
        self.model.fc1.weight.data = torch.from_numpy(kernel).to(device, dtype=torch.float)
        if self.model.fc1.bias is not None:
            nn.init.zeros_(self.model.fc1.bias.data)

    def fit(self, response, stimuli):
        ln_model, best_loss, best_epoch = self.model, float("inf"), 0

        self.batch_size = response.shape[0]

        # Create the dataloader
        dataloader_trn = torch.utils.data.DataLoader(
            LinearNetworkModel.LGNDataset(response, stimuli),
            batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        print("Returning loader with", len(dataloader_trn.dataset), "samples")
        num_samples = len(dataloader_trn.dataset)
        num_batches_in_epoch = num_samples / self.batch_size

        criterion_mse = nn.MSELoss(reduction='none')
        # optimizer = optim.Adam(ln_model.parameters(), lr=self.learning_rate, weight_decay=0)
        # optimizer = optim.SGD(ln_model.parameters(), lr=self.learning_rate, weight_decay=0, momentum=0.1)
        optimizer = optim.SGD(ln_model.parameters(), lr=self.learning_rate, weight_decay=0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, threshold=0.0002, patience=5,
                                                         cooldown=4, verbose=True)

        transform_crop = None
        if self.use_crop:
            CROP_SIZE = 64
            transform_crop = transforms.Compose([
                transforms.CenterCrop((CROP_SIZE, CROP_SIZE)),
            ])

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(self.num_epochs):
            epoch_mse_loss = torch.zeros((1, )).to(device, dtype=torch.float)
            # For each batch in the dataloader
            for i, data in enumerate(dataloader_trn, 0):
                data_stimulus = data['stimulus'].to(device, dtype=torch.float)
                data_response = data['response'].to(device, dtype=torch.float)

                # Prepare the network
                optimizer.zero_grad()
                # Compute the predictions
                predictions = ln_model(data_response)
                if transform_crop is not None:
                    data_stimulus = transform_crop(data_stimulus)
                    predictions = transform_crop(predictions)
                # Compute the loss
                loss_mse = criterion_mse(data_stimulus, predictions)
                epoch_mse_loss += loss_mse.mean(dim=1).mean(dim=1).mean(dim=1).sum()
                loss_mse = loss_mse.mean()
                # Backpropagate the loss
                loss_mse.backward()
                optimizer.step()

                # Current state info
                print("   - epoch {}/{}, batch {}/{:.1f}: MSE loss {}"
                      .format(epoch + 1, self.num_epochs, i + 1, num_batches_in_epoch, loss_mse.item()))
                if loss_mse.item() < best_loss:
                    best_loss = loss_mse.item()
                    best_epoch = epoch
            epoch_mse_loss = epoch_mse_loss / num_samples
            print(" * epoch {}/{}: MSE loss {}, LR {}".format(epoch + 1, self.num_epochs, epoch_mse_loss.item(),
                                                              scheduler.state_dict()))

            # Adjust the learning rate
            if epoch >= 5:
                scheduler.step(epoch_mse_loss)

        return ln_model, best_loss, best_epoch

    def load(self, stimuli_shape, response_shape):
        # Define the network
        self.model = LinearNetworkModel.NNModel(stimuli_shape, response_shape, self.use_bias)
        self.model.to(device)

        if self.init_zeros:
            self._init_zeros()
        if self.init_kernel is not None:
            self._init_kernel(self.init_kernel)

        best_loss = float("inf")
        if os.path.isfile(self.model_filepath):
            checkpoint = torch.load(self.model_filepath)
            best_loss = checkpoint['best_loss']
            print("Loaded network with best loss {}, epoch {}".format(best_loss, checkpoint['epoch']))
            self.model.load_state_dict(checkpoint['network'])

        return best_loss

    def train(self, stimuli, response, continue_training=False):
        self.stimuli_shape = stimuli.shape[-2:]
        self.response_shape = response.shape[-2:]

        best_loss = self.load(self.stimuli_shape, self.response_shape)
        self.model.train()

        # Train the network if not available
        if not os.path.exists(self.model_filepath) or continue_training:
            print("Training new network...")
            ln_model, loss, epoch = self.fit(response, stimuli)

            if loss < best_loss:
                print(" - new loss {} is better than previous {} -> saving the new model...".format(loss, best_loss))
                best_loss = loss
                state = {
                    'network': ln_model.state_dict(),
                    'best_loss': best_loss,
                    'epoch': epoch,
                }
                os.makedirs(os.path.dirname(self.model_filepath), exist_ok=True)
                torch.save(state, self.model_filepath)
                self.model = ln_model

    def predict(self, response_np):
        if self.model is None:
            raise Exception("Model not trained")

        self.model.eval()

        # Create the dataloader
        dataloader_tst = torch.utils.data.DataLoader(
            LinearNetworkModel.LGNDataset(response_np, None),
            batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        print("Returning loader with", len(dataloader_tst.dataset), "samples")

        # For each batch in the dataloader
        predictions = None
        for i, data in enumerate(dataloader_tst, 0):
            data_response = data['response'].to(device, dtype=torch.float)

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

        weights = self.model.fc1.weight.detach().cpu().numpy()
        if self.model.fc1.bias is not None:
            biases = self.model.fc1.bias.detach().cpu().numpy()
        else:
            biases = np.zeros((110, 110))

        weights = np.reshape(weights, (-1, 51, 51))
        biases = np.reshape(biases, (110, 110))

        return weights, biases
