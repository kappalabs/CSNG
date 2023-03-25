import torch

import numpy as np

from typing import Tuple
from ml_models.model_base import ModelBase
from sklearn.linear_model import LinearRegression


class LinearRegressionModel(ModelBase):

    def __init__(self, checkpoint_filepath: str, device: torch.device, config: dict,
                 data_stimuli_shape: tuple, data_response_shape: tuple):
        super().__init__(checkpoint_filepath, device, config, data_stimuli_shape, data_response_shape)

        self.load_model_data()

    @staticmethod
    def flatten(data):
        return np.reshape(data, (data.shape[0], -1))

    def save_model(self):
        data = {
            'model': self.model,
            'wandb_run_id': self.wandb_run_id,
            'num_epochs': self.num_epochs_curr,
        }
        super().save_model_data(data)

    def load_model(self):
        data = super().load_model_data()

        self.model = data['model']
        self.wandb_run_id = data['wandb_run_id']
        self.num_epochs_curr = data['num_epochs']

    def train(self, dataloader_trn: torch.utils.data.DataLoader, dataloader_val: torch.utils.data.DataLoader):
        # Set up the model
        self.model = LinearRegression()

        gold_stimuli = None
        gold_response = None
        for batch_idx, batch in enumerate(dataloader_trn):
            stimulus_, response_ = batch['stimulus'], batch['response']
            if gold_stimuli is None:
                gold_stimuli = stimulus_.numpy()
                gold_response = response_.numpy()
            else:
                gold_stimuli = np.concatenate((gold_stimuli, stimulus_.numpy()), axis=0)
                gold_response = np.concatenate((gold_response, response_.numpy()), axis=0)

        # Fit the model
        print("fitting the model")
        response_flat = LinearRegressionModel.flatten(gold_response)
        stimuli_flat = LinearRegressionModel.flatten(gold_stimuli)
        self.model.fit(response_flat, stimuli_flat)
        print("model fitted")
        self.num_epochs_curr += 1

        # Save the fitted model
        self.save_model()

    def predict(self, dataloader: torch.utils.data.DataLoader):
        super().predict(dataloader)

        # For each batch in the dataloader
        predictions = None
        for batch_idx, batch in enumerate(dataloader):
            # Convert torch tensor to numpy array
            data_response = batch['response'].detach().cpu().numpy()

            flattened_prediction = self.model.predict(LinearRegressionModel.flatten(data_response))
            prediction = flattened_prediction.reshape((-1, *self.stimuli_shape))

            if predictions is None:
                predictions = prediction
            else:
                predictions = np.concatenate([predictions, prediction], axis=0)

        prediction = predictions

        return prediction

    def get_kernel(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise Exception("Model not trained")

        weights = self.model.coef_
        biases = np.zeros((110, 110))

        weights = np.reshape(weights, (-1, 51, 51))
        biases = np.reshape(biases, (110, 110))

        return weights, biases
