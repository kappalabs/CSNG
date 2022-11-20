import os
import pickle

import numpy as np

from typing import Tuple
from sklearn.linear_model import LinearRegression


class LinearRegressionModel:

    def __init__(self, checkpoint_filepath: str):
        self.checkpoint_filepath = checkpoint_filepath

        self.wandb_run_id = None
        self.num_epochs = 0

        self.model = None
        # self.model_name = self.get_name()
        # self.model_path = os.path.join(self.get_name(), subfolder)
        # self.model_filepath = os.path.join(self.model_path, 'model.pkl')

        self.stimuli_shape = None
        self.response_shape = None
        self.lr_model = None

        self.load_model()

    # @staticmethod
    # def get_name():
    #     name = "linear_regression_model"
    #
    #     return name

    @staticmethod
    def flatten(data):
        return np.reshape(data, (data.shape[0], -1))

    def save_model(self):
        os.makedirs(os.path.dirname(self.checkpoint_filepath), exist_ok=True)
        with open(self.checkpoint_filepath, 'wb') as f:
            pickle.dump({
                'model': self.lr_model,
                'wandb_run_id': self.wandb_run_id,
                'num_epochs': self.num_epochs,
                'stimuli_shape': self.stimuli_shape,
                'response_shape': self.response_shape,
            }, f)

    def load_model(self):
        if not os.path.isfile(self.checkpoint_filepath):
            print("The model {} does not exist".format(self.checkpoint_filepath))
            return

        with open(self.checkpoint_filepath, 'rb') as f:
            data = pickle.load(f)
        self.lr_model = data['model']
        self.wandb_run_id = data['wandb_run_id']
        self.num_epochs = data['num_epochs']
        self.stimuli_shape = data['stimuli_shape']
        self.response_shape = data['response_shape']

        print("Loaded model from {} with num_epochs {}".format(self.checkpoint_filepath, self.num_epochs))

    def train(self, stimuli, response):
        # Set up the model
        self.lr_model = LinearRegression(fit_intercept=False)
        self.stimuli_shape = stimuli.shape[-2:]
        self.response_shape = response.shape[-2:]

        # Fit the model
        print("fitting the model")
        response_flat = LinearRegressionModel.flatten(response)
        stimuli_flat = LinearRegressionModel.flatten(stimuli)
        self.lr_model.fit(response_flat, stimuli_flat)
        print("model fitted")
        self.num_epochs += 1

        # Save the fitted model
        self.save_model()

    def predict(self, response):
        if self.lr_model is None:
            raise Exception("Model not trained")

        flattened_prediction = self.lr_model.predict(LinearRegressionModel.flatten(response))
        prediction = flattened_prediction.reshape((-1, *self.stimuli_shape))

        return prediction

    def get_kernel(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.lr_model is None:
            raise Exception("Model not trained")

        weights = self.lr_model.coef_
        biases = np.zeros((110, 110))

        weights = np.reshape(weights, (-1, 51, 51))
        biases = np.reshape(biases, (110, 110))

        return weights, biases
