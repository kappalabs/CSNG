import os
import pickle
from typing import Tuple

import numpy as np

from sklearn.linear_model import LinearRegression


class LinearRegressionModel:

    def __init__(self, subfolder: str):
        self.model = None
        self.model_name = self.get_name()
        self.model_path = os.path.join(self.get_name(), subfolder)
        self.model_filepath = os.path.join(self.model_path, 'model.pkl')

        self.stimuli_shape = None
        self.response_shape = None
        self.lr_model = None

    def get_name(self):
        name = "linear_regression_model"

        return name

    @staticmethod
    def flatten(data):
        return np.reshape(data, (data.shape[0], -1))

    def train(self, stimuli, response):
        lr_model = LinearRegression(fit_intercept=False)
        self.stimuli_shape = stimuli.shape[-2:]
        self.response_shape = response.shape[-2:]

        if not os.path.isfile(self.model_filepath):
            lr_model.fit(LinearRegressionModel.flatten(response), LinearRegressionModel.flatten(stimuli))
            os.makedirs(os.path.dirname(self.model_filepath), exist_ok=True)
            with open(self.model_filepath, 'wb') as f:
                pickle.dump(lr_model, f)
            self.lr_model = lr_model

            return
        with open(self.model_filepath, 'rb') as f:
            self.lr_model = pickle.load(f)

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
        biases = self.lr_model.singular_

        weights = np.reshape(weights, (-1, 51, 51))
        biases = np.reshape(biases, (51, 51))

        return weights, biases
