import torch
import torch.nn as nn

from lgn_deconvolve.lgn_data import LGNData
from lgn_deconvolve.linear_regression import LinearModel


class ModelEvaluator:

    @staticmethod
    def evaluate_l1_whole(gold_data, prediction_data):
        criterion_l1 = nn.L1Loss(reduction='none')
        gold_data_torch = torch.from_numpy(gold_data)
        prediction_data_torch = torch.from_numpy(prediction_data)
        loss_l1 = criterion_l1(gold_data_torch, prediction_data_torch)

        return loss_l1


if __name__ == '__main__':
    data = LGNData()

    for percent_part_100 in range(10, 110, 10):
        # Select part of the training set to train on
        percent_part = percent_part_100 / 100.
        train_samples = int(percent_part * data.num_train_data)
        print("Using {} samples for training out of {}".format(train_samples, data.num_train_data))
        train_stimuli_subset = data.stimuli_dataset_train[:train_samples]
        train_response_subset = data.response_dataset_train[:train_samples]

        #
        # First model - linear regression

        # Train first model -
        model_name = "linear_regression_model_{}%.pkl".format(int(percent_part * 100))
        lm = LinearModel(model_name)
        print("Training the model")
        lm.train(train_stimuli_subset, train_response_subset)
        lr_predictions = lm.predict(data.response_dataset_test)
        print("Computed predictions")

        # Evaluate the first model
        print("Evaluating model on {} samples".format(data.num_test_data))
        loss_lr_l1 = ModelEvaluator.evaluate_l1_whole(data.stimuli_dataset_test, lr_predictions)
        print(loss_lr_l1.mean().item())
        print("-------------------")
