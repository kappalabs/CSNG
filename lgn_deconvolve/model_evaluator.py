import os.path
import time

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from lgn_deconvolve.lgn_data import LGNData
from lgn_deconvolve.linear_network import LinearNetworkModel
from lgn_deconvolve.linear_regression import LinearRegressionModel


class ModelEvaluator:

    @staticmethod
    def evaluate_mean_loss(gold_data, prediction_data, loss):
        num_samples = gold_data.shape[0]
        sums = torch.zeros((gold_data.shape[1], gold_data.shape[2]))
        for sample_idx in range(num_samples):
            gold_dato = torch.from_numpy(gold_data[sample_idx]).squeeze()
            prediction_dato = torch.from_numpy(prediction_data[sample_idx]).squeeze()
            sums += loss(gold_dato, prediction_dato)

        return sums / num_samples

    @staticmethod
    def evaluate_l1_whole(gold_data, prediction_data):
        criterion_l1 = nn.L1Loss(reduction='none')
        loss_l1 = ModelEvaluator.evaluate_mean_loss(gold_data, prediction_data, criterion_l1)

        return loss_l1

    @staticmethod
    def save_filters(filters_dir, name_prefix, weights, biases):
        if not os.path.exists(filters_dir):
            os.mkdir(filters_dir)

        for position_idx in range(110 * 110):
            if position_idx != 110 * 55 + 55:
                continue

            out_file = os.path.join(filters_dir, "{}_weight_{:04d}".format(name_prefix, position_idx))

            plt.figure(figsize=(5, 5))
            plt.subplot(1, 1, 1)
            plt.title("Weights")
            kernel = np.reshape(weights[position_idx], (51, 51))
            plt.imshow(kernel)
            plt.colorbar()
            plt.savefig(out_file)
            plt.close()

        out_file = os.path.join(filters_dir, "{}_bias".format(name_prefix))
        plt.figure(figsize=(5, 5))
        plt.subplot(1, 1, 1)
        plt.title("Biases")
        plt.imshow(biases)
        plt.colorbar()
        plt.savefig(out_file)
        plt.close()


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

        # Train first model
        model_name = "linear_regression_model_{}%.pkl".format(int(percent_part * 100))
        lrm = LinearRegressionModel(model_name)
        print("Training the LR model")
        lrm.train(train_stimuli_subset, train_response_subset)
        lr_predictions = lrm.predict(data.response_dataset_test)
        print("Computed LR predictions")

        # Evaluate the model
        print("Evaluating LR (#train {}) model on {} samples".format(train_samples, data.num_test_data))
        loss_lr_l1 = ModelEvaluator.evaluate_l1_whole(data.stimuli_dataset_test, lr_predictions)
        print(loss_lr_l1.mean().item())

        # Save the filters
        w, b = lrm.get_kernel()
        ModelEvaluator.save_filters("{}_LR".format(time.time()), "deconv_filter_LR_", w, b)

        print("-------------------")

        #
        # Second model - linear network

        # Train second model
        model_name = "linear_network_model_{}%".format(int(percent_part * 100))
        lnm = LinearNetworkModel(model_name)
        print("Training the LN model")
        lnm.train(train_stimuli_subset, train_response_subset)
        lnm_predictions = lnm.predict(data.response_dataset_test)
        print("Computed LN predictions")

        # Evaluate the model
        print("Evaluating LN (#train {}) model on {} samples".format(train_samples, data.num_test_data))
        loss_ln_l1 = ModelEvaluator.evaluate_l1_whole(data.stimuli_dataset_test, lnm_predictions)
        print(loss_ln_l1.mean().item())

        # Save the filters
        w, b = lnm.get_kernel()
        ModelEvaluator.save_filters("{}_LN".format(time.time()), "deconv_filter_LN_", w, b)

        print("-------------------")

        print("\n===================")
