import os.path
import time

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from lgn_deconvolve.lgn_data import LGNData
from lgn_deconvolve.linear_network import LinearNetworkModel
from lgn_deconvolve.linear_regression import LinearRegressionModel
from lgn_deconvolve.convolution_network import ConvolutionalNetworkModel


class ModelEvaluator:

    @staticmethod
    def evaluate_mean_loss(gold_data, prediction_data, loss, transform):
        num_samples = gold_data.shape[0]
        sums = 0
        for sample_idx in range(num_samples):
            gold_dato = torch.from_numpy(gold_data[sample_idx]).squeeze()
            prediction_dato = torch.from_numpy(prediction_data[sample_idx]).squeeze()
            if transform is not None:
                gold_dato = transform(gold_dato)
                prediction_dato = transform(prediction_dato)
            sums += loss(gold_dato, prediction_dato)

        return sums / num_samples

    @staticmethod
    def evaluate_mse_whole(gold_data, prediction_data, transform=None):
        criterion_l1 = nn.MSELoss(reduction='none')
        loss_l1 = ModelEvaluator.evaluate_mean_loss(gold_data, prediction_data, criterion_l1, transform)

        return loss_l1

    @staticmethod
    def evaluate_l1_whole(gold_data, prediction_data, transform=None):
        criterion_l1 = nn.L1Loss(reduction='none')
        loss_l1 = ModelEvaluator.evaluate_mean_loss(gold_data, prediction_data, criterion_l1, transform)

        return loss_l1

    @staticmethod
    def evaluate(data, model):
        gold_data = data.stimuli_dataset_test
        gold_labels = data.response_dataset_test

        # Compute the predictions on testing dataset
        print("Computing predictions...")
        prediction_data = model.predict(gold_labels)
        print(" - computed the predictions on {} samples".format(data.num_test_data))

        CROP_SIZE = 64
        transform_crop = transforms.Compose([
            transforms.CenterCrop((CROP_SIZE, CROP_SIZE)),
        ])

        # Compute the losses
        loss_l1 = ModelEvaluator.evaluate_l1_whole(gold_data, prediction_data)
        print(" - L1:", loss_l1.mean().item())

        loss_l1 = ModelEvaluator.evaluate_l1_whole(gold_data, prediction_data, transform_crop)
        print(" - L1 central:", loss_l1.mean().item())

        loss_mse = ModelEvaluator.evaluate_mse_whole(gold_data, prediction_data)
        print(" - MSE:", loss_mse.mean().item())

        loss_mse = ModelEvaluator.evaluate_mse_whole(gold_data, prediction_data, transform_crop)
        print(" - MSE central:", loss_mse.mean().item())

    @staticmethod
    def save_filters(filters_dir, name_prefix, weights, biases):
        os.makedirs(filters_dir, exist_ok=True)
        print("filters_dir", filters_dir)

        for position_idx in range(110 * 110):
            if position_idx != 110 * 55 + 55:
                continue

            out_file = os.path.join(filters_dir, "{}_weight_{:04d}".format(name_prefix, position_idx))
            print("out_file", out_file)

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


def main():
    data = LGNData()

    for percent_part_100 in range(50, 100, 10):
        # Select part of the training set to train on
        percent_part = percent_part_100 / 100.
        train_samples = int(percent_part * data.num_train_data)
        print("Using {} samples for training out of {}".format(train_samples, data.num_train_data))
        train_stimuli_subset = data.stimuli_dataset_train[:train_samples]
        train_response_subset = data.response_dataset_train[:train_samples]
        percent_subfolder = "{}%".format(int(percent_part * 100))

        #
        # First model - linear regression

        # Train first model
        lrm = LinearRegressionModel(percent_subfolder)
        print("Training the LR model", lrm.model_name)
        lrm.train(train_stimuli_subset, train_response_subset)

        # Evaluate the model
        print("Evaluating LR (#train {}) model {}".format(train_samples, lrm.model_name))
        ModelEvaluator.evaluate(data, lrm)

        # Save the filters
        w, b = lrm.get_kernel()
        ModelEvaluator.save_filters(os.path.join(lrm.model_path), "deconv_filter", w, b)

        print("-------------------")

        #
        # Second model - linear network

        # Train second model
        # lnm = LinearNetworkModel(percent_subfolder, init_zeros=True, use_crop=False)
        lnm = LinearNetworkModel(percent_subfolder, init_zeros=True, use_crop=True)
        # NOTE: try to initialize with LR kernel - TEST OK -> same results as LR
        # lnm = LinearNetworkModel(model_name, init_zeros=True, use_crop=True, init_kernel=w)
        print("Training the LN model", lnm.model_name)
        lnm.train(train_stimuli_subset, train_response_subset, continue_training=False)

        # Evaluate the model
        print("Evaluating LN (#train {}) model".format(train_samples))
        ModelEvaluator.evaluate(data, lnm)

        # Save the filters
        w, b = lnm.get_kernel()
        ModelEvaluator.save_filters(os.path.join(lnm.model_path, "{}".format(time.time_ns())), "deconv_filter", w, b)

        print("-------------------")

        # #
        # # Third model - convolutional network
        #
        # # Train second model
        # model_name = "convolution_network_model_{}%".format(int(percent_part * 100))
        # cnm = ConvolutionalNetworkModel(model_name)
        # print("Training the CN model")
        # cnm.train(train_stimuli_subset, train_response_subset)
        # lnm_predictions = cnm.predict(data.response_dataset_test)
        # print("Computed CN predictions")
        #
        # # Evaluate the model
        # print("Evaluating CN (#train {}) model on {} samples".format(train_samples, data.num_test_data))
        # loss_ln_l1 = ModelEvaluator.evaluate_l1_whole(data.stimuli_dataset_test, lnm_predictions)
        # print(loss_ln_l1.mean().item())
        #
        # # Save the filters
        # w, b = cnm.get_kernel()
        # ModelEvaluator.save_filters("{}_CN".format(time.time()), "deconv_filter_CN_", w, b)
        #
        # print("-------------------")
        #
        # #
        # # Third model - linear network zeroed
        #
        # # Train second model
        # model_name = "linear_network_zeroed_model_{}%".format(int(percent_part * 100))
        # lnm = LinearNetworkModel(model_name, init_zeros=True)
        # print("Training the LN-zero model")
        # lnm.train(train_stimuli_subset, train_response_subset)
        # lnm_predictions = lnm.predict(data.response_dataset_test)
        # print("Computed LN-zero predictions")
        #
        # # Evaluate the model
        # print("Evaluating LN-zero (#train {}) model on {} samples".format(train_samples, data.num_test_data))
        # loss_ln_l1 = ModelEvaluator.evaluate_l1_whole(data.stimuli_dataset_test, lnm_predictions)
        # print(loss_ln_l1.mean().item())
        #
        # # Save the filters
        # w, b = lnm.get_kernel()
        # ModelEvaluator.save_filters("{}_LN-zero".format(time.time()), "deconv_filter_LN_", w, b)
        #
        # print("-------------------")

        print("\n===================")


if __name__ == '__main__':
    main()
