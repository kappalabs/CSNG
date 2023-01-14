import os.path
import time

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import wandb

from lgn_deconvolve.lgn_data import LGNData
from ml_models.linear_network import LinearNetworkModel
from ml_models.linear_regression import LinearRegressionModel
from ml_models.convolution_network import ConvolutionalNetworkModel


# Decide which device we want to run on
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print('device', device)


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
        criterion_mse = nn.MSELoss(reduction='none')
        loss_l1 = ModelEvaluator.evaluate_mean_loss(gold_data, prediction_data, criterion_mse, transform)

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
            # CentralPxCropTransform(),
        ])

        log_dict = {"test": {}}

        # Compute the losses
        loss_l1 = ModelEvaluator.evaluate_l1_whole(gold_data, prediction_data)
        print(" - L1:", loss_l1.mean().item())
        log_dict['test']['L1'] = float(loss_l1.mean().item())

        loss_l1 = ModelEvaluator.evaluate_l1_whole(gold_data, prediction_data, transform_crop)
        print(" - L1 central:", loss_l1.mean().item())
        log_dict['test']['L1_central'] = float(loss_l1.mean().item())

        loss_mse = ModelEvaluator.evaluate_mse_whole(gold_data, prediction_data)
        print(" - MSE:", loss_mse.mean().item())
        log_dict['test']['MSE'] = float(loss_mse.mean().item())

        loss_mse = ModelEvaluator.evaluate_mse_whole(gold_data, prediction_data, transform_crop)
        print(" - MSE central:", loss_mse.mean().item())
        log_dict['test']['MSE_central'] = float(loss_mse.mean().item())

        wandb.log(log_dict, commit=True)

    @staticmethod
    def save_filters(filters_dir, name_prefix, weights, biases):
        os.makedirs(filters_dir, exist_ok=True)
        print("filters_dir", filters_dir)
        weights_shape = weights.shape

        for position_idx in range(110 * 110):
            if position_idx != 110 * 55 + 55:
                continue

            out_file = os.path.join(filters_dir, "{}_weight_{:04d}".format(name_prefix, position_idx))
            print("out_file", out_file)

            plt.figure(figsize=(5, 5))
            plt.subplot(1, 1, 1)
            plt.title("Weights")
            # kernel = np.reshape(weights[position_idx], (51, 51))
            kernel = weights[position_idx]
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

    @staticmethod
    def manual_output(predictions_dir, name_prefix, weights, biases, data, num_save=16):
        os.makedirs(predictions_dir, exist_ok=True)
        print("predictions_dir", predictions_dir)

        gold_stimuli = data.stimuli_dataset_test[:num_save]
        gold_responses = data.response_dataset_test[:num_save]

        criterion_mse = nn.MSELoss(reduction='none')
        criterion_l1 = nn.L1Loss(reduction='none')

        weights = np.reshape(weights, (-1, 51 * 51))
        for response_idx, gold_response in enumerate(gold_responses):
            gold_response = np.reshape(gold_response, (51 * 51, 1))
            prediction = weights @ gold_response
            prediction = np.reshape(prediction, (110, 110)) + biases

            out_file = os.path.join(predictions_dir, "{}_{}".format(name_prefix, response_idx))

            fig, axs = plt.subplots(2, 2)
            fig.tight_layout()

            # Plot stimuli
            ax = axs[0, 0]
            ax.title.set_text("Stimuli")
            im = ax.imshow(gold_stimuli[response_idx])
            plt.colorbar(im, ax=ax)

            # Plot the model prediction
            ax = axs[1, 0]
            ax.title.set_text("Prediction")
            im = ax.imshow(prediction)
            plt.colorbar(im, ax=ax)

            gold_stimuli_torch = torch.from_numpy(gold_stimuli[response_idx]).squeeze()
            prediction_torch = torch.from_numpy(prediction).squeeze()

            loss_mse = criterion_mse(gold_stimuli_torch, prediction_torch)
            loss_l1 = criterion_l1(gold_stimuli_torch, prediction_torch)

            # Plot L1 loss
            ax = axs[0, 1]
            ax.title.set_text("Loss L1")
            im = ax.imshow(loss_l1)
            plt.colorbar(im, ax=ax)

            # Plot MSE loss
            ax = axs[1, 1]
            ax.title.set_text("Loss MSE")
            im = ax.imshow(loss_mse)
            plt.colorbar(im, ax=ax)

            plt.savefig(out_file)
            plt.close()

    @staticmethod
    def save_outputs(predictions_dir, name_prefix, data, model, num_save=16):
        os.makedirs(predictions_dir, exist_ok=True)
        print("predictions_dir", predictions_dir)

        gold_stimuli = data.stimuli_dataset_test[:num_save]
        gold_responses = data.response_dataset_test[:num_save]

        # Compute the predictions on testing dataset
        print("Computing predictions...")
        predictions_stimuli = model.predict(gold_responses)
        print(" - computed the predictions on {} samples".format(data.num_test_data))

        criterion_mse = nn.MSELoss(reduction='none')
        criterion_l1 = nn.L1Loss(reduction='none')

        CROP_SIZE = 64
        transform_crop = transforms.Compose([
            transforms.CenterCrop((CROP_SIZE, CROP_SIZE)),
        ])

        for response_idx, prediction_stimuli in enumerate(predictions_stimuli):
            prediction_stimuli = np.squeeze(prediction_stimuli)
            out_file = os.path.join(predictions_dir, "{}_{}".format(name_prefix, response_idx))

            gold_stimulus_torch = torch.from_numpy(gold_stimuli[response_idx]).squeeze()
            gold_stimulus_torch_crop = transform_crop(gold_stimulus_torch)
            prediction_stimuli_torch = torch.from_numpy(prediction_stimuli).squeeze()
            prediction_stimuli_torch_crop = transform_crop(prediction_stimuli_torch)

            fig, axs = plt.subplots(2, 2)
            fig.tight_layout()

            # Plot stimuli
            ax = axs[0, 0]
            ax.title.set_text("Stimuli")
            im = ax.imshow(gold_stimulus_torch_crop)
            plt.colorbar(im, ax=ax)
            im.set_clim(gold_stimulus_torch_crop.min(), gold_stimulus_torch_crop.max())

            # Plot the model prediction
            ax = axs[1, 0]
            ax.title.set_text("Prediction")
            im = ax.imshow(prediction_stimuli_torch_crop)
            plt.colorbar(im, ax=ax)
            im.set_clim(gold_stimulus_torch_crop.min(), gold_stimulus_torch_crop.max())

            loss_mse = criterion_mse(gold_stimulus_torch_crop, prediction_stimuli_torch_crop)
            loss_l1 = criterion_l1(gold_stimulus_torch_crop, prediction_stimuli_torch_crop)

            # Plot L1 loss
            ax = axs[0, 1]
            ax.title.set_text("Loss L1")
            im = ax.imshow(loss_l1)
            plt.colorbar(im, ax=ax)

            # Plot MSE loss
            ax = axs[1, 1]
            ax.title.set_text("Loss MSE")
            im = ax.imshow(loss_mse)
            plt.colorbar(im, ax=ax)

            plt.savefig(out_file)
            plt.close()

    @staticmethod
    def log_outputs(data, model, num_save=16):
        gold_stimuli = data.stimuli_dataset_test[:num_save]
        gold_responses = data.response_dataset_test[:num_save]

        # Compute the predictions on testing dataset
        print("Computing predictions...")
        predictions_stimuli = model.predict(gold_responses)
        print(" - computed the predictions on {} samples".format(data.num_test_data))

        criterion_mse = nn.MSELoss(reduction='none')
        criterion_l1 = nn.L1Loss(reduction='none')

        CROP_SIZE = 64
        transform_crop = transforms.Compose([
            transforms.CenterCrop((CROP_SIZE, CROP_SIZE)),
        ])

        columns = ['id', 'stimulus', 'stimulus_center', 'prediction', 'prediction_center',
                   'loss_l1_2D', 'loss_mse_2D',
                   'loss_l1_center_2D', 'loss_mse_center_2D',
                   'loss_l1_value', 'loss_mse_value',
                   'loss_l1_center_value', 'loss_mse_center_value',
                   ]
        wandb_table = wandb.Table(columns=columns)

        for response_idx, prediction_stimuli in enumerate(predictions_stimuli):
            prediction_stimuli = np.squeeze(prediction_stimuli)

            gold_stimulus_torch = torch.from_numpy(gold_stimuli[response_idx]).squeeze()
            gold_stimulus_torch_crop = transform_crop(gold_stimulus_torch)
            prediction_stimuli_torch = torch.from_numpy(prediction_stimuli).squeeze()
            prediction_stimuli_torch_crop = transform_crop(prediction_stimuli_torch)

            loss_mse = criterion_mse(gold_stimulus_torch, prediction_stimuli_torch)
            loss_mse_crop = criterion_mse(gold_stimulus_torch_crop, prediction_stimuli_torch_crop)
            loss_l1 = criterion_l1(gold_stimulus_torch, prediction_stimuli_torch)
            loss_l1_crop = criterion_l1(gold_stimulus_torch_crop, prediction_stimuli_torch_crop)

            wandb_table.add_data(
                response_idx,
                wandb.Image(gold_stimulus_torch.numpy()),
                wandb.Image(gold_stimulus_torch_crop.numpy()),
                wandb.Image(prediction_stimuli_torch.numpy()),
                wandb.Image(prediction_stimuli_torch_crop.numpy()),
                wandb.Image(loss_l1.numpy()),
                wandb.Image(loss_mse.numpy()),
                wandb.Image(loss_l1_crop.numpy()),
                wandb.Image(loss_mse_crop.numpy()),
                loss_l1.mean(),
                loss_mse.mean(),
                loss_l1_crop.mean(),
                loss_mse_crop.mean(),
            )
        wandb.log({"predictions": wandb_table})

    @staticmethod
    def plot_linear_model_dependencies(predictions_dir, name_prefix, data: LGNData, linear_model, num_save=16):
        np.random.seed(42)

        os.makedirs(predictions_dir, exist_ok=True)
        print("predictions_dir", predictions_dir)

        # choice_indices = np.random.choice(range(data.num_test_data), num_save, replace=False)

        gold_stimuli = data.stimuli_dataset_test
        gold_responses = data.response_dataset_test

        # Get weights (& biases) for the linear model
        w, b = linear_model.get_kernel()
        criterion_mse = nn.MSELoss()

        # pro nekolik pozic
        for _ in range(num_save):
            # Select point in the central 64x64 area of stimuli
            stimuli_pos = np.random.choice(64, 2, replace=False) + 23
            response_pos = np.random.choice(51, 2, replace=False)

            X = gold_responses[:, response_pos[0], response_pos[1]]
            y = gold_stimuli[:, stimuli_pos[0], stimuli_pos[1]]

            weight = w[stimuli_pos[0] * stimuli_pos[1], response_pos[0], response_pos[1]]
            gold_stimuli_torch = torch.from_numpy(y).squeeze()
            prediction_torch = torch.from_numpy(X * weight).squeeze()

            loss_mse = criterion_mse(gold_stimuli_torch, prediction_torch)

            out_file = os.path.join(predictions_dir,
                                    "{}_{}_s{}_r{}".format(name_prefix, num_save, stimuli_pos, response_pos))

            plt.figure(figsize=(10, 10))

            # vykresli scatter pro konkretni (pixel v response)-(pixel v stimuli) + vahu/regresni primku
            plt.title("Stimuli {}, response {}, MSE={}, weight={}"
                      .format(stimuli_pos, response_pos, loss_mse.item(), weight))
            plt.scatter(X, y, s=0.1)
            tan = w[stimuli_pos[0] * stimuli_pos[1], response_pos[0], response_pos[1]]
            plt.plot(X, X * tan)
            plt.ylim(min(y), max(y))
            plt.xlabel("Response px")
            plt.ylabel("Stimuli px")
            plt.savefig(out_file)
            plt.close()


def main():
    # datanorm=""
    datanorm="mean0_std1"
    data = LGNData(datanorm=datanorm)

    for percent_part_100 in range(70, 80, 10):
        # Select part of the training set to train on
        percent_part = percent_part_100 / 100.
        train_samples = int(percent_part * data.num_train_data)
        print("Using {} samples for training out of {}".format(train_samples, data.num_train_data))
        train_stimuli_subset = data.stimuli_dataset_train[:train_samples]
        train_response_subset = data.response_dataset_train[:train_samples]
        percent_subfolder = "{}%".format(int(percent_part * 100))
        if datanorm != "":
            percent_subfolder += "_{}".format(datanorm)

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
        # ModelEvaluator.manual_output(os.path.join(lrm.model_path), "prediction", w, b, data)
        ModelEvaluator.save_outputs(os.path.join(lrm.model_path), "prediction_sklearn", data, lrm, num_save=16)
        ModelEvaluator.plot_linear_model_dependencies(os.path.join(lrm.model_path), "dependency", data, lrm, num_save=16)

        print("-------------------")

        for _ in range(30):
            #
            # Second model - linear network

            # Train second model
            # lnm = LinearNetworkModel(percent_subfolder, device=device, use_bias=False, datanorm=None, use_crop=False, init_value=0)
            lnm = LinearNetworkModel(percent_subfolder, device=device, use_bias=False, datanorm=None, use_crop=True, init_value=0)
            # lnm = LinearNetworkModel(percent_subfolder, device=device, use_bias=False, datanorm=None, use_crop=True, init_value=100)
            # lnm = LinearNetworkModel(percent_subfolder, device=device, use_bias=False, datanorm=None, use_crop=True, init_value=0, dropout=0.5)
            # lnm = LinearNetworkModel(percent_subfolder, device=device, use_bias=False, datanorm=None, use_crop=True, init_value=0, dropout=0.5)
            # lnm = LinearNetworkModel(percent_subfolder, device=device, use_bias=False, datanorm=None, use_crop=True, init_value=0, dropout=0.1, activation='tanh')
            # NOTE: try to initialize with LR kernel - TEST OK -> same results as LR
            # lnm = LinearNetworkModel(percent_subfolder, device=device, use_bias=False, datanorm=None, use_crop=True, init_value=0, init_kernel=w)
            # lnm = LinearNetworkModel(percent_subfolder, device=device, use_bias=False, datanorm=None, use_crop=True, init_value=0, optimizer='adam')
            print("Training the LN model", lnm.model_name)
            lnm.train(train_stimuli_subset, train_response_subset, continue_training=True)

            # Evaluate the model
            print("Evaluating LN (#train {}) model".format(train_samples))
            ModelEvaluator.evaluate(data, lnm)

            # Save the filters
            w, b = lnm.get_kernel()
            lnm_time_dir = os.path.join(lnm.model_path, "{}".format(time.time_ns()))
            ModelEvaluator.save_filters(lnm_time_dir, "deconv_filter", w, b)
            # ModelEvaluator.manual_output(lnm_time_dir, "prediction", w, b, data)
            ModelEvaluator.save_outputs(lnm_time_dir, "prediction_torch", data, lnm, num_save=16)
            ModelEvaluator.plot_linear_model_dependencies(lnm_time_dir, "dependency", data, lnm, num_save=16)

            print("-------------------")
        exit()

        #
        # Third model - convolutional network

        # Train second model
        # cnm = ConvolutionalNetworkModel(percent_subfolder, device=device, use_bias=False, datanorm="mean0_std1", use_crop=True, init_zeros=True)
        # cnm = ConvolutionalNetworkModel(percent_subfolder, device=device, use_bias=False, datanorm="mean0_std1", use_crop=True, init_zeros=False)
        # cnm = ConvolutionalNetworkModel(percent_subfolder, device=device, use_bias=False, datanorm=None, use_crop=True, init_zeros=True)
        # cnm = ConvolutionalNetworkModel(percent_subfolder, device=device, use_bias=False, datanorm=None, use_crop=True, init_zeros=False)
        # cnm = ConvolutionalNetworkModel(percent_subfolder, device=device, use_bias=False, datanorm=None, use_crop=False, init_zeros=True)
        cnm = ConvolutionalNetworkModel(percent_subfolder, device=device, use_bias=False, datanorm="mean0_std1", use_crop=True, init_zeros=True, filters=2)
        print("Training the CN model", cnm.model_name)
        cnm.train(train_stimuli_subset, train_response_subset, continue_training=False)

        # Evaluate the model
        print("Evaluating CN (#train {}) model".format(train_samples))
        # ModelEvaluator.evaluate(data, cnm)

        # Save the filters
        # w, b = cnm.get_kernel()
        cnm_time_dir = os.path.join(cnm.model_path, "{}".format(time.time_ns()))
        # ModelEvaluator.save_filters(cnm_time_dir, "deconv_filter_", w, b)
        ModelEvaluator.save_outputs(cnm_time_dir, "prediction_torch", data, cnm, num_save=16)

        print("-------------------")

        print("\n===================")


if __name__ == '__main__':
    main()
