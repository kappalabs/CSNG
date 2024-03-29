import os
import sys
import torch
import wandb
import logging
import argparse

import numpy as np

from PIL import Image
from definitions import project_dir_path
from ml_models.model_base import ModelBase
from ml_models.linear_network import LinearNetworkModel
from lgn_deconvolve.model_evaluator import ModelEvaluator
from ml_models.linear_regression import LinearRegressionModel
from trials_decoding.trials_data import TrialsData, TrialsDataset
from ml_models.convolution_network import ConvolutionalNetworkModel


def get_configuration():
    default_config = {
        "project_name": "trials_decoding",
        "learning_rate": 2e-4,
        "num_epochs": 15,
        "batch_size": 64,
        "num_workers": 4,
        "dataset_num_trials": 10,
        "dataset_limit_train": -1,
        "dataset_limit_test": -1,
        "dataset_normalization_stimuli": "zeroone",
        "dataset_normalization_response": "mean0_std1",
        "model_type": "linear_regression",
        "model_version": "1",
        "model_loss": "L1",
        "model_name": 'dummy.pth',
        "clear_progress": True,
        "evaluate": False,
        "dropout": 0.5,
        "gpu": 0,
        "optimizer": "adam",
        "random_erasing": False,
        "random_gaussian_noise": False,
        "dataset_limit_responses": -1,
        "output_intermediate": False,
        "dataset_dont_use_l4": False,
        "dataset_dont_use_l23": False,
        "dataset_dont_use_inhibitory": False,
        "dataset_dont_use_excitatory": False,
        "dataset_average_trials": False,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default=default_config['project_name'])
    parser.add_argument('--learning_rate', type=float, default=default_config['learning_rate'])
    parser.add_argument('--num_epochs', type=int, default=default_config['num_epochs'])
    parser.add_argument('--batch_size', type=int, default=default_config['batch_size'])
    parser.add_argument('--num_workers', type=int, default=default_config['num_workers'])
    parser.add_argument('--dataset_num_trials', type=int, default=default_config['dataset_num_trials'])
    parser.add_argument('--dataset_limit_train', type=int, default=default_config['dataset_limit_train'])
    parser.add_argument('--dataset_limit_test', type=int, default=default_config['dataset_limit_test'])
    parser.add_argument('--dataset_normalization_stimuli', type=str,
                        default=default_config['dataset_normalization_stimuli'])
    parser.add_argument('--dataset_normalization_response', type=str,
                        default=default_config['dataset_normalization_response'])
    parser.add_argument('--model_type', type=str, default=default_config['model_type'])
    parser.add_argument('--model_version', type=int, default=default_config['model_version'])
    parser.add_argument('--model_loss', type=str, default=default_config['model_loss'], help="L1/MSE/SSIM/MSSSIM/PSNR")
    parser.add_argument('--model_name', type=str, default=default_config['model_name'])
    parser.add_argument('--clear_progress', default=default_config['clear_progress'], action='store_true')
    parser.add_argument('--evaluate', default=default_config['evaluate'], action='store_true')
    parser.add_argument('--dropout', type=float, default=default_config['dropout'])
    parser.add_argument('--gpu', type=int, default=default_config['gpu'], help="GPU ID to use (default: 0)")
    parser.add_argument('--optimizer', type=str, default=default_config['optimizer'], help="adam/sgd")
    parser.add_argument('--random_erasing', default=default_config['random_erasing'], action='store_true')
    parser.add_argument('--random_gaussian_noise', default=default_config['random_gaussian_noise'], action='store_true')
    parser.add_argument('--dataset_limit_responses', type=int, default=default_config['dataset_limit_responses'])
    parser.add_argument('--output_intermediate', default=default_config['output_intermediate'], action='store_true')
    parser.add_argument('--dataset_dont_use_l4', default=default_config['dataset_dont_use_l4'], action='store_true')
    parser.add_argument('--dataset_dont_use_l23', default=default_config['dataset_dont_use_l23'], action='store_true')
    parser.add_argument('--dataset_dont_use_inhibitory', default=default_config['dataset_dont_use_inhibitory'],
                        action='store_true')
    parser.add_argument('--dataset_dont_use_excitatory', default=default_config['dataset_dont_use_excitatory'],
                        action='store_true')
    parser.add_argument('--dataset_average_trials', default=default_config['dataset_average_trials'],
                        action='store_true')

    args = parser.parse_args()
    default_config.update(vars(args))

    return default_config


def load_checkpoint(config: dict, checkpoint_filepath: str, device: torch.device) \
        -> (ModelBase, torch.utils.data, torch.utils.data, int):
    # Prepare the data
    data = TrialsData(
        datanorm_stimuli=config['dataset_normalization_stimuli'],
        datanorm_response=config['dataset_normalization_response'],
        limit_train=config['dataset_limit_train'],
        limit_test=config['dataset_limit_test'],
        num_trials=config['dataset_num_trials'],
        limit_responses=config['dataset_limit_responses'],
        dont_use_l4=config['dataset_dont_use_l4'],
        dont_use_l23=config['dataset_dont_use_l23'],
        dont_use_inhibitory=config['dataset_dont_use_inhibitory'],
        dont_use_excitatory=config['dataset_dont_use_excitatory'],
        average_trials=config['dataset_average_trials'],
    )
    dataset_trn = TrialsDataset(data, data_type='train')
    dataset_val = TrialsDataset(data, data_type='validation')
    dataset_tst = TrialsDataset(data, data_type='test')

    # Prepare dataloader
    dataloader_trn = torch.utils.data.DataLoader(
        dataset=dataset_trn,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
    )
    print("Prepared TRN dataloader with", len(dataloader_trn.dataset), "samples")
    dataloader_val = torch.utils.data.DataLoader(
        dataset=dataset_val,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
    )
    print("Prepared VAL dataloader with", len(dataloader_val.dataset), "samples")
    dataloader_tst = torch.utils.data.DataLoader(
        dataset=dataset_tst,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
    )
    print("Prepared TST dataloader with", len(dataloader_tst.dataset), "samples")

    # Initialize & load the model
    if config['model_type'] == 'linear_regression':
        model = LinearRegressionModel(checkpoint_filepath, device, config,
                                      data.get_stimuli_shape, data.get_response_shape)
    elif config['model_type'] == 'linear_network':
        model = LinearNetworkModel(checkpoint_filepath, device, config,
                                   data.get_stimuli_shape, data.get_response_shape)
    elif config['model_type'] == 'convolution_network':
        model = ConvolutionalNetworkModel(checkpoint_filepath, device, config,
                                          data.get_stimuli_shape, data.get_response_shape)
    else:
        raise NotImplementedError("Model type {} is not supported!".format(config['model_type']))

    # Get the info from loaded model
    wandb_run_id = None
    if not config['clear_progress'] and not config['evaluate']:
        wandb_run_id = model.wandb_run_id

    return model, dataloader_trn, dataloader_val, dataloader_tst, wandb_run_id


def output_intermediate(config: dict, model: ModelBase, dataloader_tst: torch.utils.data.DataLoader):
    sample_idx = 0
    num_samples_to_save = 64
    for batch_idx, batch in enumerate(dataloader_tst):
        # Load the data
        stimuli, response = batch['stimulus'], batch['response']
        stimuli = stimuli.to(model.device, dtype=torch.float)
        response = response.to(model.device, dtype=torch.float)

        # Compute the predictions
        predictions, intermediates = model.predict_batch_with_intermediate(response)

        # save the intermediate activations
        if config['output_intermediate']:
            intermediate_path = os.path.join(project_dir_path, "intermediate", config['model_type'], wandb.run.name)
            if not os.path.exists(intermediate_path):
                os.makedirs(intermediate_path)
            for sample_in_batch_idx in range(stimuli.shape[0]):
                intermediate = intermediates[sample_in_batch_idx].cpu().numpy()
                # Save the numpy image
                intermediate = np.squeeze(intermediate)
                intermediate = intermediate - np.min(intermediate)
                intermediate = intermediate / np.max(intermediate)
                intermediate = np.uint8(intermediate * 255)
                intermediate = Image.fromarray(intermediate)
                intermediate.save(os.path.join(intermediate_path, "intermediate_{}.png".format(sample_idx)))
                stimulus = stimuli[sample_in_batch_idx].cpu().numpy()
                stimulus = np.squeeze(stimulus)
                stimulus = stimulus - np.min(stimulus)
                stimulus = stimulus / np.max(stimulus)
                stimulus = np.uint8(stimulus * 255)
                stimulus = Image.fromarray(stimulus)
                stimulus.save(os.path.join(intermediate_path, "stimulus_{}.png".format(sample_idx)))
                prediction = predictions[sample_in_batch_idx].cpu().numpy()
                prediction = np.squeeze(prediction)
                prediction = prediction - np.min(prediction)
                prediction = prediction / np.max(prediction)
                prediction = np.uint8(prediction * 255)
                prediction = Image.fromarray(prediction)
                prediction.save(os.path.join(intermediate_path, "prediction_{}.png".format(sample_idx)))
                sample_idx += 1

                if sample_idx >= num_samples_to_save:
                    break
        if sample_idx >= num_samples_to_save:
            break


def train(config: dict, device: torch.device):
    checkpoint_filepath = os.path.join(project_dir_path, "checkpoints", config['model_type'], config['model_name'])
    model, dataloader_trn, dataloader_val, dataloader_tst, wandb_run_id = \
        load_checkpoint(config, checkpoint_filepath, device)  # type: ModelBase

    # Initialize W&b
    wandb.init(project=config['project_name'], id=wandb_run_id, resume='allow')
    wandb.config.update(config)
    checkpoint_filepath = os.path.join(project_dir_path, "checkpoints", config['model_type'], wandb.run.name + '.pth')
    model.checkpoint_filepath = checkpoint_filepath

    if not config['evaluate']:
        # Train the model
        model.train(dataloader_trn, dataloader_val)

    if config['output_intermediate']:
        output_intermediate(config, model, dataloader_tst)

    # Evaluate the model
    print("Evaluating LR (#train {}) model {}".format(config['dataset_limit_train'], config['model_type']))
    evaluate_dict = ModelEvaluator.evaluate(dataloader_tst, model, log_dict_prefix='test.')
    wandb.log(evaluate_dict, commit=False)

    # Save the model predictions
    outputs_dict = ModelEvaluator.log_outputs(dataloader_tst, model, num_save=16, log_dict_prefix='test.')
    wandb.log(outputs_dict, commit=True)


def main():
    # Init logger
    logger = logging.getLogger()

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    # Init configurations
    config = get_configuration()

    # Get running device
    if torch.cuda.is_available() and config['gpu'] >= 0:
        device = torch.device(config['gpu'])
    else:
        device = torch.device('cpu')
    print("device:", device)
    logger.info('Device is {}'.format(device))

    # Train the model
    train(config, device)


if __name__ == "__main__":
    main()
