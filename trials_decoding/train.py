import os
import re

import torch
import wandb
import logging
import argparse

from definitions import project_dir_path
from lgn_deconvolve.model_evaluator import ModelEvaluator
from trials_decoding.trials_data import TrialsData
from ml_models.linear_network import LinearNetworkModel
from ml_models.linear_regression import LinearRegressionModel
from ml_models.convolution_network import ConvolutionalNetworkModel


def get_configuration():
    default_config = {
        "project_name": "trials_decoding",
        "learning_rate": 1e-2,
        "num_epochs": 100,
        "batch_size": 16,
        "dataset_num_trials": 10,
        "dataset_limit_train": 100,
        "dataset_limit_test": -1,
        "dataset_normalization": "mean0_std1",
        "model_type": "linear_regression",
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default=default_config['project_name'])
    parser.add_argument('--learning_rate', type=float, default=default_config['learning_rate'])
    parser.add_argument('--num_epochs', type=int, default=default_config['num_epochs'])
    parser.add_argument('--batch_size', type=int, default=default_config['batch_size'])
    parser.add_argument('--dataset_num_trials', type=int, default=default_config['dataset_num_trials'])
    parser.add_argument('--dataset_limit_train', type=int, default=default_config['dataset_limit_train'])
    parser.add_argument('--dataset_limit_test', type=int, default=default_config['dataset_limit_test'])
    parser.add_argument('--dataset_normalization', type=str, default=default_config['dataset_normalization'])
    parser.add_argument('--model_type', type=str, default=default_config['model_type'])

    args = parser.parse_args()
    default_config.update(vars(args))

    return default_config


def load_checkpoint(config, checkpoint_filepath: str, device: torch.device):
    # Initialize & load the model
    if config['model_type'] == 'linear_regression':
        model = LinearRegressionModel(checkpoint_filepath)
    elif config['model_type'] == 'linear_network':
        model = LinearNetworkModel(checkpoint_filepath, device)
    elif config['model_type'] == 'convolution_network':
        model = ConvolutionalNetworkModel(checkpoint_filepath, device)
    else:
        raise NotImplemented("Model type {} is not supported!".format(config['model_type']))

    # Prepare the data
    data = TrialsData(
        datanorm=config['dataset_normalization'],
        limit_train=config['dataset_limit_train'],
        limit_test=config['dataset_limit_test'],
    )

    # Get the info from loaded model
    wandb_run_id = model.wandb_run_id

    return model, data, wandb_run_id


def get_model_name(config: dict):
    config = {key: val for key, val in sorted(config.items(), key=lambda ele: ele[0])}
    model_name = str(config)
    model_name = re.sub(r' ', '_', model_name)
    model_name = re.sub(r'\'', '', model_name)
    model_name = re.sub(r':_', ':', model_name)
    model_name = re.sub(r',', '__', model_name)
    model_name = re.sub(r'\{', '', model_name)
    model_name = re.sub(r'\}', '', model_name)
    model_name += '.pth'

    return model_name


def train(config: dict, device: torch.device):
    checkpoint_filename = get_model_name(config)
    checkpoint_filepath = os.path.join(project_dir_path, "checkpoints", checkpoint_filename)
    model, data, wandb_run_id = load_checkpoint(config, checkpoint_filepath, device)

    # Initialize W&b
    wandb.init(project=config['project_name'], id=wandb_run_id)
    wandb.config.update(config)

    # Prepare data
    train_stimuli_subset = data.stimuli_dataset_train
    train_response_subset = data.response_dataset_train

    # Train the model
    model.train(train_stimuli_subset, train_response_subset)

    # Evaluate the model
    print("Evaluating LR (#train {}) model {}".format(config['dataset_limit_train'], config['model_type']))
    ModelEvaluator.evaluate(data, model)


def main():
    # Init logger
    logger = logging.getLogger()

    # Init configurations
    config = get_configuration()

    # Get running device
    device = torch.device(0)
    logger.info('Device is', device)

    # Train the model
    train(config, device)


if __name__ == "__main__":
    main()
