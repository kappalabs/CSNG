import os
import sys
import torch
import wandb
import logging
import argparse
import kornia.augmentation

import numpy as np

from PIL import Image
from definitions import project_dir_path
from lgn_deconvolve.model_evaluator import ModelEvaluator
from trials_decoding.trials_data import TrialsData, TrialsDataset


def get_configuration():
    default_config = {
        "project_name": "trials_decoding",
        # "learning_rate": 2e-4,
        # "num_epochs": 20,
        "batch_size": 8,
        "num_workers": 4,
        "dataset_num_trials": 10,
        "dataset_limit_train": -1,
        "dataset_limit_test": -1,
        "dataset_normalization_stimuli": "zeroone",
        "dataset_normalization_response": "mean0_std1",
        # "model_type": "convolution_network",
        # "model_version": 4,
        # "model_loss": "MSSSIM",
        # "model_name": 'dummy.pth',
        # "clear_progress": True,
        # "evaluate": False,
        # "dropout": 0.5,
        "gpu": 0,
        # "optimizer": "adam",
        # "random_erasing": False,
        # "random_gaussian_noise": False,
        "dataset_limit_responses": -1,
        # "output_intermediate": False,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default=default_config['project_name'])
    # parser.add_argument('--learning_rate', type=float, default=default_config['learning_rate'])
    # parser.add_argument('--num_epochs', type=int, default=default_config['num_epochs'])
    parser.add_argument('--batch_size', type=int, default=default_config['batch_size'])
    parser.add_argument('--num_workers', type=int, default=default_config['num_workers'])
    parser.add_argument('--dataset_num_trials', type=int, default=default_config['dataset_num_trials'])
    parser.add_argument('--dataset_limit_train', type=int, default=default_config['dataset_limit_train'])
    parser.add_argument('--dataset_limit_test', type=int, default=default_config['dataset_limit_test'])
    parser.add_argument('--dataset_normalization_stimuli', type=str,
                        default=default_config['dataset_normalization_stimuli'])
    parser.add_argument('--dataset_normalization_response', type=str,
                        default=default_config['dataset_normalization_response'])
    # parser.add_argument('--model_type', type=str, default=default_config['model_type'])
    # parser.add_argument('--model_version', type=int, default=default_config['model_version'])
    # parser.add_argument('--model_loss', type=str, default=default_config['model_loss'], help="L1/MSE/SSIM/MSSSIM/PSNR")
    # parser.add_argument('--model_name', type=str, default=default_config['model_name'])
    # parser.add_argument('--clear_progress', default=default_config['clear_progress'], action='store_true')
    # parser.add_argument('--evaluate', default=default_config['evaluate'], action='store_true')
    # parser.add_argument('--dropout', type=float, default=default_config['dropout'])
    parser.add_argument('--gpu', type=int, default=default_config['gpu'], help="GPU ID to use (default: 0)")
    # parser.add_argument('--optimizer', type=str, default=default_config['optimizer'], help="adam/sgd")
    # parser.add_argument('--random_erasing', default=default_config['random_erasing'], action='store_true')
    # parser.add_argument('--random_gaussian_noise', default=default_config['random_gaussian_noise'], action='store_true')
    parser.add_argument('--dataset_limit_responses', type=int, default=default_config['dataset_limit_responses'])
    # parser.add_argument('--output_intermediate', default=default_config['output_intermediate'], action='store_true')

    args = parser.parse_args()
    default_config.update(vars(args))

    return default_config


def load_checkpoint(config: dict) -> (torch.utils.data, torch.utils.data, torch.utils.data):
    # Prepare the data
    data = TrialsData(
        datanorm_stimuli=config['dataset_normalization_stimuli'],
        datanorm_response=config['dataset_normalization_response'],
        limit_train=config['dataset_limit_train'],
        limit_test=config['dataset_limit_test'],
        num_trials=config['dataset_num_trials'],
        limit_responses=config['dataset_limit_responses'],
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


def output_evaluation(config: dict, device: torch.device, dataloader_tst: torch.utils.data.DataLoader):
    criteria = ModelEvaluator.get_criteria(device)

    blur = kornia.augmentation.RandomGaussianBlur()
    gauss = kornia.augmentation.RandomGaussianNoise()

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
    dataloader_trn, dataloader_val, dataloader_tst = load_checkpoint(config)

    output_evaluation(config, device, dataloader_tst)


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
