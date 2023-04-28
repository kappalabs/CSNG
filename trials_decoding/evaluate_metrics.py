import os
import sys
import torch
import logging
import argparse
import kornia.augmentation

import numpy as np

from definitions import project_dir_path
from PIL import Image, ImageDraw, ImageFont
from lgn_deconvolve.model_evaluator import ModelEvaluator
from trials_decoding.trials_data import TrialsData, TrialsDataset


def get_configuration():
    default_config = {
        "project_name": "trials_decoding",
        "batch_size": 8,
        "num_workers": 4,
        "dataset_num_trials": 10,
        "dataset_limit_train": -1,
        "dataset_limit_test": -1,
        "dataset_normalization_stimuli": "zeroone",
        "dataset_normalization_response": "mean0_std1",
        "gpu": 0,
        "dataset_limit_responses": -1,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default=default_config['project_name'])
    parser.add_argument('--batch_size', type=int, default=default_config['batch_size'])
    parser.add_argument('--num_workers', type=int, default=default_config['num_workers'])
    parser.add_argument('--dataset_num_trials', type=int, default=default_config['dataset_num_trials'])
    parser.add_argument('--dataset_limit_train', type=int, default=default_config['dataset_limit_train'])
    parser.add_argument('--dataset_limit_test', type=int, default=default_config['dataset_limit_test'])
    parser.add_argument('--dataset_normalization_stimuli', type=str,
                        default=default_config['dataset_normalization_stimuli'])
    parser.add_argument('--dataset_normalization_response', type=str,
                        default=default_config['dataset_normalization_response'])
    parser.add_argument('--gpu', type=int, default=default_config['gpu'], help="GPU ID to use (default: 0)")
    parser.add_argument('--dataset_limit_responses', type=int, default=default_config['dataset_limit_responses'])

    args = parser.parse_args()
    default_config.update(vars(args))

    return default_config


def prepare_dataloader(config: dict) -> torch.utils.data:
    # Prepare the data
    data = TrialsData(
        datanorm_stimuli=config['dataset_normalization_stimuli'],
        datanorm_response=config['dataset_normalization_response'],
        limit_train=config['dataset_limit_train'],
        limit_test=config['dataset_limit_test'],
        num_trials=config['dataset_num_trials'],
        limit_responses=config['dataset_limit_responses'],
    )
    dataset_tst = TrialsDataset(data, data_type='test')

    dataloader_tst = torch.utils.data.DataLoader(
        dataset=dataset_tst,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
    )
    print("Prepared TST dataloader with", len(dataloader_tst.dataset), "samples")

    return dataloader_tst


def get_augmented_stimulus_loss(stimulus_torch: torch.FloatTensor, stimulus_augmented: torch.FloatTensor, criterion,
                                metrics_evaluation_path: str, sample_idx: int, criterion_name: str,
                                augmentation_name: str):
    # make a blurred version of the stimulus, compute loss and save
    loss_blurred = torch.mean(criterion(stimulus_torch, stimulus_augmented))
    stimulus_augmented = stimulus_augmented.cpu().numpy()
    stimulus_augmented = np.squeeze(stimulus_augmented)
    stimulus_augmented = np.uint8(stimulus_augmented * 255)
    # Add black space above the image to fit the text
    stimulus_augmented = np.concatenate((np.zeros((20, stimulus_augmented.shape[1]), dtype=stimulus_augmented.dtype),
                                         stimulus_augmented), axis=0)
    stimulus_augmented = Image.fromarray(stimulus_augmented)
    # Draw the loss above the image
    draw = ImageDraw.Draw(stimulus_augmented)
    loss_value = round(loss_blurred.item(), 2)
    # use bigger font size
    font = ImageFont.truetype('ORIOND.TTF', 20)
    draw.text((0, 0), str(loss_value), 255, font=font)
    stimulus_augmented.save(
        os.path.join(metrics_evaluation_path,
                     "stimulus_{}_{}_{}.png".format(sample_idx, augmentation_name, criterion_name)))


def output_evaluation(device: torch.device, dataloader_tst: torch.utils.data.DataLoader):
    criteria = ModelEvaluator.get_criteria(device)

    torch.manual_seed(42)

    transform_crop = ModelEvaluator.get_central_crop_transform()

    augmentations = []
    augmentations.append((lambda x: x, "original"))

    # augmentations.append((kornia.augmentation.RandomGaussianBlur(kernel_size=65, sigma=(1, 1), p=1.0), "blur_65_1"))
    # augmentations.append((kornia.augmentation.RandomGaussianBlur(kernel_size=65, sigma=(5, 5), p=1.0), "blur_65_5"))
    augmentations.append((kornia.augmentation.RandomGaussianBlur(kernel_size=65, sigma=(10, 10), p=1.0), "blur_65_10"))

    # augmentations.append((kornia.augmentation.RandomGaussianNoise(std=0.01, p=1.0), "noise_0.01"))
    # augmentations.append((kornia.augmentation.RandomGaussianNoise(std=0.05, p=1.0), "noise_0.05"))
    augmentations.append((kornia.augmentation.RandomGaussianNoise(std=0.1, p=1.0), "noise_0.1"))

    # add value noise
    # create a torch tensor of size same as x, with values -1 or 1
    augmentations.append((lambda x: x + 0.07 * (torch.randint(0, 2, x.shape) * 2 - 1), "value_noise_0.07"))
    # augmentations.append((lambda x: x + 0.1 * (torch.randint(0, 2, x.shape) * 2 - 1), "value_noise_0.1"))
    # augmentations.append((lambda x: x + 0.5 * (torch.randint(0, 2, x.shape) * 2 - 1), "value_noise_0.5"))

    # add values shift
    augmentations.append((lambda x: x + 0.07, "value_shift_0.07"))
    # augmentations.append((lambda x: x + 0.1, "value_shift_0.1"))
    # augmentations.append((lambda x: x + 0.5, "value_shift_0.5"))

    # add image shift in x axis
    # augmentations.append((lambda x: torch.roll(x, 1, 2), "image_shift_x_1"))
    # augmentations.append((lambda x: torch.roll(x, 5, 2), "image_shift_x_5"))
    augmentations.append((lambda x: torch.roll(x, 9, 2), "image_shift_x_9"))

    sample_idx = 0
    num_samples_to_save = 4
    for batch_idx, batch in enumerate(dataloader_tst):
        # Load the data
        stimuli, response = batch['stimulus'], batch['response']

        # Create the metrics evaluation directory
        metrics_evaluation_path = os.path.join(project_dir_path, "metrics_evaluation")
        if not os.path.exists(metrics_evaluation_path):
            os.makedirs(metrics_evaluation_path)

        # Make the transformations and save
        for sample_in_batch_idx in range(stimuli.shape[0]):
            stimulus_torch = stimuli[sample_in_batch_idx]
            stimulus_torch = stimulus_torch.unsqueeze(0).detach()
            stimulus_torch = transform_crop(stimulus_torch)

            for augmentation, augmentation_name in augmentations:
                stimulus_augmented = augmentation(stimulus_torch)
                # clip to [0, 1]
                stimulus_augmented = torch.clamp(stimulus_augmented, 0, 1)

                for criterion_name, criterion in criteria:
                    # Make a blurred version of the stimulus, compute loss and save
                    get_augmented_stimulus_loss(stimulus_torch, stimulus_augmented, criterion,
                                                metrics_evaluation_path, sample_idx, criterion_name, augmentation_name)

            sample_idx += 1

            if sample_idx >= num_samples_to_save:
                break
        if sample_idx >= num_samples_to_save:
            break


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

    # Evaluate the metrics
    dataloader_tst = prepare_dataloader(config)
    output_evaluation(device, dataloader_tst)


if __name__ == "__main__":
    main()
