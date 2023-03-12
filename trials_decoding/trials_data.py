import os
import copy
import pickle
import torch.utils.data

import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from definitions import project_dir_path


class TrialsData:

    def __init__(self, train_part=0.8, datanorm_stimuli=None, datanorm_response=None, num_trials=10, seed=42,
                 limit_train=-1, limit_test=-1):
        self.train_part = train_part
        self.datanorm_stimuli = datanorm_stimuli
        self.datanorm_response = datanorm_response
        self.num_trials = num_trials
        self.seed = seed
        self.limit_train = limit_train
        self.limit_test = limit_test

        # Load the dataset from pickle file
        data_dir = os.path.join(project_dir_path, "datasets")
        if num_trials == 1:
            neuron_position_file = os.path.join(data_dir, 'one_trial.pickle')
        elif num_trials == 10:
            neuron_position_file = os.path.join(data_dir, 'ten_trials.pickle')
        else:
            raise RuntimeError("unavailable number of trials:", num_trials)
        with open(neuron_position_file, 'rb') as f:
            self.dataset = pickle.load(f)

        # Number of all available samples
        self.num_all_data = len(self.dataset)

        # Calculate training set size
        self.num_train_data = int(self.num_all_data * train_part)
        self.num_test_data = self.num_all_data - self.num_train_data

        # Update the limits
        if self.limit_train < 0 or self.limit_train > self.num_train_data:
            self.limit_train = self.num_train_data
        if self.limit_test < 0 or self.limit_test > self.num_test_data:
            self.limit_test = self.num_test_data

        # Split the data
        keys = sorted(list(self.dataset.keys()))
        np.random.seed(seed)
        keys_train = set(np.random.choice(keys, self.num_train_data, replace=False))
        self.dataset_train = {'response': [], 'stimulus': []}
        self.dataset_test = {'response': [], 'stimulus': []}
        for sample_info, sample_dict in self.dataset.items():
            sample_response = np.hstack([
                sample_dict['V1_Exc_L4'],
                sample_dict['V1_Exc_L2/3'],
                sample_dict['V1_Inh_L4'],
                sample_dict['V1_Inh_L2/3'],
            ])  # 4 x 24000
            sample_stimulus = sample_dict['stimulus']  # 110 x 110

            if sample_info in keys_train:
                self.dataset_train['response'].append(sample_response)
                self.dataset_train['stimulus'].append(sample_stimulus)
            else:
                self.dataset_test['response'].append(sample_response)
                self.dataset_test['stimulus'].append(sample_stimulus)

        # Training data
        self.response_dataset_train = np.asarray(self.dataset_train['response'][:limit_train], dtype=np.float32)
        # 110px x 110px
        self.stimuli_dataset_train = np.asarray(self.dataset_train['stimulus'][:limit_train], dtype=np.float32)

        # Stimuli: compute offsets & standard deviations
        self.stimuli_offsets_train = np.mean(self.stimuli_dataset_train, axis=0)
        self.stimuli_stds_train = np.std(self.stimuli_dataset_train, axis=0)

        # Responses: compute offsets & standard deviations
        self.response_offsets_train = np.mean(self.response_dataset_train, axis=0)
        self.response_stds_train = np.std(self.response_dataset_train, axis=0)

        # Testing data
        self.stimuli_dataset_test = np.asarray(self.dataset_test['stimulus'][:limit_test], dtype=np.float32)
        self.response_dataset_test = np.asarray(self.dataset_test['response'][:limit_test], dtype=np.float32)

        # Unchanged data
        self.stimuli_dataset_train_raw = copy.deepcopy(self.stimuli_dataset_train)
        self.response_dataset_train_raw = copy.deepcopy(self.response_dataset_train)
        self.stimuli_dataset_test_raw = copy.deepcopy(self.stimuli_dataset_test)
        self.response_dataset_test_raw = copy.deepcopy(self.response_dataset_test)

        if self.datanorm_response == 'mean0_std1':
            self.response_dataset_train -= self.response_offsets_train
            self.response_dataset_train /= (self.response_stds_train + 1e-15)

            # Testing data
            self.response_dataset_test -= self.response_offsets_train
            self.response_dataset_test /= (self.response_stds_train + 1e-15)
        else:
            raise NotImplemented("Data normalization {} not implemented!".format(self.datanorm_response))
        if self.datanorm_stimuli == 'mean0_std1':
            self.stimuli_dataset_train -= self.stimuli_offsets_train
            self.stimuli_dataset_train /= (self.stimuli_stds_train + 1e-15)

            # Testing data
            self.stimuli_dataset_test -= self.stimuli_offsets_train
            self.stimuli_dataset_test /= (self.stimuli_stds_train + 1e-15)
        elif self.datanorm_stimuli == 'zeroone':
            self.stimuli_dataset_train /= 255.0

            # Testing data
            self.stimuli_dataset_test /= 255.0
        else:
            raise NotImplemented("Data normalization {} not implemented!".format(self.datanorm_stimuli))

        self.stimuli_shape = self.stimuli_dataset_train[0].shape
        self.response_shape = self.response_dataset_train[0].shape

        self.num_train_data = len(self.stimuli_dataset_train)
        self.num_test_data = len(self.stimuli_dataset_test)

    def get_train(self):
        return self.stimuli_dataset_train, self.response_dataset_train, \
               self.stimuli_dataset_train_raw, self.response_dataset_train_raw

    def get_test(self):
        return self.stimuli_dataset_test, self.response_dataset_test, \
               self.stimuli_dataset_test_raw, self.response_dataset_test_raw

    @property
    def get_stimuli_shape(self):
        return self.stimuli_shape

    @property
    def get_response_shape(self):
        return self.response_shape


class TrialsDataset(torch.utils.data.Dataset):

    def __init__(self, data: TrialsData, train: bool = True):
        self.data = data
        self.train = train

        if self.train:
            self.stimuli, self.responses, self.stimuli_raw, self.responses_raw = data.get_train()
        else:
            self.stimuli, self.responses, self.stimuli_raw, self.responses_raw = data.get_test()

        assert len(self.stimuli) == len(self.responses), \
            "Number of stimuli ({}) and responses ({}) differs!".format(len(self.stimuli), len(self.responses))

        self.num_samples = len(self.stimuli)

        self._inspect_data()

    def _inspect_data(self):
        mi, ma, mean, std = \
            self.responses.min(), self.responses.max(), \
            self.responses.mean(), self.responses.std()
        print(" - responses raw description: min {}, max {}, mean {}, std {}".format(mi, ma, mean, std))

        mi, ma, mean, std = \
            self.stimuli.min(), self.stimuli.max(), \
            self.stimuli.mean(), self.stimuli.std()
        print(" - stimuli raw description: min {}, max {}, mean {}, std {}".format(mi, ma, mean, std))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        response = self.responses[idx]
        response_torch = torch.from_numpy(response).float()

        stimulus = self.stimuli[idx]
        stimulus_torch = torch.from_numpy(stimulus).float()

        sample = {
            "stimulus": stimulus_torch,
            "response": response_torch,
        }

        return sample


class CentralPxCropTransform:
    def __call__(self, x):
        x = TF.crop(x, 55, 55, 10, 10)
        return x


if __name__ == "__main__":
    TrialsData()
