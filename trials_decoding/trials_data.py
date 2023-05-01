import os
import ast
import copy
import pickle
import torch.utils.data

import numpy as np
import torchvision.transforms.functional as TF

from definitions import project_dir_path


class TrialsData:

    def __init__(self, train_part=0.7, val_part=0.1, datanorm_stimuli=None, datanorm_response=None, num_trials=10,
                 seed=42, limit_train=-1, limit_val=-1, limit_test=-1, limit_responses=-1,
                 dont_use_l4=False, dont_use_l23=False, dont_use_inhibitory=False, dont_use_excitatory=False,
                 average_trials=False, debug_save_images=False):
        self.train_part = train_part
        self.val_part = val_part
        self.datanorm_stimuli = datanorm_stimuli
        self.datanorm_response = datanorm_response
        self.num_trials = num_trials
        self.seed = seed
        self.limit_train = limit_train
        self.limit_test = limit_test
        self.limit_val = limit_val
        self.limit_responses = limit_responses
        self.dont_use_l4 = dont_use_l4
        self.dont_use_l23 = dont_use_l23
        self.dont_use_inhibitory = dont_use_inhibitory
        self.dont_use_excitatory = dont_use_excitatory
        self.average_trials = average_trials
        self.debug_save_images = debug_save_images

        # Load the dataset from pickle file
        data_dir = os.path.join(project_dir_path, "datasets")
        if self.num_trials == 1:
            neuron_position_file = os.path.join(data_dir, 'one_trial.pickle')
        elif self.num_trials == 10:
            neuron_position_file = os.path.join(data_dir, 'ten_trials.pickle')
        else:
            raise RuntimeError("unavailable number of trials:", self.num_trials)
        with open(neuron_position_file, 'rb') as f:
            self.dataset = pickle.load(f)

        # Number of all available samples
        self.num_all_data = len(self.dataset)

        def get_image_name(sample_info: str):
            sample_info_dict = ast.literal_eval(sample_info)
            image_name = sample_info_dict['image_location']
            image_name = image_name.split('/')[-1]

            return image_name

        if self.debug_save_images:
            from PIL import Image
            for sample_info, sample_dict in self.dataset.items():
                sample_stimulus = sample_dict['stimulus']
                image_name = get_image_name(sample_info)

                img_np = sample_stimulus
                img_np = np.array(img_np, dtype=np.uint8)
                img_pil = Image.fromarray(img_np)
                img_pil.save("tmp_both/stimuli_{}.png".format(image_name))
                img_pil.close()
            exit()

        # Calculate training set size
        self.num_train_data = int(self.num_all_data * self.train_part)
        self.num_val_data = int(self.num_all_data * self.val_part)
        self.num_test_data = self.num_all_data - self.num_train_data - self.num_val_data

        # Update the limits
        if self.limit_train < 0 or self.limit_train > self.num_train_data:
            self.limit_train = self.num_train_data
        if self.limit_val < 0 or self.limit_val > self.num_val_data:
            self.limit_val = self.num_val_data
        if self.limit_test < 0 or self.limit_test > self.num_test_data:
            self.limit_test = self.num_test_data

        # Split the data
        original_keys = list(self.dataset.keys())
        image_names = sorted(list(set(map(get_image_name, original_keys))))
        num_train_keys = int(len(image_names) * self.train_part)
        num_val_keys = int(len(image_names) * self.val_part)
        np.random.seed(self.seed)
        keys_train = set(np.random.choice(image_names, num_train_keys, replace=False))
        remaining_keys = sorted(list(set(image_names) - keys_train))
        keys_val = set(np.random.choice(remaining_keys, num_val_keys, replace=False))
        self.dataset_train = {'response': [], 'stimulus': []}
        self.dataset_val = {'response': [], 'stimulus': []}
        self.dataset_test = {'response': [], 'stimulus': []}
        prechoised_indices = None
        for sample_info, sample_dict in self.dataset.items():
            sample_response = None
            if not self.dont_use_excitatory:
                if not self.dont_use_l4:
                    sample_response = sample_dict['V1_Exc_L4']
                if not self.dont_use_l23:
                    if sample_response is None:
                        sample_response = sample_dict['V1_Exc_L2/3']
                    else:
                        sample_response = np.hstack([sample_response, sample_dict['V1_Exc_L2/3']])
            if not self.dont_use_inhibitory:
                if not self.dont_use_l4:
                    if sample_response is None:
                        sample_response = sample_dict['V1_Inh_L4']
                    else:
                        sample_response = np.hstack([sample_response, sample_dict['V1_Inh_L4']])
                if not self.dont_use_l23:
                    if sample_response is None:
                        sample_response = sample_dict['V1_Inh_L2/3']
                    else:
                        sample_response = np.hstack([sample_response, sample_dict['V1_Inh_L2/3']])
            # sample_response = np.hstack([
            #     sample_dict['V1_Exc_L4'],  # 24000
            #     sample_dict['V1_Exc_L2/3'],  # 24000
            #     sample_dict['V1_Inh_L4'],  # 6000
            #     sample_dict['V1_Inh_L2/3'],  # 6000
            # ])  # 60000
            if self.limit_responses < 0 or self.limit_responses > len(sample_response):
                self.limit_responses = len(sample_response)
            if self.limit_responses < len(sample_response):
                if prechoised_indices is None:
                    np.random.seed(self.seed)
                    prechoised_indices = np.random.choice(len(sample_response), self.limit_responses, replace=False)
                sample_response = sample_response[prechoised_indices]
            sample_stimulus = sample_dict['stimulus']  # 110 x 110
            # Add channels dimension
            sample_stimulus = np.expand_dims(sample_stimulus, axis=0)  # 1 x 110 x 110

            # Get the image name of the sample
            sample_filename = get_image_name(sample_info)

            if sample_filename in keys_train:
                self.dataset_train['response'].append((sample_response, sample_filename))
                self.dataset_train['stimulus'].append((sample_stimulus, sample_filename))
            elif sample_filename in keys_val:
                self.dataset_val['response'].append((sample_response, sample_filename))
                self.dataset_val['stimulus'].append((sample_stimulus, sample_filename))
            else:
                self.dataset_test['response'].append((sample_response, sample_filename))
                self.dataset_test['stimulus'].append((sample_stimulus, sample_filename))

        def average_trials(data):
            # Get the unique filenames
            unique_filenames = np.unique([x[1] for x in data])
            # Average the data with the same sample_filename
            averaged_data = []
            for filename in unique_filenames:
                # Get the data with the same sample_filename
                data_with_same_filename = [x[0] for x in data if x[1] == filename]
                # Average the data
                averaged_data_with_same_filename = np.mean(data_with_same_filename, axis=0)
                # Append the averaged data
                averaged_data.append((averaged_data_with_same_filename, filename))
            return averaged_data

        # Average the data with the same sample_filename
        if self.average_trials:
            self.dataset_train['response'] = average_trials(self.dataset_train['response'])
            self.dataset_train['stimulus'] = average_trials(self.dataset_train['stimulus'])
            # Dont average the validation and test data
            # self.dataset_val['response'] = average_trials(self.dataset_val['response'])
            # self.dataset_val['stimulus'] = average_trials(self.dataset_val['stimulus'])
            # self.dataset_test['response'] = average_trials(self.dataset_test['response'])
            # self.dataset_test['stimulus'] = average_trials(self.dataset_test['stimulus'])

        # Remove the trial dimension
        self.dataset_train['response'] = [x[0] for x in self.dataset_train['response']]
        self.dataset_train['stimulus'] = [x[0] for x in self.dataset_train['stimulus']]
        self.dataset_val['response'] = [x[0] for x in self.dataset_val['response']]
        self.dataset_val['stimulus'] = [x[0] for x in self.dataset_val['stimulus']]
        self.dataset_test['response'] = [x[0] for x in self.dataset_test['response']]
        self.dataset_test['stimulus'] = [x[0] for x in self.dataset_test['stimulus']]

        # Training data
        self.response_dataset_train = np.asarray(self.dataset_train['response'][:self.limit_train], dtype=np.float32)
        # 110px x 110px
        self.stimuli_dataset_train = np.asarray(self.dataset_train['stimulus'][:self.limit_train], dtype=np.float32)

        # Stimuli: compute offsets & standard deviations
        self.stimuli_offsets_train = np.mean(self.stimuli_dataset_train, axis=0)
        self.stimuli_stds_train = np.std(self.stimuli_dataset_train, axis=0)

        # Responses: compute offsets & standard deviations
        self.response_offsets_train = np.mean(self.response_dataset_train, axis=0)
        self.response_stds_train = np.std(self.response_dataset_train, axis=0)

        # Validation data
        self.stimuli_dataset_val = np.asarray(self.dataset_val['stimulus'][:self.limit_val], dtype=np.float32)
        self.response_dataset_val = np.asarray(self.dataset_val['response'][:self.limit_val], dtype=np.float32)

        # Testing data
        self.stimuli_dataset_test = np.asarray(self.dataset_test['stimulus'][:self.limit_test], dtype=np.float32)
        self.response_dataset_test = np.asarray(self.dataset_test['response'][:self.limit_test], dtype=np.float32)

        # Unchanged data
        self.stimuli_dataset_train_raw = copy.deepcopy(self.stimuli_dataset_train)
        self.response_dataset_train_raw = copy.deepcopy(self.response_dataset_train)
        self.stimuli_dataset_val_raw = copy.deepcopy(self.stimuli_dataset_val)
        self.response_dataset_val_raw = copy.deepcopy(self.response_dataset_val)
        self.stimuli_dataset_test_raw = copy.deepcopy(self.stimuli_dataset_test)
        self.response_dataset_test_raw = copy.deepcopy(self.response_dataset_test)

        if self.datanorm_response == 'mean0_std1':
            self.response_dataset_train -= self.response_offsets_train
            self.response_dataset_train /= (self.response_stds_train + 1e-15)

            # Validation data
            self.response_dataset_val -= self.response_offsets_train
            self.response_dataset_val /= (self.response_stds_train + 1e-15)

            # Testing data
            self.response_dataset_test -= self.response_offsets_train
            self.response_dataset_test /= (self.response_stds_train + 1e-15)
        else:
            raise NotImplemented("Data normalization {} not implemented!".format(self.datanorm_response))
        if self.datanorm_stimuli == 'mean0_std1':
            self.stimuli_dataset_train -= self.stimuli_offsets_train
            self.stimuli_dataset_train /= (self.stimuli_stds_train + 1e-15)

            # Validation data
            self.stimuli_dataset_val -= self.stimuli_offsets_train
            self.stimuli_dataset_val /= (self.stimuli_stds_train + 1e-15)

            # Testing data
            self.stimuli_dataset_test -= self.stimuli_offsets_train
            self.stimuli_dataset_test /= (self.stimuli_stds_train + 1e-15)
        elif self.datanorm_stimuli == 'zeroone':
            self.stimuli_dataset_train /= 255.0

            # Validation data
            self.stimuli_dataset_val /= 255.0

            # Testing data
            self.stimuli_dataset_test /= 255.0
        else:
            raise NotImplemented("Data normalization {} not implemented!".format(self.datanorm_stimuli))

        self.stimuli_shape = self.stimuli_dataset_train[0].shape
        self.response_shape = self.response_dataset_train[0].shape

        self.num_train_data = len(self.stimuli_dataset_train)
        self.num_val_data = len(self.stimuli_dataset_val)
        self.num_test_data = len(self.stimuli_dataset_test)

    def get_train(self):
        return self.stimuli_dataset_train, self.response_dataset_train, \
               self.stimuli_dataset_train_raw, self.response_dataset_train_raw

    def get_val(self):
        return self.stimuli_dataset_val, self.response_dataset_val, \
               self.stimuli_dataset_val_raw, self.response_dataset_val_raw

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

    def __init__(self, data: TrialsData, data_type: str = 'train'):
        self.data = data
        self.data_type = data_type

        if self.data_type == 'train':
            self.stimuli, self.responses, self.stimuli_raw, self.responses_raw = data.get_train()
        elif self.data_type == 'validation':
            self.stimuli, self.responses, self.stimuli_raw, self.responses_raw = data.get_val()
        elif self.data_type == 'test':
            self.stimuli, self.responses, self.stimuli_raw, self.responses_raw = data.get_test()
        else:
            raise NotImplemented("Data type {} not implemented!".format(self.data_type))

        assert len(self.stimuli) == len(self.responses), \
            "Number of stimuli ({}) and responses ({}) differs!".format(len(self.stimuli), len(self.responses))

        self.num_samples = len(self.stimuli)

        self._inspect_data()

    def _inspect_data(self):
        print("Inspecting {} data:".format(self.data_type))
        mi, ma, mean, std = \
            self.responses.min(), self.responses.max(), \
            self.responses.mean(), self.responses.std()
        print(" - responses description: shape {}, min {}, max {}, mean {}, std {}"
              .format(self.responses.shape, mi, ma, mean, std))

        mi, ma, mean, std = \
            self.stimuli.min(), self.stimuli.max(), \
            self.stimuli.mean(), self.stimuli.std()
        print(" - stimuli description: shape {}, min {}, max {}, mean {}, std {}"
              .format(self.responses.shape, mi, ma, mean, std))

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
