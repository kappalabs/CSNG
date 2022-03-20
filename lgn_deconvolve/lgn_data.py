import os
import pickle


class LGNData:

    def __init__(self, train_part=0.8):
        # Load the dataset from pickle file
        data_dir = os.path.join("..", "datasets")
        neuron_position_file = \
            os.path.join(data_dir, 'lgn_convolved_imagenet_val_greyscale110x110_resize110_cropped.pickle')
        with open(neuron_position_file, 'rb') as f:
            self.dataset = pickle.load(f)

        # Number of all available samples
        self.num_all_data = len(self.dataset['stim'])

        # Calculate training set size
        self.num_train_data = int(self.num_all_data * train_part)

        # Training data
        self.stimuli_dataset_train = self.dataset['stim'][:self.num_train_data]  # 110 x 110
        self.response_dataset_train = self.dataset['resp'][:self.num_train_data]  # 51 x 51

        # Testing data
        self.stimuli_dataset_test = self.dataset['stim'][self.num_train_data:]
        self.response_dataset_test = self.dataset['resp'][self.num_train_data:]

        self.stimuli_shape = self.stimuli_dataset_train[0].shape
        self.response_shape = self.response_dataset_train[0].shape

        self.num_test_data = len(self.stimuli_dataset_test)

    def get_train(self):
        return self.stimuli_dataset_train, self.response_dataset_train

    def get_test(self):
        return self.stimuli_dataset_test, self.response_dataset_test
