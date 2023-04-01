import os
import abc
import wandb
import torch
import pickle


class ModelBase(abc.ABC):

    @abc.abstractmethod
    def __init__(self, checkpoint_filepath: str, device: torch.device, config: dict,
                 data_stimuli_shape: tuple, data_response_shape: tuple):
        self.checkpoint_filepath = checkpoint_filepath
        self.device = device
        self.config = config

        self.learning_rate = config['learning_rate']
        self.num_epochs = config['num_epochs']
        self.batch_size = config['batch_size']

        self.model = None
        self.wandb_run_id = None
        self.num_epochs_curr = 0
        self.stimuli_shape = data_stimuli_shape
        self.response_shape = data_response_shape

        print("ModelBase.__init__")

    def save_model_data(self, state: dict):
        os.makedirs(os.path.dirname(self.checkpoint_filepath), exist_ok=True)
        with open(self.checkpoint_filepath, 'wb') as f:
            pickle.dump(state, f)

        print("Saved model to {}".format(self.checkpoint_filepath))

        wandb.save(self.checkpoint_filepath)
        print("Saved model to wandb")

    def load_model_data(self) -> dict:
        if not os.path.isfile(self.checkpoint_filepath):
            print("The model {} does not exist".format(self.checkpoint_filepath))
            return None

        with open(self.checkpoint_filepath, 'rb') as f:
            data = pickle.load(f)

        print("Loaded model from {} with num_epochs {}".format(self.checkpoint_filepath, self.num_epochs_curr))

        return data

    @abc.abstractmethod
    def train(self, dataloader_trn: torch.utils.data.DataLoader, dataloader_val: torch.utils.data.DataLoader):
        pass

    @abc.abstractmethod
    def predict(self, dataloader: torch.utils.data.DataLoader):
        if self.model is None:
            raise Exception("Model not trained")
        pass
