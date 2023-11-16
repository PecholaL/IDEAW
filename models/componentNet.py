""" Other components of IDEAW
    configuration is located at IDEAW/models/config.yaml
    * Discirminator
    * Restorer
"""

import torch
import torch.nn as nn
import yaml


class Discriminator(nn.Module):
    def __init__(self, config_path):
        super(Discriminator, self).__init__()
        self.load_config(config_path)
        self.input_size = self.config["Discriminator"]["input_size"]
        self.net = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def __forward__(self, data):  # audio data in time domain
        return self.net(data)

    def load_config(self, config_path):
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)


class Restorer(nn.Module):
    def __init__(self, config_path):
        super(Restorer, self).__init__()
        self.load_config(config_path)

    def load_config(self, config_path):
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
