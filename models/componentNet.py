""" Other components of IDEAW
    configuration is located at IDEAW/models/config.yaml
    * Discirminator
    * BalanceBlock
"""

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

    def forward(self, data):  # audio data in time domain
        return self.net(data)

    def load_config(self, config_path):
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)


class BalanceBlock(nn.Module):
    def __init__(self, config_path):
        super(BalanceBlock, self).__init__()
        self.load_config(config_path)

    def forward(self, data):
        pass

    def load_config(self, config_path):
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
