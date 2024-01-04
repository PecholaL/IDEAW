""" Classifier for coarse-fined locating
"""

import torch.nn as nn
import yaml


class LClassifier(nn.Module):
    def __init__(self, config_path):
        super(LClassifier, self).__init__()
        self.load_config(config_path)
        self.input_size = self.config["LClassifier"]["input_size"]
        self.net = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, data):
        return self.net(data)

    def load_config(self, config_path):
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
