""" Attack Simulate Layer
    configuration is located at IDEAW/models/config
    * Gaussian Noise
    * Reverberation
    * MP3 compress
    * ...
"""

import torch
import torch.nn as nn
import yaml

class AttackLayer(nn.Module):
    def __init__(self, config_path):
        super(AttackLayer, self).__init__()
        self.load_config(config_path)


    def load_config(self, config_path):
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
