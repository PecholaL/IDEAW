""" Modified from DeepMIH
    configuration is at IDEAW/models/config.yaml
"""

import torch
import torch.nn as nn
import yaml

class Mihnet(nn.Module):
    def __init__(self, config_path):
        super(Mihnet, self).__init__()
        self.load_config(config_path)


    def load_config(self, config_path):
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)