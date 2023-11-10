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


    def load_config(self, config_path):
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)


class Restorer(nn.Module):
    def __init__(self, config_path):
        super(Restorer, self).__init__()
        self.load_config(config_path)


    def load_config(self, config_path):
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
