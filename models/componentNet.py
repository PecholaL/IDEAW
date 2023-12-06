""" Other components of IDEAW
    configuration is located at IDEAW/models/config.yaml
    * Discirminator
    * BalanceBlock
"""

import torch.nn as nn
import yaml

from models.dense import DenseBlock


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
        self.channel = self.config["BalanceBlock"]["channel"]
        self.net = DenseBlock(self.channel, self.channel)

    def forward(self, data):  # i.e. audio_wmd2_stft undergone attLayer
        # orig shape [B,F,T,C]
        print(f"[TEST-BalanceBlock]input shape:{data.shape}")
        data = data.permute(0, 3, 2, 1)  # [B, C, T, F]
        ret = self.net(data).permute(0, 3, 2, 1)
        print(f"[TEST-BalanceBlock]output shape:{ret.shape}")
        return ret

    def load_config(self, config_path):
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
