""" Modified from DeepHIN
    * define invertible block InnBlock for Mihnet
"""

import torch
import torch.nn as nn
import yaml

from models.dense import DenseBlock


class InnBlock(nn.Module):
    def __init__(self, config_path):
        super(InnBlock, self).__init__()
        self.load_config(config_path)
        self.channel = self.config["InnBlock"]["channel"]
        self.clamp = self.config["InnBlock"]["clamp"]

        # ρ
        self.r = DenseBlock(self.channel, self.channel)
        # η
        self.y = DenseBlock(self.channel, self.channel)
        # φ
        self.f = DenseBlock(self.channel, self.channel)
        # ψ
        self.p = DenseBlock(self.channel, self.channel)

    def forward(self, x1, x2, rev=False):
        if not rev:
            t2 = self.f(x2)
            s2 = self.p(x2)
            y1 = self.e(s2) * x1 + t2
            s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * x2 + t1
        else:
            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.f(y2)
            s2 = self.p(y2)
            y1 = (x1 - t2) / self.e(s2)
        return y1, y2

    def load_config(self, config_path):
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))
