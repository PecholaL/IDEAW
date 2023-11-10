""" Modified from DeepHIN
    * define invertible block InnBlock for Mihnet
"""

import torch
import torch.nn as nn
import yaml

from componentNet import DenseBlock


class InnBlock(nn.Module):
    def __init__(self, config_path, imp_map=True):
        super(InnBlock, self).__init__()
        self.load_config(config_path)

        self.channel = self.config['InnBlock']['channel']
        self.clamp = self.config['InnBlock']['clamp']
        if imp_map:
            self.imp = self.config['InnBlock']['imp']
        else:
            self.imp = 0


        # ρ
        self.r = DenseBlock(self.channel + self.imp, self.channel)
        # η
        self.y = DenseBlock(self.channel + self.imp, self.channel)
        # φ
        self.f = DenseBlock(self.channel, self.channel + self.imp)
        # ψ
        self.p = DenseBlock(self.channel, self.channel + self.imp)


    def load_config(self, config_path):
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)


    def forward(self, x1, x2, rev=False):
        # x1, x2 = (x.narrow(1, 0, self.split_len1 + self.imp),
        #           x.narrow(1, self.split_len1 + self.imp, self.split_len2))

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
