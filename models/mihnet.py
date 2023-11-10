""" Modified from DeepMIH
    configuration is at IDEAW/models/config.yaml
"""

import torch
import torch.nn as nn
import yaml

from inn import InnBlock


class Mihnet_s1(nn.Module):
    def __init__(self, config_path):
        super(Mihnet_s1, self).__init__()
        self.load_config(config_path)
        self.inv1 = InnBlock()
        self.inv2 = InnBlock()
        self.inv3 = InnBlock()
        self.inv4 = InnBlock()

        self.inv5 = InnBlock()
        self.inv6 = InnBlock()
        self.inv7 = InnBlock()
        self.inv8 = InnBlock()


    def load_config(self, config_path):
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)


    def forward(self, x, rev=False):
        if not rev:
            out = self.inv1(x)
            out = self.inv2(out)
            out = self.inv3(out)
            out = self.inv4(out)
        else:
            out = self.inv5(x, rev=True)
            out = self.inv6(out, rev=True)
            out = self.inv7(out, rev=True)
            out = self.inv8(out, rev=True)

        return out


class Mihnet_s2(nn.Module):
    def __init__(self, config_path):
        super(Mihnet_s2, self).__init__()
        self.load_config(config_path)
        self.inv1 = InnBlock()
        self.inv2 = InnBlock()
        self.inv3 = InnBlock()
        self.inv4 = InnBlock()

        self.inv5 = InnBlock()
        self.inv6 = InnBlock()
        self.inv7 = InnBlock()
        self.inv8 = InnBlock()


    def load_config(self, config_path):
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)


    def forward(self, x, rev=False):
        if not rev:
            out = self.inv1(x)
            out = self.inv2(out)
            out = self.inv3(out)
            out = self.inv4(out)
        else:
            out = self.inv5(x, rev=True)
            out = self.inv6(out, rev=True)
            out = self.inv7(out, rev=True)
            out = self.inv8(out, rev=True)

        return out

