""" Modified from DeepMIH
    configuration is at IDEAW/models/config.yaml
"""

import torch.nn as nn
import yaml

from inn import InnBlock


class Mihnet_s1(nn.Module):
    def __init__(self, config_path):
        super(Mihnet_s1, self).__init__()
        self.inv1 = InnBlock(config_path)
        self.inv2 = InnBlock(config_path)
        self.inv3 = InnBlock(config_path)
        self.inv4 = InnBlock(config_path)

        self.inv5 = InnBlock(config_path)
        self.inv6 = InnBlock(config_path)
        self.inv7 = InnBlock(config_path)
        self.inv8 = InnBlock(config_path)

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
        self.inv1 = InnBlock(config_path)
        self.inv2 = InnBlock(config_path)
        self.inv3 = InnBlock(config_path)
        self.inv4 = InnBlock(config_path)

        self.inv5 = InnBlock(config_path)
        self.inv6 = InnBlock(config_path)
        self.inv7 = InnBlock(config_path)
        self.inv8 = InnBlock(config_path)

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
