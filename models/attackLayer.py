""" Attack Simulate Layer
    configuration is located at IDEAW/models/config
    * Gaussian Noise
    * Reverberation
    * MP3 compress
    * ...
"""

import numpy
import random
import torch
import torch.nn as nn
import yaml


class AttackLayer(nn.Module):
    def __init__(self, config_path):
        super(AttackLayer, self).__init__()

    def forward(self, audio):
        pass


class GaussianNoise(nn.Module):
    def __init__(self, opt, device):
        super(GaussianNoise, self).__init__()
        self.mean = opt["noise"]["GaussianNoise"]["mean"]
        self.variance = opt["noise"]["GaussianNoise"]["variance"]
        self.amplitude = opt["noise"]["GaussianNoise"]["amplitude"]
        self.p = opt["noise"]["GaussianNoise"]["p"]
        self.device = device

    def forward(self, encoded, cover_img=None):
        if random.uniform(0, 1) < self.p:
            b, c, h, w = encoded.shape
            Noise = self.amplitude * torch.Tensor(
                numpy.random.normal(
                    loc=self.mean, scale=self.variance, size=(b, 1, h, w)
                )
            ).to(self.device)
            Noise = Noise.repeat(1, c, 1, 1)
            #
            img_ = Noise + encoded

            return img_

        else:
            print("Gaussian noise error!")
            exit()
