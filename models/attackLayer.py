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
        self.load_config(config_path)
        self.GaussianNoise = GaussianNoise(self.config)
        self.Bandpass = Bandpass(self.config)
        self.Dropout = Dropout(self.config)

    def forward(self, audio):
        pass

    def load_config(self, config_path):
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)


class GaussianNoise(nn.Module):
    def __init__(self, opt, device):
        super(GaussianNoise, self).__init__()
        self.mean = opt["AttackLayer"]["GaussianNoise"]["mean"]
        self.variance = opt["AttackLayer"]["GaussianNoise"]["variance"]
        self.amplitude = opt["AttackLayer"]["GaussianNoise"]["amplitude"]
        self.p = opt["AttackLayer"]["GaussianNoise"]["p"]
        self.device = device

    def forward(self, audio):
        if random.uniform(0, 1) < self.p:
            b, c, h, w = audio.shape
            Noise = self.amplitude * torch.Tensor(
                numpy.random.normal(
                    loc=self.mean, scale=self.variance, size=(b, 1, h, w)
                )
            ).to(self.device)
            Noise = Noise.repeat(1, c, 1, 1)
            output = Noise + audio
            return output
        else:
            print("Gaussian noise error!")
            exit()


class Bandpass(nn.Module):
    def __init__(self, opt):
        super(Bandpass, self).__init__()
        self.upper = opt["AttackLayer"]["Bandpass"]["upper"]
        self.lower = opt["AttackLayer"]["Bandpass"]["lower"]

    def forward(self, audio):
        pass


class Dropout(nn.Module):
    def __init__(self, opt):
        super(Dropout, self).__init__()
        self.p = opt["AttackLayer"]["Dropout"]["p"]

    def forward(self, audio, host_audio):
        # p% bits replace with host audio
        mask = numpy.random.choice([0.0, 1.0], audio.shape[2:], p=[self.p, 1 - self.p])
        mask_tensor = torch.tensor(mask, device=audio.device, dtype=torch.float32)
        mask_tensor = mask_tensor.expand_as(audio)
        output = audio * mask_tensor + host_audio * (1 - mask_tensor)
        return output
