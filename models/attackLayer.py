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
        self.gaussianNoise = GaussianNoise(self.config)
        self.bandpass = Bandpass(self.config)
        self.dropout = Dropout(self.config)
        self.resample = Resample(self.config)

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
        # input: audio wave
        noise = torch.randn(len(audio))
        power_audio = torch.sum(audio**2) / len(audio)
        power_noise = torch.sum(noise**2) / len(noise)


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


class Resample(nn.Module):
    def __init__(self, opt):
        super(Dropout, self).__init__()
        self.p = opt["AttackLayer"]["Resample"]["p"]

    def forward(self, audio):
        pass


class AmplitudeModify(nn.Module):
    def __init__(self, opt):
        super(Dropout, self).__init__()
        self.p = opt["AttackLayer"]["AmplitudeModify"]["p"]

    def forward(self, audio):
        pass
