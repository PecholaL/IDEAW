""" Attack Simulate Layer
    configuration is located at IDEAW/models/config
    * Gaussian Noise
    * Reverberation
    * MP3 compress
    * ...
"""

import math
import numpy
import random
import resampy
import torch
import torch.nn as nn
import yaml

from scipy import signal


class AttackLayer(nn.Module):
    def __init__(self, config_path, device):
        super(AttackLayer, self).__init__()
        self.load_config(config_path)
        self.gaussianNoise = GaussianNoise(self.config, device)
        self.bandpass = Bandpass(self.config, device)
        self.dropout = Dropout(self.config, device)
        self.resample = Resample(self.config, device)
        self.ampMdf = AmplitudeModify(self.config, device)

    def forward(self, audio):
        pass

    def load_config(self, config_path):
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)


class GaussianNoise(nn.Module):
    def __init__(self, opt, device):
        super(GaussianNoise, self).__init__()
        self.snr = opt["AttackLayer"]["GaussianNoise"]["snr"]
        self.device = device

    def forward(self, audio):  # input: audio wave batch
        B = audio.shape[0]
        L = audio.shape[1]
        noise = torch.rand([B, L]).to(self.device)
        p_s = torch.sum(audio**2) / (B * L)
        p_n = torch.sum(noise**2) / (B * L)
        k = math.sqrt(p_s / (10 ** (self.snr / 10) * p_n))
        noise_ = noise * k

        ret = audio + noise_
        return ret


class Bandpass(nn.Module):
    def __init__(self, opt, device):
        super(Bandpass, self).__init__()
        self.upper = opt["AttackLayer"]["Bandpass"]["upper"]
        self.lower = opt["AttackLayer"]["Bandpass"]["lower"]
        self.device = device
        self.b, self.a = signal.butter(
            8, [2 * self.lower / 16000, 2 * self.upper / 16000], "bandpass"
        )

    def forward(self, audio):
        ret = signal.filtfilt(self.b, self.a, audio)
        ret = torch.from_numpy(ret.copy()).float().to(self.device)
        return ret


class Dropout(nn.Module):
    def __init__(self, opt, device):
        super(Dropout, self).__init__()
        self.p = opt["AttackLayer"]["Dropout"]["p"]
        self.device = device

    def forward(self, audio, host_audio):
        # p% bits replace with host audio
        mask = numpy.random.choice([0.0, 1.0], audio.shape[1], p=[self.p, 1 - self.p])
        mask_tensor = torch.tensor(mask, device=audio.device, dtype=torch.float32)
        mask_tensor = mask_tensor.expand_as(audio)
        ret = audio * mask_tensor + host_audio * (1 - mask_tensor)
        return ret


class Resample(nn.Module):
    def __init__(self, opt, device):
        super(Dropout, self).__init__()
        self.sr = opt["AttackLayer"]["Resample"]["sr"]
        self.device = device

    def forward(self, audio):
        audio = resampy.resample(audio, 16000, self.sr)
        audio = resampy.resample(audio, self.sr, 16000)
        audio = torch.from_numpy(audio).float().to(self.device)
        return audio


class AmplitudeModify(nn.Module):
    def __init__(self, opt):
        super(Dropout, self).__init__()
        self.p = opt["AttackLayer"]["AmplitudeModify"]["p"]

    def forward(self, audio):
        pass
