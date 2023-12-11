""" Attack Simulate Layer
    configuration is located at IDEAW/models/config.yaml
    Each attack is designed in an sample-wise form, NOT FOR BATCH
    * Gaussian Noise
    * Bandpass
    * Random Dropout
    * Resample
    * Amplitude Modify
    * Lossy Compress
    * Time Stretch
    * ...
"""

import librosa
import math
import numpy
import pydub
import random
import resampy
import torch
import torch.nn as nn
import yaml

from scipy import signal
from scipy.io.wavfile import write


class AttackLayer(nn.Module):
    def __init__(self, config_path, device):
        super(AttackLayer, self).__init__()
        self.load_config(config_path)
        self.att_num = self.config["AttackLayer"]["att_num"]
        self.gaussianNoise = GaussianNoise(self.config, device)
        self.bandpass = Bandpass(self.config, device)
        self.erase = Erase(self.config, device)
        self.dropout = Dropout(self.config, device)
        self.resample = Resample(self.config, device)
        self.ampMdf = AmplitudeModify(self.config)
        self.mp3compress = Mp3Compress(self.config, device)
        self.timeStretch = TimeStretch(self.config, device)

    """ Attack layer strategy:
        Each attack is sample-wise. 
        During training, the attack is randomly selected and 
        applied to EACH sample in the batch.
    """

    def forward(self, audio_batch, host_audio_batch):
        # orig shape [B, L]
        batch_size = audio_batch.shape[0]
        att_audio_list = []
        for i in range(batch_size):
            audio = audio_batch[i].squeeze()
            # att_index = random.randint(1, self.att_num)
            att_index = 8
            if att_index == 1:
                att_audio = self.gaussianNoise(audio)
            elif att_index == 2:
                att_audio = self.bandpass(audio)
            elif att_index == 3:
                att_audio = self.erase(audio)
            elif att_index == 4:
                host_audio = host_audio_batch[i].squeeze()
                att_audio = self.dropout(audio, host_audio)
            elif att_index == 5:
                att_audio = self.resample(audio)
            elif att_index == 6:
                att_audio = self.ampMdf(audio)
            elif att_index == 7:
                att_audio = self.mp3compress(audio)
            else:  # elif att_index == 8:
                att_audio = self.timeStretch(audio)
            att_audio_list.append(att_audio)
        ret = torch.stack(att_audio_list)
        return ret

    def load_config(self, config_path):
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)


class GaussianNoise(nn.Module):
    def __init__(self, opt, device):
        super(GaussianNoise, self).__init__()
        self.snr = opt["AttackLayer"]["GaussianNoise"]["snr"]
        self.device = device

    def forward(self, audio):  # input: audio wave sample
        l = len(audio)
        noise = torch.rand(l).to(self.device)
        p_s = torch.sum(audio**2) / l
        p_n = torch.sum(noise**2) / l
        k = math.sqrt(p_s / (10 ** (self.snr / 10) * p_n))
        noise_ = noise * k

        ret = audio + noise_
        return ret


class Bandpass(nn.Module):
    def __init__(self, opt, device):
        super(Bandpass, self).__init__()
        self.sr = opt["AttackLayer"]["Bandpass"]["sr"]
        self.upper = opt["AttackLayer"]["Bandpass"]["upper"]
        self.lower = opt["AttackLayer"]["Bandpass"]["lower"]
        self.device = device
        self.b, self.a = signal.butter(
            8,
            [2 * self.lower / self.sr, 2 * self.upper / self.sr],
            "bandpass",
        )

    def forward(self, audio):
        ret = signal.filtfilt(self.b, self.a, audio.cpu().detach())
        ret = torch.from_numpy(ret.copy()).float().to(self.device)
        return ret


class Erase(nn.Module):
    def __init__(self, opt, device):
        super(Erase, self).__init__()
        self.p = opt["AttackLayer"]["Erase"]["p"]
        self.device = device

    def forward(self, audio):
        mask = numpy.random.choice([0.0, 1.0], len(audio), p=[self.p, 1 - self.p])
        mask_tensor = torch.tensor(mask, device=audio.device, dtype=torch.float32)
        ret = audio * mask_tensor
        return ret.to(self.device)


class Dropout(nn.Module):
    def __init__(self, opt, device):
        super(Dropout, self).__init__()
        self.p = opt["AttackLayer"]["Dropout"]["p"]
        self.device = device

    def forward(self, audio, host_audio):
        # p% bits replace with host audio
        mask = numpy.random.choice([0.0, 1.0], len(audio), p=[self.p, 1 - self.p])
        mask_tensor = torch.tensor(mask, device=audio.device, dtype=torch.float32)
        ret = audio * mask_tensor + host_audio * (1 - mask_tensor)
        return ret


class Resample(nn.Module):
    def __init__(self, opt, device):
        super(Resample, self).__init__()
        self.orig_sr = opt["AttackLayer"]["Resample"]["orig_sr"]
        self.sr = opt["AttackLayer"]["Resample"]["sr"]
        self.device = device

    def forward(self, audio):
        audio = resampy.resample(audio.cpu().detach().numpy(), self.orig_sr, self.sr)
        audio = resampy.resample(audio, self.sr, self.orig_sr)
        audio = torch.from_numpy(audio).float().to(self.device)
        return audio


class AmplitudeModify(nn.Module):
    def __init__(self, opt):
        super(AmplitudeModify, self).__init__()
        self.f = opt["AttackLayer"]["AmplitudeModify"]["f"]

    def forward(self, audio):
        return audio * self.f


class Mp3Compress(nn.Module):
    def __init__(self, opt, device):
        super(Mp3Compress, self).__init__()
        self.sr = opt["AttackLayer"]["Mp3Compress"]["sr"]
        self.bitrate = opt["AttackLayer"]["Mp3Compress"]["bitrate"]
        self.device = device

    def forward(self, audio):
        write("tmp.wav", self.sr, audio.detach().numpy())
        wav_segment = pydub.AudioSegment.from_wav("tmp.wav")
        wav_segment.export(
            "tmp.mp3",
            format="mp3",
            bitrate=self.bitrate,
        )
        mp3_segment = pydub.AudioSegment.from_mp3("tmp.mp3")
        mp3_segment.export("tmp.wav", format="wav")
        wav, _ = librosa.load("tmp.wav", sr=self.sr)
        return torch.from_numpy(wav.copy()).float().to(self.device)


class TimeStretch(nn.Module):
    def __init__(self, opt, device):
        super(TimeStretch, self).__init__()
        self.tsr = opt["AttackLayer"]["TimeStretch"]["rate"]
        self.device = device

    def forward(self, audio):
        l = len(audio)
        audio_s_1 = librosa.effects.time_stretch(
            audio.cpu().detach().numpy(), rate=self.tsr
        )
        l_s_1 = len(audio_s_1)
        tsr_r = l_s_1 / l + 0.00000001  # for sure that len(audio_s_2)>len(audio)
        audio_s_2 = librosa.effects.time_stretch(audio_s_1, rate=tsr_r)
        audio = audio_s_2[:l]
        return torch.from_numpy(audio).float().to(self.device)
