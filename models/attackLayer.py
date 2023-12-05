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

import io
import librosa
import math
import numpy
import pydub
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
        self.ampMdf = AmplitudeModify(self.config)
        self.mp3compress = Mp3Compress(self.config, device)
        self.timeStretch = TimeStretch(self.config, device)

    def forward(self, audio_batch):
        pass

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
        mask = numpy.random.choice([0.0, 1.0], len(audio), p=[self.p, 1 - self.p])
        mask_tensor = torch.tensor(mask, device=audio.device, dtype=torch.float32)
        ret = audio * mask_tensor + host_audio * (1 - mask_tensor)
        return ret


class Resample(nn.Module):
    def __init__(self, opt, device):
        super(Resample, self).__init__()
        self.sr = opt["AttackLayer"]["Resample"]["sr"]
        self.device = device

    def forward(self, audio):
        audio = resampy.resample(audio.numpy(), 16000, self.sr)
        audio = resampy.resample(audio, self.sr, 16000)
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
        self.bitrate = opt["AttackLayer"]["Mp3Compress"]["bitrate"]
        self.device = device

    def forward(self, audio):
        wav = audio.numpy()
        sample_width = wav.dtype.itemsize
        wav_segment = pydub.AudioSegment(
            wav.tobytes(), frame_rate=16000, sample_width=sample_width, channels=1
        )
        mp3_byte = wav_segment.export(format="mp3", bitrate=self.bitrate).read()
        mp3_segment = pydub.AudioSegment.from_file(
            io.BytesIO(mp3_byte),
            format="mp3",
            frame_rate=16000,
            sample_width=sample_width,
        )
        mp3_segment = mp3_segment.set_frame_rate(16000).set_sample_width(sample_width)
        wav_byte = mp3_segment.export(format="wav").read()
        wav = numpy.frombuffer(wav_byte, dtype=numpy.float32)
        return torch.from_numpy(wav.copy()).float().to(self.device)


class TimeStretch(nn.Module):
    def __init__(self, opt, device):
        super(TimeStretch, self).__init__()
        self.tsr = opt["AttackLayer"]["TimeStretch"]["rate"]
        self.device = device

    def forward(self, audio):
        l = len(audio)
        audio_s_1 = librosa.effects.time_stretch(audio.numpy(), rate=self.tsr)
        l_s_1 = len(audio_s_1)
        tsr_r = l_s_1 / l + 0.00000001  # for sure that len(audio_s_2)>len(audio)
        audio_s_2 = librosa.effects.time_stretch(audio_s_1, rate=tsr_r)
        audio = audio_s_2[:l]
        return torch.from_numpy(audio).float().to(self.device)
