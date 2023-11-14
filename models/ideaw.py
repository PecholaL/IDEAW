""" IDEAW
    * Embed & Extract functions
"""

import torch
import torch.nn as nn
import yaml

from mihnet import Mihnet_s1, Mihnet_s2


class IDEAW(nn.Module):
    def __init__(self, config_path):
        super(IDEAW, self).__init__()
        self.load_config(config_path)
        self.hinet = Mihnet_s1(config_path)
        self.watermark_fc = nn.Linear(self.num_bit, self.num_point)
        self.watermark_fc_back = nn.Linear(self.num_point, self.num_bit)

    def load_config(self, config_path):
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.win_len = config["IDEAW"]["win_len"]
            self.n_fft = config["IDEAW"]["n_fft"]
            self.hop_len = config["IDEAW"]["hop_len"]
            self.num_bit = config["IDEAW"]["num_bit"]
            self.num_point = config["IDEAW"]["num_point"]

    def stft(self, data):
        window = torch.hann_window(self.win_len).to(data.device)
        ret = torch.stft(
            input=data,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            window=window,
            return_complex=False,
        )
        return ret  # [B, F, T, C]

    def istft(self, data):
        window = torch.hann_window(self.win_len).to(data.device)
        ret = torch.istft(
            input=data,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            window=window,
            return_complex=False,
        )
        return ret

    def embed(self, audio, msg):
        audio_stft = self.stft(audio)
        msg_expand = self.watermark_fc(msg)
        print(f"expand msg shape: {msg_expand.shape}")
        msg_stft = self.stft(msg_expand)
        print(f"stft:{audio_stft.shape}, {msg_stft.shape}")
        wm_audio_stft, _ = self.enc_dec(audio_stft, msg_stft, rev=False)
        wm_audio = self.istft(wm_audio_stft)

        return wm_audio

    def extract(self, wm_audio):
        wm_audio_stft = self.stft(wm_audio)
        aux_signal_stft = wm_audio_stft
        _, extr_msg_expand_stft = self.enc_dec(wm_audio_stft, aux_signal_stft, rev=True)
        extr_msg_expand = self.istft(extr_msg_expand_stft)
        extr_msg = self.watermark_fc_back(extr_msg_expand).clamp(-1, 1)
        return extr_msg

    def enc_dec(self, audio, msg, rev):
        audio = audio.permute(0, 3, 2, 1)  # [B, C, T, F]
        msg = msg.permute(0, 3, 2, 1)

        audio_, msg_ = self.hinet(audio, msg, rev)

        return audio_.permute(0, 3, 2, 1), msg_.permute(0, 3, 2, 1)
