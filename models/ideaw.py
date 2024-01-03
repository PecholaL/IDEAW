""" IDEAW
    * Embed & Extract functions
"""

import random
import torch
import torch.nn as nn
import yaml

from models.mihnet import Mihnet_s1, Mihnet_s2
from models.componentNet import Discriminator, BalanceBlock
from models.attackLayer import AttackLayer


class IDEAW(nn.Module):
    def __init__(self, config_path, device):
        super(IDEAW, self).__init__()
        self.load_config(config_path)
        self.hinet_1 = Mihnet_s1(config_path, self.num_inn_1)  # for embedding msg
        self.hinet_2 = Mihnet_s2(config_path, self.num_inn_2)  # for embedding lcode
        self.msg_fc = nn.Linear(self.num_bit, self.num_point)
        self.msg_fc_back = nn.Linear(self.num_point, self.num_bit)
        self.lcode_fc = nn.Linear(
            self.num_lc_bit, int(self.num_point / self.chunk_ratio)
        )
        self.lcode_fc_back = nn.Linear(
            int(self.num_point / self.chunk_ratio), self.num_lc_bit
        )
        self.discriminator = Discriminator(config_path)
        self.attack_layer = AttackLayer(config_path, device)
        self.balance_block = BalanceBlock(config_path)

    def forward(self, audio, msg, lcode, robustness, shift):
        audio_wmd1, audio_wmd1_stft = self.embed_msg(audio, msg)
        msg_extr1 = self.extract_msg(audio_wmd1_stft)
        audio_wmd2, audio_wmd2_stft = self.embed_lcode(audio_wmd1, lcode)

        if shift == True:
            host_audio_stft = self.stft(audio)
            audio_wmd2_stft = self.shift(
                host_audio_stft, audio_wmd2_stft, self.extract_stripe
            )
            audio_wmd2 = self.istft(audio_wmd2_stft)

        if robustness == False:
            mid_stft, lcode_extr = self.extract_lcode(audio_wmd2)
            msg_extr2 = self.extract_msg(mid_stft)

        else:  # robustness == True
            # robustness training
            audio_att = self.attack_layer(audio_wmd2, audio)
            audio_att_stft = self.stft(audio_att)
            audio_att_stft = self.balance_block(audio_att_stft)
            mid_stft, lcode_extr = self.extract_lcode(audio_att)
            msg_extr2 = self.extract_msg(mid_stft)

        orig_output = self.discriminator(audio)
        wmd_output = self.discriminator(audio_wmd2)

        return (
            audio_wmd1,
            audio_wmd1_stft,
            audio_wmd2,
            audio_wmd2_stft,
            msg_extr1,
            msg_extr2,
            lcode_extr,
            orig_output,
            wmd_output,
        )

    def load_config(self, config_path):
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.win_len = config["IDEAW"]["win_len"]
            self.n_fft = config["IDEAW"]["n_fft"]
            self.hop_len = config["IDEAW"]["hop_len"]
            self.num_inn_1 = config["IDEAW"]["num_inn_1"]
            self.num_inn_2 = config["IDEAW"]["num_inn_2"]
            self.num_bit = config["IDEAW"]["num_bit"]
            self.num_lc_bit = config["IDEAW"]["num_lc_bit"]
            self.num_point = config["IDEAW"]["num_point"]
            self.chunk_ratio = config["IDEAW"]["chunk_ratio"]
            self.extract_stripe = config["IDEAW"]["extract_stripe"]

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

    # INN#1 Embedding & Extracting watermark message
    def embed_msg(self, audio, msg):
        audio_stft = self.stft(audio)
        msg_expand = self.msg_fc(msg)
        msg_stft = self.stft(msg_expand)
        wm_audio_stft, _ = self.enc_dec_1(audio_stft, msg_stft, rev=False)
        wm_audio = self.istft(wm_audio_stft)

        return wm_audio, wm_audio_stft

    def extract_msg(self, wm_mid_stft):
        aux_signal_stft = wm_mid_stft
        _, extr_msg_expand_stft = self.enc_dec_1(wm_mid_stft, aux_signal_stft, rev=True)
        extr_msg_expand = self.istft(extr_msg_expand_stft)
        extr_msg = self.msg_fc_back(extr_msg_expand).clamp(-1, 1)

        return extr_msg

    def enc_dec_1(self, audio_stft, msg_stft, rev):
        audio_stft = audio_stft.permute(0, 3, 2, 1)  # [B, C, T, F]
        msg_stft = msg_stft.permute(0, 3, 2, 1)

        audio_stft_, msg_stft_ = self.hinet_1(audio_stft, msg_stft, rev)

        return audio_stft_.permute(0, 3, 2, 1), msg_stft_.permute(0, 3, 2, 1)

    def shift(self, host_audio_stft, wmd_audio_stft, step_size):
        X = random.randint(0, step_size)
        for i in range(X):
            wmd_audio_stft[:, :, i, :] = host_audio_stft[:, :, i, :]
        return wmd_audio_stft

    # INN#2 Embedding & Extracting watermark locating code
    def embed_lcode(self, audio, lcode):
        lcode_expand = self.lcode_fc(lcode)
        lcode_stft = self.stft(lcode_expand)
        # l_code will be embedded into the head of the audio
        audio_1_stft = self.stft(audio[:, : int(self.num_point / self.chunk_ratio)])
        audio_2 = audio[:, int(self.num_point / self.chunk_ratio) :]
        wm_audio_1_stft, _ = self.enc_dec_2(audio_1_stft, lcode_stft, rev=False)
        wm_audio_1 = self.istft(wm_audio_1_stft)
        wm_audio = torch.concat([wm_audio_1, audio_2], dim=1)
        wm_audio_stft = self.stft(wm_audio)

        return wm_audio, wm_audio_stft

    def extract_lcode(self, wm_audio):
        wm_audio_1_stft = self.stft(
            wm_audio[:, : int(self.num_point / self.chunk_ratio)]
        )
        wm_audio_2 = wm_audio[:, int(self.num_point / self.chunk_ratio) :]
        aux_signal_stft = wm_audio_1_stft
        mid_stft, extr_lcode_expand_stft = self.enc_dec_2(
            wm_audio_1_stft, aux_signal_stft, rev=True
        )
        extr_lcode_expand = self.istft(extr_lcode_expand_stft)
        extr_lcode = self.lcode_fc_back(extr_lcode_expand).clamp(-1, 1)
        mid_1 = self.istft(mid_stft)
        mid = torch.concat([mid_1, wm_audio_2], dim=1)
        mid_stft = self.stft(mid)

        return mid_stft, extr_lcode

    def enc_dec_2(self, audio_stft, lcode_stft, rev):
        audio_stft = audio_stft.permute(0, 3, 2, 1)  # [B, C, T, F]
        lcode_stft = lcode_stft.permute(0, 3, 2, 1)

        audio_stft_, lcode_stft_ = self.hinet_2(audio_stft, lcode_stft, rev)

        return audio_stft_.permute(0, 3, 2, 1), lcode_stft_.permute(0, 3, 2, 1)
