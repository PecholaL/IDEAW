""" Test Only
"""

import sys
import torch
import soundfile
from scipy.io.wavfile import write

sys.path.append("..")
from models.attackLayer import AttackLayer

att = AttackLayer("../models/config.yaml", "cpu")

# audio tensor batch
# data = torch.rand([8, 28000])
# d = data[0].squeeze()

d, sr = soundfile.read("/Users/pecholalee/Coding/Watermark/miniAWdata/p225_003.wav")
d = d[:96150]
d = torch.from_numpy(d).float()
print(f"audio shape: {d.shape}, audio device: {d.device}")

# GNoise
d_gn = att.gaussianNoise(d)
print("Gaussian Noise")
print(d_gn.shape, d_gn.device)

# BPass
d_bpf = att.bandpass(d)
print("Band Pass")
print(d_bpf.shape, d_bpf.device)

# Dropout
d_do = att.dropout(d, d)
print("Dropout")
print(d_do.shape, d_do.device)

# Resample
d_rs = att.resample(d)
print("Resample")
print(d_rs.shape, d_rs.device)

# AModify
d_am = att.ampMdf(d)
print("Amplitude Modify")
print(d_am.shape, d_am.device)

# Mp3Compress
d_mp = att.mp3compress(d)
print("Mp3 Compress")
print(d_mp.shape, d_mp.device)

# Time Stretch
d_ts = att.timeStretch(d)
print("Time Stretch")
print(d_ts.shape, d_ts.device)

# plot
d_np = d_mp.numpy()
write("/Users/pecholalee/Coding/Watermark/ideaw_data/output/att_test.wav", 16000, d_np)
