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
d = torch.from_numpy(d).float()
print(f"audio shape: {d.shape}, audio device: {d.device}")

# GNoise
d_ = att.gaussianNoise(d)
print("Gaussian Noise")
print(d_.shape, d_.device)

# # BPass
# d_ = att.bandpass(d)
# print("Band Pass")
# print(d_.shape, d_.device)

# # Dropout
# d_ = att.dropout(d, d)
# print("Dropout")
# print(d_.shape, d_.device)

# # Resample
# d_ = att.resample(d)
# print("Resample")
# print(d_.shape, d_.device)

# # AModify
# d_ = att.ampMdf(d)
# print("Amplitude Modify")
# print(d_.shape, d_.device)

# Mp3Compress
# d_ = att.mp3compress(d)
# print("Mp3 Compress")
# print(d_.shape, d_.device)
# d_np = d_.numpy()


# # Time Stretch
# d_ = att.timeStretch(d)
# print("Time Stretch")
# print(d_.shape, d_.device)

d_np = d_.numpy()
write("/Users/pecholalee/Coding/Watermark/ideaw_data/output/att_test.wav", 16000, d_np)
