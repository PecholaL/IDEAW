""" Test Only
"""

import sys
import torch

sys.path.append("..")
from models.attackLayer import AttackLayer

att = AttackLayer("config.yaml", "cpu")

# audio tensor batch
data = torch.rand([8, 18000])

d = data[0].squeeze()
print(f"audio shape: {d.shape}, audio device: {d.device}")

# # GNoise
# d_ = att.gaussianNoise(d)
# print("Gaussian Noise")
# print(d_.shape, d_.device)

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
d_ = att.mp3compress(d)
print("Mp3 Compress")
print(d_.shape, d_.device)

# # Time Stretch
# d_ = att.timeStretch(d)
# print("Time Stretch")
# print(d_.shape, d_.device)
