""" Test the locating speed of IDEAW
"""

import sys
import time
import torch
import warnings

sys.path.append("..")
warnings.filterwarnings("ignore")

from models.ideaw import IDEAW

# mini config
msg_bit = 16
lcode_bit = 10

# config paths
config_model_path = "/Users/pecholalee/Coding/IDEAW/models/config.yaml"
ckpt_path = "/Users/pecholalee/Coding/Watermark/ideaw_data/save/stage_I/ideaw.ckpt"
device = "cpu"

# data prepare
position = int(input("Embedding position (seconds): "))
virtual_length = position + 1
virtual_data = torch.rand(virtual_length * 16000).unsqueeze(0)

# model prepare
ideaw = IDEAW(config_model_path, device)
ideaw.load_state_dict(torch.load(ckpt_path))

# locating, assume that ACC(l.code)=100%
start_time = time.time()
for i in range(position * 16000 + 1):
    chunk = virtual_data[:, i : i + 16000]
    _, _ = ideaw.extract_lcode(chunk)

end_time = time.time()
cost_time = end_time - start_time
print(f"time consumn: {cost_time}")
