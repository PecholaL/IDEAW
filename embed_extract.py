""" Test the trained IDEAW
    * get the marked audio
    * extract the msg and compute ACC
    * ...
"""

import torch
import torch.nn as nn
import yaml

from models.ideaw import IDEAW

# paths
config_model_path = "/Users/pecholalee/Coding/IDEAW/models/config.yaml"
ckpt_path = "/Users/pecholalee/Coding/Watermark/ideaw_data/save/ideaw.ckpt"

# build model and load trained parameters
ideaw = IDEAW(config_model_path)
print("[IDEAW]model built")

ideaw.load_state_dict(torch.load(ckpt_path))
print("[IDEAW]model loaded")
