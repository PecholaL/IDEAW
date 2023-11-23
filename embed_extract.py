""" Test the trained IDEAW (hard coding, only for specific test)
    * get the marked audio
    * extract the msg and compute ACC
    * ...
"""

import torch
import torch.nn as nn
import tqdm

from models.ideaw import IDEAW
from data.process import read_resample

# config paths
config_model_path = "/Users/pecholalee/Coding/IDEAW/models/config.yaml"
ckpt_path = "/Users/pecholalee/Coding/Watermark/ideaw_data/save/ideaw.ckpt"

# audio path
audio_path = "/Users/pecholalee/Coding/Watermark/miniAWdata/p225_008.wav"

# build model and load trained parameters
ideaw = IDEAW(config_model_path)
print("[IDEAW]model built")

ideaw.load_state_dict(torch.load(ckpt_path))
print("[IDEAW]model loaded")

# generate msg and lcode
watermark_msg = torch.randint(0, 2, (1, 16), dtype=torch.float)
watermark_msg[watermark_msg == 0] = -1
locate_code = torch.randint(0, 2, (1, 10), dtype=torch.float)

with torch.no_grad():
    ideaw.eval()

    audio, _, _ = read_resample(audio_path=audio_path, sr=16000, audio_limit_len=None)
    chunk_size = 16000 * (1 + 0.5)
    chunk_num = int(len(audio) / chunk_size)

    it = range(chunk_num)
    it = tqdm.tqdm(it, desc="Embedding")
    for i in it:
        start_pos = i * chunk_size
        chunk = audio[start_pos, start_pos + chunk_size].copy()

        # embed msg
        audio_wmd1, audio_wmd1_stft = ideaw.embed_msg(chunk, watermark_msg)
        # embed lcode
        audio_wmd2, _ = ideaw.embed_lcode(audio_wmd1_stft, locate_code)

        # output the watermarked wave
