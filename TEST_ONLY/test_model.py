""" Test Only
"""

import sys
import torch

sys.path.append("..")
from models.ideaw import IDEAW
from data.dataset import AWdataset, get_data_loader, infinite_iter

IDEAW = IDEAW("../models/config.yaml", "cpu")
print(f"toral parameter count: {sum(x.numel() for x in IDEAW.parameters())}")

# prepare data
dataset = AWdataset("../../Watermark/miniAWdata_pickle/audio.pkl")
loader = get_data_loader(dataset=dataset, batch_size=1, num_workers=0)
INF_loader = infinite_iter(loader)

data = next(INF_loader)
print(f"batch data shape: {data.shape}")


# embedding&extracting process test
## shape check
audio = torch.FloatTensor(data[0].float().unsqueeze(0)).to("cpu")
audio_stft = IDEAW.stft(audio)
msg = [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1]  # 16bit
lcode = [1, 1, 0, 0, 0, 1, 1, 1, 0, 0]  # 10bit
msg = torch.FloatTensor(torch.tensor(msg).float().unsqueeze(0)).to("cpu")
lcode = torch.FloatTensor(torch.tensor(lcode).float().unsqueeze(0)).to("cpu")
print(f"audio shape: {audio.shape}")
print(f"stft audio shape: {audio_stft.shape}")
print(f"msg&lcode shape: {msg.shape}&{lcode.shape}")


# # embedding msg and lcode into 1 second chunk
# chunk_size = 16000
# chunk = audio[:, 0 : 0 + chunk_size]  # 1s

# chunk_wmd, chunk_wmd_stft = IDEAW.embed_msg(chunk, msg)
# chunk_wmd = chunk_wmd.detach().cpu()
# print(f"watermarked audio shape: {chunk_wmd.shape}")

# chunk_wmd_lcd, chunk_wmd_lcd_stft = IDEAW.embed_lcode(chunk_wmd_stft, lcode)
# print(f"lcode embedded audio shape: {chunk_wmd_lcd.shape}")


# # extracing lcode and msg from chunk_wmd_lcd
# mid_stft, extr_lcode = IDEAW.extract_lcode(chunk_wmd_lcd_stft)
# extr_lcode = extr_lcode.int().detach()
# print(f"extracted lcode shape: {extr_lcode.shape}")
# print(f"mid signal shape: {mid_stft.shape}")

# extr_msg = IDEAW.extract_msg(mid_stft).int().detach().cpu().numpy()
# print(f"extracted msg shape: {extr_msg.shape}")


# test the forward of IDEAW
print("Input dtype:")
print(data.dtype)
print(msg.dtype)
print(lcode.dtype)
(
    audio_wmd1,
    audio_wmd1_stft,
    audio_wmd2,
    audio_wmd2_stft,
    msg_extr1,
    msg_extr2,
    lcode_extr,
    orig_output,
    wmd_output,
) = IDEAW(data, msg, lcode, True)


print("Output shape:")
print(audio_wmd1.shape)
print(audio_wmd1_stft.shape)
print(audio_wmd2.shape)
print(audio_wmd2_stft.shape)
print(msg_extr1.shape)
print(msg_extr2.shape)
print(lcode_extr.shape)
print(orig_output.shape)
print(wmd_output.shape)
