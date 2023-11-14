""" Test Only
"""

import sys
import torch

sys.path.append("..")
from ideaw import IDEAW
from data.dataset import AWdataset, get_data_loader, infinite_iter

IDEAW = IDEAW("config.yaml")

dataset = AWdataset("../../Watermark/miniAWdata_pickle/stft.pkl")
loader = get_data_loader(dataset=dataset, batch_size=10, num_workers=0)
INF_loader = infinite_iter(loader)

data = next(INF_loader)
print(f"batch data shape: {data.shape}")

# embedding process test
audio = torch.FloatTensor(data[0].float().unsqueeze(0)).to("cpu")
print(f"audio shape: {audio.shape}")
audio_stft = IDEAW.stft(audio)
print(f"stft audio shape: {audio_stft.shape}")
msg = [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1]
pos_code = [1, 1, 1, 1, 0, 0, 0, 0]
msg = torch.FloatTensor(torch.tensor(msg).float().unsqueeze(0)).to("cpu")
print(f"msg shape: {msg.shape}")

chunk_size = 16000
chunk = audio[:, 0 : 0 + chunk_size]

chunk_wmd = IDEAW.embed(chunk, msg).detach().cpu()
print(f"watermarked audio shape: {chunk_wmd.shape}")

extr_msg = IDEAW.extract(chunk_wmd.to("cpu")).int().detach().cpu().numpy()
print(f"extracted msg shape: {extr_msg.shape}")
