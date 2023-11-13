""" Test Only
"""

import sys

sys.path.append("..")
from ideaw import IDEAW
from data.dataset import AWdataset, get_data_loader, infinite_iter

IDEAW = IDEAW("config.yaml")

dataset = AWdataset("../../Watermark/miniAWdata_pickle/stft.pkl")
loader = get_data_loader(dataset=dataset, batch_size=10, num_workers=0)
INF_loader = infinite_iter(loader)

data = next(INF_loader)
print(f"Batch data shape: {data.shape}")

data_stft = IDEAW.stft(data)
print(f"stft data shape: {data_stft.shape}")

data_istft = IDEAW.istft(data_stft)
print(f"istft data shape: {data_istft.shape}")
