""" build Dataset and DataLoader
    * DataLoader provides an audio segment in STFT each time
"""

import pickle
import torch
import numpy

from torch.utils.data import Dataset, DataLoader


class CollateFn(object):
    def __init__(self):
        pass

    def __call__(self, batch):
        tensor = torch.from_numpy(numpy.array(batch))
        return tensor

class AWdataset(Dataset):
    def __init__(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, index):
        audio_stft = self.data[index]
        return audio_stft

    def __len__(self):
        return len(self.data)

def get_data_loader(dataset, batch_size, shuffle=True, num_workers=4, drop_last=False):
    _collate_fn = CollateFn()
    dataLoader = DataLoader(dataset=dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle,
                            num_workers=num_workers, 
                            collate_fn=_collate_fn, 
                            pin_memory=True,
                            drop_last=drop_last)
    return dataLoader

def infinite_iter(iterable):
    it = iter(iterable)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(iterable)

