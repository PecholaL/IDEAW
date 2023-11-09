""" Preprocess audio data
    * Read from .mp3, .flac, .wav files
    * STFT
    * Build Dataset
"""

import os
import random
import pickle
import yaml

from argparse import ArgumentParser

from utils import *

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-config', '-c', default='dataConfig.yaml')
    args = parser.parse_args()

    # Read from dataConfig, get hyper parameters for STFT
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    data_path = config['data_path']
    out_path = config['out_path']
    sample_rate = config['sample_rate']
    n_fft = config['n_fft']
    hop_len = config['hop_len']
    win_len = config['win_len']

    data = [] # save all audio STFT

    # Read audio data & STFT
    audio_path_list = [] # save absolute paths       
    for root_path, dirs, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root_path, file)
            if file_path.split('.')[-1].lower() in ['mp3', 'flac', 'wav']:
                audio_path_list.append(file_path)
    random.shuffle(audio_path_list)
    print(f'[Dataset]get {len(audio_path_list)} files')

    for i, audio_path in enumerate(audio_path_list):
        if i % 1000 == 0 or i == len(audio_path_list)-1:
            print(f'[Dataset]processed {i} audio files')
        # Read & Resample
        audio, _, _ = read_resample(audio_path=audio_path, 
                                   sr=sample_rate, 
                                   audio_limit_len=None)
        # STFT
        audio_stft = stft(wav=audio, 
                          n_fft=n_fft, 
                          hop_length=hop_len, 
                          win_length=win_len)
        data.append(audio_stft)

    # Dump Pickle
    with open(os.path.join(out_path, 'stft.pkl'), 'wb') as f:
        pickle.dump(data, f)
        print(f'[Dataset]dumped pickle to {out_path}')

