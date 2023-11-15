""" Process audio data
    * Read from .mp3, .flac, .wav files
    * STFT
    * Build Dataset
"""

import os
import random
import pickle
import yaml

from utils import *

if __name__ == "__main__":
    config_path = "./data/config.yaml"

    # Read from dataConfig, get hyper parameters for STFT
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_path = config["data_path"]
    out_path = config["out_path"]
    sample_rate = config["sample_rate"]
    audio_limit_len = config["audio_limit_len"]

    data = []  # save all audio signal

    # Read audio data
    audio_path_list = []  # save absolute paths of audio files
    for root_path, dirs, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root_path, file)
            if file_path.split(".")[-1].lower() in ["mp3", "flac", "wav"]:
                audio_path_list.append(file_path)
    random.shuffle(audio_path_list)
    print(f"[Dataset]got {len(audio_path_list)} audio files")

    for i, audio_path in enumerate(audio_path_list):
        if i % 1000 == 0 or i == len(audio_path_list) - 1:
            print(f"[Dataset]processed {i} audio files")
        # Read & Resample
        audio, _, _ = read_resample(
            audio_path=audio_path, sr=sample_rate, audio_limit_len=audio_limit_len
        )
        data.append(audio)

    # Dump Pickle
    with open(os.path.join(out_path, "audio.pkl"), "wb") as f:
        pickle.dump(data, f)
        print(f"[Dataset]dumped pickle to {out_path}")
