""" Process audio data
    * Read from .mp3, .flac, .wav files
    * Build Dataset
"""

import os
import pickle
import random
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
    audio_segment_len = config["audio_segment_len"]

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
        # Read & Resample
        audio, _, _ = read_resample(
            audio_path=audio_path, sr=sample_rate, audio_limit_len=None
        )
        audio_len = audio_len_second(audio, sample_rate)
        sample_num = int(audio_len / audio_segment_len)
        for j in range(sample_num):
            audio_segment = audio[j * sample_rate : (j + 1) * sample_rate]
            data.append(audio_segment)
        print(
            f"[Dataset]processed {i+1} audio file(s), got {len(data)} training sample(s)",
            end="\r",
        )
    print()

    # Dump Pickle
    with open(os.path.join(out_path, "audio.pkl"), "wb") as f:
        pickle.dump(data, f)
        print(f"[Dataset]dumped pickle to {os.path.join(out_path, 'audio.pkl')}")
