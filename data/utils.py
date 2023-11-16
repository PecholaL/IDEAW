""" Read audio data from file
    * Func1: Read & Resample
"""

import os
import soundfile
import librosa
import resampy
import numpy

""" Get single-channel audio and resample to 16kHz
"""


def read_resample(audio_path, sr=16000, audio_limit_len=None):
    assert os.path.exists(audio_path)
    # get file extension (i.e. wav, mp3, flac)
    audio_type = audio_path.split(".")[-1].lower()

    if audio_type == "mp3":
        data, origin_sr = librosa.load(audio_path, sr=None)
    elif audio_type in ["wav", "flac"]:
        data, origin_sr = soundfile.read(audio_path)
    else:
        raise Exception("[Error]unsupported file type: " + audio_type)

    # resample
    if origin_sr != sr:
        data = resampy.resample(data, origin_sr, sr)

    # limit to setted length (unit: second)
    audio_len = audio_len_second(data, sr)
    if audio_limit_len is not None:
        assert len(data) > 0
        if audio_len < audio_limit_len:
            repeats = int(audio_limit_len / audio_len) + 1
            data = numpy.tile(data, repeats)
        data = data[0 : sr * audio_limit_len]

    return data, sr, audio_len


def audio_len_second(audio, sr):
    return 1.0 * len(audio) / sr
