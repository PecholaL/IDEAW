""" Test the trained IDEAW (hard coding, only for specific test)
    * get the marked audio
    * extract the msg and compute ACC
    * ...
"""

import numpy
import time
import torch
import tqdm
from scipy.io.wavfile import write

from models.ideaw import IDEAW
from data.process import read_resample
from metrics import calc_acc

# mini config
msg_bit = 16
lcode_bit = 10

# config paths
config_model_path = "/Users/pecholalee/Coding/IDEAW/models/config.yaml"
ckpt_path = "/Users/pecholalee/Coding/Watermark/ideaw_data/save/stage_I/ideaw.ckpt"

# audio path
audio_path = "/Users/pecholalee/Coding/Watermark/miniAWdata/p225_008.wav"
output_path = "/Users/pecholalee/Coding/Watermark/ideaw_data/output/wmd_audio.wav"

# device
device = "cpu"

if __name__ == "__main__":
    # build model and load trained parameters
    ideaw = IDEAW(config_model_path, device)
    print("[IDEAW]model built")

    ideaw.load_state_dict(torch.load(ckpt_path))
    print("[IDEAW]model loaded")

    # generate msg and lcode
    watermark_msg = torch.randint(0, 2, (1, msg_bit), dtype=torch.float32)
    locate_code = torch.randint(0, 2, (1, lcode_bit), dtype=torch.float32)

    # EMBEDDING
    chunk_wmd_list = []
    # meta data
    embed_time_cost = 0
    audio_length = 0

    with torch.no_grad():
        ideaw.eval()

        start_time = time.time()
        audio, _, _ = read_resample(
            audio_path=audio_path, sr=16000, audio_limit_len=None
        )
        audio_length = len(audio)
        audio = torch.tensor(audio).to(torch.float32).unsqueeze(0)

        chunk_size = int(16000 * (1 + 0.5))
        chunk_num = int(audio_length / chunk_size)
        if chunk_num == 0:
            print("[IDEAW]ERROR, audio is too short.")
        else:
            it = range(chunk_num)
            it = tqdm.tqdm(it, desc="Embedding")
            end_pos = 0
            for i in it:
                start_pos = i * chunk_size
                end_wm_pos = int(start_pos + chunk_size / 1.5)
                end_pos = (i + 1) * chunk_size
                chunk = audio[:, start_pos:end_wm_pos]
                chunk_rest = audio[:, end_wm_pos:end_pos]

                # embed msg/lcode
                audio_wmd1, audio_wmd1_stft = ideaw.embed_msg(chunk, watermark_msg)
                audio_wmd2, _ = ideaw.embed_lcode(audio_wmd1, locate_code)

                # concat watermarked chunk
                chunk_wmd = audio_wmd2.squeeze().cpu().numpy()
                chunk_rest = chunk_rest.squeeze().cpu().numpy()

                chunk_wmd_list.append(chunk_wmd)
                chunk_wmd_list.append(chunk_rest)

        audio_rest = audio[:, end_pos:]
        audio_rest = audio_rest.squeeze().cpu().numpy()
        chunk_wmd_list.append(audio_rest)
        end_time = time.time()
        embed_time_cost = end_time - start_time

    # output the watermarked audio
    audio_wmd = numpy.concatenate(chunk_wmd_list)
    write(output_path, 16000, audio_wmd)

    print(
        f"[IDEAW]audio length: {audio_length}, \
        embedding time cost: {embed_time_cost}"
    )

    # EXTRACTION
    # extract and compute acc (w/o random clip)
    # meta data
    extract_time_cost = 0
    acc_msg_list = []
    acc_lcode_list = []
    with torch.no_grad():
        ideaw.eval()

        start_time = time.time()
        audio, _, _ = read_resample(
            audio_path=output_path, sr=16000, audio_limit_len=None
        )
        audio_length = len(audio)
        audio = torch.tensor(audio).to(torch.float32).unsqueeze(0)

        chunk_size = int(16000 * (1 + 0.5))
        chunk_num = int(audio_length / chunk_size)
        print(audio_length)

        it = range(chunk_num)
        it = tqdm.tqdm(it, desc="Extracting")
        for i in it:
            start_pos = i * chunk_size
            chunk = audio[:, start_pos : int(start_pos + chunk_size / 1.5)]

            # extract lcode/msg
            mid_stft, extract_lcode = ideaw.extract_lcode(chunk)
            extract_msg = ideaw.extract_msg(mid_stft)

            extract_lcode = extract_lcode >= 0.5
            extract_msg = extract_msg >= 0.5

            # compute acc
            acc_lcode = calc_acc(extract_lcode, locate_code, 0.5)
            acc_msg = calc_acc(extract_msg, watermark_msg, 0.5)

            acc_lcode_list.append(acc_lcode.cpu())
            acc_msg_list.append(acc_msg.cpu())

        end_time = time.time()
        extract_time_cost = end_time - start_time

    acc_lcode_all = numpy.array(acc_lcode_list)
    acc_msg_all = numpy.array(acc_msg_list)

    print(
        f"[IDEAW]lcode/msg acc: {acc_lcode_all.mean()}/{acc_msg_all.mean()}, \
        extraction time cost: {extract_time_cost}"
    )
