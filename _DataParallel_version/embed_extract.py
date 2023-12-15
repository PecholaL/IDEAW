""" Test the trained IDEAW (hard coding, only for specific test)
    * get the marked audio
    * calculate SNR
    * extract the msg and calculate ACC
"""

""" Train on multiple GPUs
    Inference with one GPU
"""

import numpy
import time
import torch
import torch.nn
import tqdm
import warnings

from scipy.io.wavfile import write

from models.ideaw import IDEAW
from data.process import read_resample
from metrics import calc_acc, signal_noise_ratio

warnings.filterwarnings("ignore")

# mini config
msg_bit = 48
lcode_bit = 10

# config paths
config_model_path = "/Users/pecholalee/Coding/IDEAW/models/config.yaml"
ckpt_path = "/Users/pecholalee/Coding/Watermark/ideaw_data/save/stage_I/ideaw.ckpt"

# audio path
audio_path = "/Users/pecholalee/Coding/Watermark/miniAWdata/p225_008.wav"
output_path = "/Users/pecholalee/Coding/Watermark/ideaw_data/output/wmd_audio.wav"

# device
device = "cuda"

if __name__ == "__main__":
    # build model and load trained parameters
    ideaw = IDEAW(config_model_path, device)
    ideaw = torch.nn.DataParallel(ideaw).to(device)
    print("[IDEAW]model built")

    ideaw.load_state_dict(torch.load(ckpt_path))
    print("[IDEAW]model loaded")

    # generate msg and lcode
    watermark_msg = torch.randint(0, 2, (1, msg_bit), dtype=torch.float32).to(device)
    locate_code = torch.randint(0, 2, (1, lcode_bit), dtype=torch.float32).to(device)

    print(
        """
#############################################
#####               IDEAW               #####
#############################################
          """
    )

    """ EMBEDDING
    """
    chunk_wmd_list = []
    # meta data
    embed_time_cost = 0
    audio_length = 0

    with torch.no_grad():
        ideaw.eval()

        audio, _, _ = read_resample(
            audio_path=audio_path, sr=16000, audio_limit_len=None
        )
        audio_length = len(audio)
        audio = torch.tensor(audio).to(torch.float32).unsqueeze(0).to(device)

        start_time = time.time()
        chunk_size = 16000
        interval_size = 8000
        chunk_num = int(audio_length / (chunk_size + interval_size))
        if chunk_num == 0:
            print("[IDEAW]ERROR, audio is too short.")
        else:
            it = range(chunk_num)
            it = tqdm.tqdm(it, desc="Embedding")
            end_pos = 0
            for i in it:
                start_pos = i * (chunk_size + interval_size)
                wm_end_pos = start_pos + chunk_size
                end_pos = start_pos + chunk_size + interval_size
                chunk = audio[:, start_pos:wm_end_pos]
                chunk_rest = audio[:, wm_end_pos:end_pos]

                # embed msg/lcode
                audio_wmd1, audio_wmd1_stft = ideaw.module.embed_msg(
                    chunk, watermark_msg
                )
                audio_wmd2, _ = ideaw.module.embed_lcode(audio_wmd1, locate_code)

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

    # calculate SNR
    SNR = signal_noise_ratio(audio.squeeze().cpu().numpy(), audio_wmd)

    print(f"[IDEAW]audio length: {audio_length}")
    print(f"[IDEAW]embedding time cost: {embed_time_cost}s")
    print(f"[IDEAW]SNR: {SNR:4f}")

    """ EXTRACTION
    """
    # extract and compute acc (w/o random clip)
    # meta data
    extract_time_cost = 0
    acc_msg_list = []
    acc_lcode_list = []
    with torch.no_grad():
        ideaw.eval()

        audio, _, _ = read_resample(
            audio_path=output_path, sr=16000, audio_limit_len=None
        )
        audio_length = len(audio)
        audio = torch.tensor(audio).to(torch.float32).unsqueeze(0)

        start_time = time.time()
        chunk_size = 16000
        interval_size = 8000
        chunk_num = int(audio_length / (chunk_size + interval_size))

        it = range(chunk_num)
        it = tqdm.tqdm(it, desc="Extracting")
        for i in it:
            start_pos = i * (chunk_size + interval_size)
            chunk = audio[:, start_pos : start_pos + chunk_size]

            # extract lcode/msg
            mid_stft, extract_lcode = ideaw.module.extract_lcode(chunk)
            extract_msg = ideaw.module.extract_msg(mid_stft)

            # compute acc
            acc_lcode = calc_acc(extract_lcode, locate_code, 0.5)
            acc_msg = calc_acc(extract_msg, watermark_msg, 0.5)

            acc_lcode_list.append(acc_lcode.cpu())
            acc_msg_list.append(acc_msg.cpu())

        end_time = time.time()
        extract_time_cost = end_time - start_time

    acc_lcode_all = numpy.array(acc_lcode_list)
    acc_msg_all = numpy.array(acc_msg_list)

    print(f"[IDEAW]lcode/msg acc: {acc_lcode_all.mean():4f}/{acc_msg_all.mean():4f}")
    print(f"[IDEAW]extraction time cost: {extract_time_cost}s")
