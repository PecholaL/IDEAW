""" Test the robustness of the trained IDEAW
    * get the watermarked audio via IDEAW
    * apply the preset attack to the watermarked audio
    * extract the msg and compute ACC
"""

import numpy
import torch
import warnings

from scipy.io.wavfile import write

from data.process import read_resample
from models.ideaw import IDEAW
from models.attackLayer import AttackLayer
from metrics import calc_acc

warnings.filterwarnings("ignore")

# mini config
msg_bit = 16
lcode_bit = 10
att_index = 1

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

    with torch.no_grad():
        ideaw.eval()

        audio, _, _ = read_resample(
            audio_path=audio_path, sr=16000, audio_limit_len=None
        )
        audio_length = len(audio)
        audio = torch.tensor(audio).to(torch.float32).unsqueeze(0).to(device)

        chunk_size = 16000
        interval_size = 8000
        chunk_num = int(audio_length / (chunk_size + interval_size))
        if chunk_num == 0:
            print("[IDEAW]ERROR, audio is too short.")
        else:
            it = range(chunk_num)
            end_pos = 0
            for i in it:
                start_pos = i * (chunk_size + interval_size)
                wm_end_pos = start_pos + chunk_size
                end_pos = start_pos + chunk_size + interval_size
                chunk = audio[:, start_pos:wm_end_pos]
                chunk_rest = audio[:, wm_end_pos:end_pos]

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

    # get the watermarked audio
    audio_wmd = numpy.concatenate(chunk_wmd_list)
    audio_wmd = torch.from_numpy(audio_wmd)
    print(f"[IDEAW]audio length: {audio_length}")

    """ ATTACK
    """
    attacker = AttackLayer(config_model_path, device)
    if att_index == 1:
        att_audio = attacker.gaussianNoise(audio)
    elif att_index == 2:
        att_audio = attacker.bandpass(audio)
    elif att_index == 5:
        att_audio = attacker.erase(audio)
    # elif att_index == 4:
    elif att_index == 6:
        att_audio = attacker.resample(audio)
    elif att_index == 7:
        att_audio = attacker.ampMdf(audio)
    elif att_index == 3:
        att_audio = attacker.mp3compress(audio)
    else:  # elif att_index == 8:
        att_audio = attacker.timeStretch(audio)
    att_audio = att_audio.cpu().numpy()
    write(output_path, 16000, att_audio)

    """ EXTRACTION
    """
    acc_msg_list = []
    acc_lcode_list = []
    with torch.no_grad():
        ideaw.eval()

        audio, _, _ = read_resample(
            audio_path=output_path, sr=16000, audio_limit_len=None
        )
        audio_length = len(audio)
        audio = torch.tensor(audio).to(torch.float32).unsqueeze(0)

        chunk_size = 16000
        interval_size = 8000
        chunk_num = int(audio_length / (chunk_size + interval_size))

        it = range(chunk_num)
        for i in it:
            start_pos = i * (chunk_size + interval_size)
            chunk = audio[:, start_pos : start_pos + chunk_size]

            # extract lcode/msg
            mid_stft, extract_lcode = ideaw.extract_lcode(chunk)
            extract_msg = ideaw.extract_msg(mid_stft)

            # compute acc
            acc_lcode = calc_acc(extract_lcode, locate_code, 0.5)
            acc_msg = calc_acc(extract_msg, watermark_msg, 0.5)

            acc_lcode_list.append(acc_lcode.cpu())
            acc_msg_list.append(acc_msg.cpu())

    acc_lcode_all = numpy.array(acc_lcode_list)
    acc_msg_all = numpy.array(acc_msg_list)

    print(f"[IDEAW]lcode/msg acc: {acc_lcode_all.mean():4f}/{acc_msg_all.mean():4f}")
