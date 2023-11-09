
import pickle
from scipy.io.wavfile import write
from utils import *

with open('../../Watermark/miniAWdata_pickle/stft.pkl', 'rb') as f:
    data = pickle.load(f)
    print(len(data))

stft_0 = data[1]
istft_0 = istft(stft_0)
write('istft_0.wav', 16000, istft_0)
