
import pickle
import yaml
from random import randint
from scipy.io.wavfile import write
from utils import *
from dataset import AWdataset, get_data_loader, infinite_iter

pickle_path = '../../Watermark/miniAWdata_pickle/stft.pkl'

# Load pickle dataset
with open(pickle_path, 'rb') as f:
    data = pickle.load(f)
    print(f'[test]load {len(data)} data items')

# Load data processing configure
with open('dataConfig.yaml', 'rb') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

sample_rate = config['sample_rate']
n_fft = config['n_fft']
hop_len = config['hop_len']
win_len = config['win_len']

# Check whether ISTFT can restore the wav
i = randint(0, len(data))
stft_test = data[12]
# print(f'{stft_test.shape}')
istft_test = istft(stft_test, hop_len, win_len, n_fft)
write('./tmp/istft_test.wav', sample_rate, istft_test)
print('ISTFT completed')

# Check DataLoader
dataset = AWdataset(pickle_path=pickle_path)
dataLoader = get_data_loader(dataset=dataset, 
                             batch_size=16, 
                             shuffle=False, 
                             num_workers=0, 
                             drop_last=False)
train_iter = infinite_iter(dataLoader)
print(f'[test]built infinite dataloader')
d = next(train_iter)
print(f'[test]data shape: {d.shape}')

