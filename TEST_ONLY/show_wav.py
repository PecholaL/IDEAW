import librosa
import matplotlib.pyplot as plt
import numpy

audio_path = "/Users/pecholalee/Coding/Watermark/miniAWdata/p225_003.wav"


y, sr = librosa.load(audio_path, sr=None, mono=True)
x = numpy.arange(0, len(y))

plt.figure(figsize=(5, 5))
plt.plot(x, y)
plt.show()
