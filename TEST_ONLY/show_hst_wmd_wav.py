import librosa
import matplotlib.pyplot as plt
import librosa.display

host_audio_path = "/Users/pecholalee/Coding/Watermark/miniAWdata/p225_003.wav"
watermarked_audio_path = (
    "/Users/pecholalee/Coding/Watermark/ideaw_data/output/att_test.wav"
)

y1, sr1 = librosa.load(host_audio_path, sr=None, mono=True)
y2, sr2 = librosa.load(watermarked_audio_path, sr=None, mono=True)

f, ((ax11, ax12)) = plt.subplots(1, 2, sharex=False, sharey=False)

ax11.set_title("Host Audio")
ax11.plot(y1)


ax12.set_title("Watermarked Audio")
ax12.plot(y2)


plt.show()
