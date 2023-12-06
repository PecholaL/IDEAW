import librosa
import librosa.display
import matplotlib.pyplot as plt

from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--p1")
    parser.add_argument("--p1")

    args = parser.parse_args()

    y1, sr1 = librosa.load(args.p1, sr=None, mono=True)
    y2, sr2 = librosa.load(args.p2, sr=None, mono=True)

    f, ((ax11, ax12)) = plt.subplots(1, 2, sharex=False, sharey=False)

    ax11.set_title("Host Audio")
    ax11.plot(y1)

    ax12.set_title("Watermarked Audio")
    ax12.plot(y2)

    plt.show()
