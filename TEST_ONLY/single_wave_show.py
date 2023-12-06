import librosa
import matplotlib.pyplot as plt
import numpy

from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--path", required=True)
    parser.add_argument("-s", "--save", action="store_true", required=False)
    parser.add_argument("-n", "--name", required=False)

    args = parser.parse_args()

    y, sr = librosa.load(args.path, sr=None, mono=True)
    x = numpy.arange(0, len(y))

    plt.figure(figsize=(15, 3))
    plt.plot(x, y, color="burlywood")

    if args.save == True:
        if args.name:
            plt.savefig(args.name + ".pdf")
        else:
            plt.savefig("tmp.pdf")

    plt.show()
