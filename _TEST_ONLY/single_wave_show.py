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
    y = y[:64000]
    x = numpy.arange(0, len(y))

    plt.figure(figsize=(12, 3))
    plt.plot(x, y, color="burlywood")
    """ mark the wmd audio with diff. color
    """
    # chunk = 16000
    # inter = 8000
    # start = 0
    # wmd_color = "thistle"
    # host_color = "tan"
    # while start < len(y):
    #     if start + inter < len(y):
    #         plt.plot(
    #             x[start : start + inter], y[start : start + inter], color=host_color
    #         )
    #     else:
    #         plt.plot(x[start : len(y)], y[start : len(y)], color=host_color)
    #     start = start + inter
    #     if start + chunk < len(y):
    #         plt.plot(
    #             x[start : start + chunk], y[start : start + chunk], color=wmd_color
    #         )
    #     else:
    #         plt.plot(x[start : len(y)], y[start : len(y)], color=host_color)
    #     start = start + chunk
    # plt.plot(x[11000:12000], y[11000:12000], color="burlywood")

    plt.axis("off")

    if args.save == True:
        if args.name:
            plt.savefig(args.name + ".pdf")
        else:
            plt.savefig("tmp.pdf")

    plt.show()
