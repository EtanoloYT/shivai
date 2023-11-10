import numpy as np
import matplotlib.pyplot as plt
import os
import sys

INPUTS_FOLDER = "outputs/"


def view_song(song):
    print("Viewing song: ", song)
    song = np.load(song)
    for i in range(100):
        print(song)


def plot_song(song):
    song = np.load(song)
    song = song[::]
    plt.figure(figsize=(20, 5))
    plt.plot(song)
    plt.show()


def main(argv):
    if argv[1] == "view":
        """
        argv1: view
        argv2: input file
        """
        view_song(argv[2])
    elif argv[1] == "plot":
        """
        argv1: plot
        argv2: input file
        """
        plot_song(argv[2])


if __name__ == "__main__":
    main(sys.argv)
