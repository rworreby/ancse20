import sys
import json
import glob
import time

import numpy as np
import matplotlib.pyplot as plt


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def load_snapshot(filename):
    j = load_json(filename)
    return j["time"], np.array(j["cell_centers"]), np.array(j["data"])


def plot_snapshot(t, x, u):
    plt.plot(x, u, marker="s", linestyle="-")
    plt.title(f"t = {t:0.3f}")


if __name__ == "__main__":
    stem = sys.argv[1]

    files = sorted(glob.glob(f"{stem}*.json"))

    plt.ion()
    plt.show()

    for fn in files:
        t, x, u = load_snapshot(fn)
        plot_snapshot(t, x, u)
        plt.pause(0.01)
        plt.clf()
