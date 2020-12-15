import json
import glob

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import LogLocator
from matplotlib import rc
from matplotlib import rcParams
font = {'family' : 'Dejavu Sans',
        'weight' : 'normal',
        'size'   : 22}
rc('font', **font)
rcParams['lines.linewidth'] = 4
rcParams['lines.markersize'] = 12
rcParams['markers.fillstyle'] = 'none'

def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def load_snapshot(filename):
    j = load_json(filename)
    return j["time"], np.array(j["cell_centers"]), np.array(j["height"]), np.array(j["velocity"])

def plot_snapshot(t, x, u, label):
    plt.plot(x, u, label=label)
    plt.title(f"t = {t:0.6f}")

if __name__ == "__main__":

    files = sorted(glob.glob("output/swe_riemann-*.json"))

    for fn in files:
        t, x, h, v = load_snapshot(fn)
        plot_snapshot(t, x, h, 'h')
        plot_snapshot(t, x, h*v, 'm')
        plt.legend()
        plt.show()
