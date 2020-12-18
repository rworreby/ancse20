import json
import glob
import sys

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
    return j["time"], np.array(j["cell_centers"]), np.array(j["density"]), np.array(j["velocity"]), np.array(j["pressure"])

def plot_snapshot(t, x, u):
    plt.plot(x, u)
    plt.title(f"t = {t:0.6f}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        prefix  = "";
    else:
        prefix = sys.argv[1]

    files = sorted(glob.glob("{}*.json".format(prefix)))

    for fn in files:
        t, x, rho, v, p = load_snapshot(fn)
        plt.plot(x, rho, label="rho")
        plt.plot(x, v, label="v")
        plt.plot(x, p, label="p")
        plt.legend()
        plt.title(f"t = {t:0.6f}")
        plt.show()