#!/usr/bin/env python3

import IPython
import matplotlib.pyplot as plt
import numpy as np

def obval(obs):
    return obs.val

def main():
    true = np.load("data/true.npy")
    obs = np.load("data/obs.npy")
    np_obval = np.vectorize(obval)
    obs = np_obval(obs)
    letkf = np.mean(np.load("data/letkf_1_cycle.npy"), axis=1)
    assert true.shape == letkf.shape

    d0 = letkf - true
    dp1 = np.roll(letkf, 1, axis=0) - true
    dm1 = np.roll(letkf, -1, axis=0) - true

    true2 = true[:, :20] + true[:, 20:]
    od0 = obs - true2
    odp1 = np.roll(obs, 1, axis=0) - true2
    odm1 = np.roll(obs, -1, axis=0) - true2

    for name in ["d0", "dp1", "dm1", "od0", "odp1", "odm1"]:
        cmax = 1 if name in ["d0", "dp1", "dm1"] else 5
        data = eval(name)
        cm = plt.imshow(data, cmap="RdBu_r", vmin=-cmax, vmax=cmax)
        plt.rcParams["font.size"] = 14
        plt.colorbar(cm)
        plt.savefig(f"{name}.pdf", bbox_inches="tight")
        plt.close()

main()

