#!/usr/bin/env python3

import IPython
import matplotlib.pyplot as plt
import numpy as np

def main():
    true = np.load("data/true.npy")
    letkf = np.mean(np.load("data/letkf_1_cycle.npy"), axis=1)
    assert true.shape == letkf.shape

    d0 = letkf - true
    dp1 = np.roll(letkf, 1, axis=0) - true
    dm1 = np.roll(letkf, -1, axis=0) - true

    for name in ["d0", "dp1", "dm1"]:
        data = eval(name)
        cm = plt.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1)
        plt.rcParams["font.size"] = 14
        plt.colorbar(cm)
        plt.savefig(f"{name}.pdf", bbox_inches="tight")
        plt.close()

main()

