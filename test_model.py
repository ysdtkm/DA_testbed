#!/usr/bin/env python

import unittest
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import subprocess
from const import N_MODEL
from model import model_step

class TestModel(unittest.TestCase):
    def test_model(self):
        DT = 0.05
        TMAX = 500.0
        plt.rcParams["font.size"] = 14
        np.random.seed(0)
        x0 = np.random.randn(N_MODEL)
        nt = int(TMAX / DT)
        tlist = np.linspace(0, TMAX, nt, endpoint=False)
        x_traj = np.empty((nt, N_MODEL))
        x = x0
        for it in range(nt):
            x = model_step(x, DT)
            x_traj[it, :] = x[:]
        x_traj -= np.mean(x_traj, axis=0)[None, :]
        plt.plot(tlist, x_traj[:, N_MODEL // 2] + x_traj[:, 0], alpha=0.8, label="Sum")
        plt.plot(tlist, x_traj[:, N_MODEL // 2], alpha=0.8, label="LF")
        plt.legend()
        plt.ylabel("x1")
        plt.xlabel("time units")
        subprocess.run("mkdir -p unittest", shell=True, check=True)
        plt.savefig("unittest/img.pdf", bbox_inches="tight")
        plt.close()
