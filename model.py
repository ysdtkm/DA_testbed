#!/usr/bin/env python

from numba import jit
import numpy as np
from const import N_MODEL

ALPHA_HF = 0.5
ALPHA_LF = 75.0

@jit("f8[:](f8[:], f8)", nopython=True)
def tendency(x_in, alpha):
    f = 8.0
    n = len(x_in)
    dx = np.empty(n)
    for i in range(n):
        dx[i] = (x_in[(i + 1) % n] - x_in[(i - 2) % n]) * x_in[(i - 1) % n] - x_in[i] + f
    return dx / alpha

@jit("f8[:](f8[:], f8, f8)", nopython=True)
def rk4(x, dt, alpha):
    x0 = np.copy(x)
    k1 = tendency(x0, alpha)
    x2 = x0 + k1 * dt / 2.0
    k2 = tendency(x2, alpha)
    x3 = x0 + k2 * dt / 2.0
    k3 = tendency(x3, alpha)
    x4 = x0 + k3 * dt
    k4 = tendency(x4, alpha)
    x5 = x0 + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * dt / 6.0
    return x5

@jit("f8[:](f8[:], f8)", nopython=True)
def model_step(x, dt):
    assert x.shape == (N_MODEL,)
    x[:N_MODEL // 2] = rk4(x[:N_MODEL // 2], dt, ALPHA_HF)
    x[N_MODEL // 2:] = rk4(x[N_MODEL // 2:], dt, ALPHA_LF)
    return x

def test_model():
    import matplotlib.pyplot as plt
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
    plt.savefig("img.pdf", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    test_model()
