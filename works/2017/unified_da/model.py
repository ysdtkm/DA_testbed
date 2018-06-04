#!/usr/bin/env python

import numpy as np
from const import N_MODEL, P_OBS, pos_obs, OERR

class Model:
    def __init__(self):
        pass

    def rk4(self, x, dt):
        assert isinstance(x, np.ndarray)
        assert isinstance(dt, float)
        assert dt > 0.0
        x0 = np.copy(x)
        k1 = self.tendency(x0)
        x2 = x0 + k1 * dt / 2.0
        k2 = self.tendency(x2)
        x3 = x0 + k2 * dt / 2.0
        k3 = self.tendency(x3)
        x4 = x0 + k3 * dt
        k4 = self.tendency(x4)
        return x0 + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * dt / 6.0

    def tendency(self, x_in):
        assert isinstance(x_in, np.ndarray)
        f = 8.0
        dx = (np.roll(x_in, -1) - np.roll(x_in, 2)) * np.roll(x_in, 1) - x_in + f
        return dx


def finite_time_tangent_using_nonlinear(x0: np.ndarray, dt: float, iw: int) -> np.ndarray:
    """
    Return tangent linear matrix, calculated numerically using the NL model

    :param x0: [N_MODEL] state vector at the beginning of the window
    :param dt: length of a time step
    :param iw: integration window (time in steps)
    :return:   [N_MODEL,N_MODEL]
    """

    m_finite = np.identity(N_MODEL)
    eps = 1.0e-9
    for j in range(N_MODEL):
        xctl = np.copy(x0)
        xptb = np.copy(x0)
        xptb[j] += eps
        for i in range(iw):
            xctl = rk4(xctl, dt)
            xptb = rk4(xptb, dt)
        m_finite[:, j] = (xptb[:] - xctl[:]) / eps
    return m_finite


def test_model():
    x = np.random.randn(N_MODEL)
    dt = 0.01
    for i in range(100):
        x = rk4(x, dt)
    print(x)


def dist(i1: int, i2: int) -> int:
    d1 = abs(i1 - i2)
    dist = min(d1, N_MODEL - d1)
    return dist


def obs_within(i: int, l_loc: int) -> list:
    # O(p). This can be modified to O(1)

    list_j = []
    for j in range(P_OBS):
        if dist(i, pos_obs(j)) <= l_loc:
            list_j.append(j)
    return list_j


def getr() -> np.ndarray:
    """
    note: Non-diagonal element in R is ignored in main.exec_obs()
    :return r: [P_OBS, P_OBS]
    """
    r = np.identity(P_OBS) * OERR ** 2
    return r


def geth() -> np.ndarray:
    """
    :return h: [P_OBS, N_MODEL]
    """
    h = np.zeros((P_OBS, N_MODEL))
    for j in range(P_OBS):
        i = pos_obs(j)
        h[j, i] = 1.0
    return h


if __name__ == "__main__":
    test_model()
