#!/usr/bin/env python

import numpy as np
import numba
from const import N_MODEL

class Model:
    def __init__(self):
        pass

    def rk4(self, x, dt):
        return step(x, dt)

    def finite_time_tangent_using_nonlinear(self, x0, dt, iw):
        assert isinstance(x0, np.ndarray)
        assert isinstance(dt, float)
        assert dt > 0.0
        assert isinstance(iw, int)
        assert iw > 0
        m_finite = np.identity(N_MODEL)
        eps = 1.0e-9
        for j in range(N_MODEL):
            xctl = np.copy(x0)
            xptb = np.copy(x0)
            xptb[j] += eps
            for i in range(iw):
                xctl = self.rk4(xctl, dt)
                xptb = self.rk4(xptb, dt)
            m_finite[:, j] = (xptb[:] - xctl[:]) / eps
        return m_finite

    @classmethod
    def sample_state(cls):
        tmp = np.random.get_state()
        np.random.seed(0)
        x = np.random.randn(N_MODEL)
        np.random.set_state(tmp)
        for i in range(1000):
            x = step(x, dt=0.05)
        return x

@numba.jit("f8[:](f8[:])", nopython=True)
def tendency(x_in):
    f = 8.0
    n = len(x_in)
    dx = np.empty(n)
    for i in range(n):
        dx[i] = (x_in[(i + 1) % n] - x_in[(i - 2) % n]) * x_in[(i - 1) % n] - x_in[i] + f
    return dx

@numba.jit("f8[:](f8[:], f8)", nopython=True)
def step(x, dt):
    assert dt > 0.0
    x0 = np.copy(x)
    k1 = tendency(x0)
    x2 = x0 + k1 * dt / 2.0
    k2 = tendency(x2)
    x3 = x0 + k2 * dt / 2.0
    k3 = tendency(x3)
    x4 = x0 + k3 * dt
    k4 = tendency(x4)
    return x0 + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * dt / 6.0

if __name__ == "__main__":
    print(Model.sample_state())
