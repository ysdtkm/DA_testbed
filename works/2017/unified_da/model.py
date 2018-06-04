#!/usr/bin/env python

import numpy as np
from const import N_MODEL

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

    def test_model(self):
        x = np.random.randn(N_MODEL)
        dt = 0.01
        for i in range(100):
            x = self.rk4(x, dt)
        print(x)

if __name__ == "__main__":
    Model().test_model()
