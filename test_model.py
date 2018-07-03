#!/usr/bin/env python

import unittest
import numpy as np
from const import N_MODEL
from model import Model

class TestModel(unittest.TestCase):
    def test_model(self):
        x = np.random.randn(N_MODEL)
        dt = 0.01
        m = Model()
        for i in range(100):
            x = m.rk4(x, dt)
        self.assertEqual(x.shape, (N_MODEL,))

    def test_negative_dt(self):
        x = np.random.randn(N_MODEL)
        m = Model()
        with self.assertRaises(AssertionError):
            x = m.rk4(x, dt=-0.01)

    def test_convergence(self):
        def evolve(x_init, t, dt):
            m = Model()
            x = np.copy(x_init)
            for i in range(int(t / dt)):
                x = m.rk4(x, dt)
            return x

        t = 1.0
        dt_true = 0.001
        dt_list = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
        np.random.seed(0)
        x = np.random.randn(N_MODEL)
        rmse_all = []
        for dt in dt_list:
            x_true = evolve(x, t, dt_true)
            x_test = evolve(x, t, dt)
            rmse_all.append(np.linalg.norm(x_test - x_true))
        self.assertEqual(rmse_all, sorted(rmse_all))

