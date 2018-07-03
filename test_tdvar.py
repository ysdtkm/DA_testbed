#!/usr/bin/env python3

import unittest
import numpy as np
from const import N_MODEL
from obs import Scaler_obs
from tdvar import tdvar, tdvar_analytic

class TestTdvar(unittest.TestCase):
    def test_equal_analytic(self):
        np.random.seed(0)
        sigma_b = 1.0
        x = np.random.randn(N_MODEL) * sigma_b
        t = 0
        olist = self.generate_sample_obs_list(10, t)
        anl_tdvar_minimization = tdvar(x, olist, sigma_b, t)
        anl_tdvar_analytic = tdvar_analytic(x, olist, sigma_b, t)
        self.assertLess(np.max(np.abs(anl_tdvar_minimization - anl_tdvar_analytic)), 0.1 ** 5)

    @classmethod
    def generate_sample_obs_list(cls, num, t):
        np.random.seed(0)
        oerr = 1.0
        li = []
        for i in range(num):
            oval = np.random.randn()
            pos = np.random.randint(0, N_MODEL)
            li.append(Scaler_obs(oval, "", t, pos, oerr))
        return li
