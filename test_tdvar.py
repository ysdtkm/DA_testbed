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
        anl_tdvar_opt = tdvar(x, olist, sigma_b, t, False)
        anl_tdvar_cvt = tdvar(x, olist, sigma_b, t, True)
        anl_tdvar_anl = tdvar_analytic(x, olist, sigma_b, t)
        opt_minus_anl = np.max(np.abs(anl_tdvar_opt - anl_tdvar_anl))
        cvt_minus_anl = np.max(np.abs(anl_tdvar_cvt - anl_tdvar_anl))
        self.assertLess(opt_minus_anl, 0.1 ** 5)
        self.assertLess(cvt_minus_anl, 0.1 ** 5)
        self.assertGreater(opt_minus_anl, 0.0)
        self.assertGreater(cvt_minus_anl, 0.0)
        self.assertNotEqual(opt_minus_anl, cvt_minus_anl)

    @classmethod
    def generate_sample_obs_list(cls, num, t):
        np.random.seed(0)
        oerr = 1.0
        li = []
        for i in range(num):
            oval = np.random.randn()
            pos = np.random.randint(0, N_MODEL)
            li.append(Scaler_obs(oval, "raw", t, pos, oerr))
        return li
