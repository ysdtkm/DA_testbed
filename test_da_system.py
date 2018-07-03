#!/usr/bin/env python

import unittest
import numpy as np
from const import N_MODEL
from da_system import Da_system
from obs import Scaler_obs

class TestDaSystem(unittest.TestCase):
    def test_analysis_closer_to_truth(self):
        np.random.seed(0)
        explist = [
            dict(name="fdvar", method="fdvar", rho=None, k_ens=1, l_loc=None, amp_b=0.5),
            dict(name="tdvar", method="tdvar", rho=None, k_ens=1, l_loc=None, amp_b=2.0),
            dict(name="letkf", method="letkf", rho=1.0, k_ens=21, l_loc=10, amp_b=None),
        ]
        for exp in explist:
            with self.subTest(expname=exp["name"]):
                nrep = 5
                res = []
                for rep in range(nrep):
                    das = Da_system(exp)
                    aint = 1; oerr = 1.0; t_anl = 0
                    truth = np.random.randn(N_MODEL)
                    fcst = np.random.randn(aint, das.k_ens, N_MODEL)
                    j = np.random.randint(N_MODEL)
                    obs = [[Scaler_obs(truth[j], "", t_anl, j, oerr)]]
                    anl = das.analyze_one_window(fcst, obs, t_anl, aint)
                    self.assertEqual(anl.shape, (das.k_ens, N_MODEL))
                    fcst_err = np.mean(fcst[:, :, :], axis=(0, 1)) - truth
                    anl_err = np.mean(anl[:, :], axis=0) - truth
                    is_closer = np.linalg.norm(anl_err) < np.linalg.norm(fcst_err)
                    res.append(is_closer)
                cnt = len([x for x in res if x])
                self.assertGreater(cnt, nrep // 2)

