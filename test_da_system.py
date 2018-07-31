#!/usr/bin/env python3

import unittest
import warnings
import numpy as np
from const import N_MODEL, DT, OERR, FERR_INI, P_OBS, pos_obs
from da_system import Da_system
from obs import Scaler_obs, generate_single_obs
from model import model_step

class TestDaSystem(unittest.TestCase):
    def test_single_filter(self):
        np.random.seed(0)
        true = self.get_sample_model_state()
        k_ens = 19
        xb = np.empty((k_ens, N_MODEL))
        for i in range(k_ens):
            xb[i, :] = self.get_sample_model_state()
        das = Da_system({"method": "letkf", "rho": 1.0, "k_ens": k_ens, "l_loc": 999999999, "amp_b": None})

        olist = []
        for i in range(P_OBS):
            k = pos_obs(i)
            o = generate_single_obs(true, k, OERR, 0)
            olist.append(o)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                message="numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88")
            xa = das.analyze_one_window(xb[None, :, :], [olist], 0, 1, False)
        rmse_xb = np.linalg.norm(np.mean(xb, axis=0) - true)
        rmse_xa = np.linalg.norm(np.mean(xa, axis=0) - true)
        self.assertLess(rmse_xa, rmse_xb)

    @classmethod
    def get_sample_model_state(cls):
        x = np.random.randn(N_MODEL) * FERR_INI
        for i in range(1000):
            x = model_step(x, DT)
        return x

if __name__ == "__main__":
    unittest.main()
