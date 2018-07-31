#!/usr/bin/env python3

from IPython import embed
import unittest
import warnings
import numpy as np
from const import N_MODEL, DT, OERR, FERR_INI, P_OBS, pos_obs
from da_system import Da_system
from obs import Scaler_obs, generate_single_obs
from model import model_step

class TestDaSystem(unittest.TestCase):
    def test_single_window(self):
        for smoother in [True, False]:
            if smoother:
                nt = 3
            else:
                nt = 1
            k_ens = 19

            np.random.seed(0)
            true = np.empty((nt, N_MODEL))
            true[0, :] = self.get_sample_model_state()
            for t in range(1, nt):
                true[t, :] = model_step(true[t - 1, :], DT)
            xb = np.empty((nt, k_ens, N_MODEL))
            for i in range(k_ens):
                xb[0, i, :] = self.get_sample_model_state()
                for t in range(1, nt):
                    xb[t, i, :] = model_step(xb[t - 1, i, :], DT)
            das = Da_system({"method": "letkf", "rho": 1.0, "k_ens": k_ens, "l_loc": 999999999, "amp_b": None})

            olist = []
            dt_obs = 1
            for i in range(P_OBS):
                k = pos_obs(i)
                o = generate_single_obs(true[nt - dt_obs:nt, :], k, 5.0, nt - 1, dt_obs)
                olist.append(o)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",
                    message="numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88")
                xa = das.analyze_one_window(xb[:, :, :], [olist], nt - 1, nt, smoother)
            if smoother:
                assert xa.shape == (nt, k_ens, N_MODEL)
                rmse_xb = np.linalg.norm(np.mean(xb, axis=1) - true)
                rmse_xa = np.linalg.norm(np.mean(xa, axis=1) - true)
                self.assertLess(rmse_xa, rmse_xb)
            else:
                assert xa.shape == (k_ens, N_MODEL)
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
