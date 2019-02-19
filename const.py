#!/usr/bin/env python

from functools import lru_cache
import numpy as np

N_MODEL = 3  # dimension of model variable
P_OBS = N_MODEL  # dimension of observation

DT = 0.01
TMAX = 10
STEPS = int(TMAX / DT)
STEP_FREE = STEPS // 4
FCST_LT = 0

OERR = 2.0 ** 0.5  # ttk
FERR_INI = 10.0
SEED = 10 ** 6 + 3

EXPLIST = [
    dict(name="letkf", method="letkf", rho=1.05**1, k_ens=21, l_loc=1, aint=25),
]

def pos_obs(j):
    # return model grid i of j-th observation
    assert isinstance(j, int)
    assert P_OBS > 0
    return j * (N_MODEL // P_OBS)

@lru_cache(maxsize=1)
def static_b():
    b = np.load("blob/mean_b_cov.npy")
    assert b.shape == (N_MODEL, N_MODEL)
    eigs = np.linalg.eigvalsh(b)
    assert np.all(eigs > 0.0)
    b /= np.max(eigs)
    return b

