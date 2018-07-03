#!/usr/bin/env python

from functools import lru_cache
import numpy as np

N_MODEL = 20  # dimension of model variable
P_OBS = N_MODEL  # dimension of observation

DT = 0.05
TMAX = 10
STEPS = int(TMAX / DT)
STEP_FREE = STEPS // 4
FCST_LT = 0

OERR = 5.0
FERR_INI = 10.0
SEED = 10 ** 6 + 3

EXPLIST = [
    dict(name="fdvar", method="fdvar", rho=None, k_ens=1, l_loc=None, amp_b=2.0, aint=4),
    dict(name="tdvar", method="tdvar", rho=None, k_ens=1, amp_b=2.0, l_loc=None, aint=1),
    dict(name="letkf_1", method="letkf", rho=1.2**1, k_ens=21, l_loc=10, amp_b=None, aint=1),
    # dict(name="letkf_2", method="letkf", rho=1.2**3, k_ens=21, l_loc=10, amp_b=None, aint=3),
    # dict(name="letkf_3", method="letkf", rho=1.2**10, k_ens=21, l_loc=10, amp_b=None, aint=10),
    # dict(name="letkf_4", method="letkf", rho=1.2**30, k_ens=21, l_loc=10, amp_b=None, aint=30),
    # dict(name="letkf_5", method="letkf", rho=1.2**100, k_ens=21, l_loc=10, amp_b=None, aint=100),
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

