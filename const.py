#!/usr/bin/env python

import numpy as np

N_MODEL = 40  # dimension of model variable
P_OBS = N_MODEL // 2 # dimension of observation

DT = 0.01
TMAX = 1
STEPS = int(TMAX / DT)
STEP_FREE = STEPS // 4
FCST_LT = 0

OERR = 0.5 ** 0.5
FERR_INI = 3.0
SEED = 10 ** 6 + 3

EXPLIST = [
    dict(name="letkf_smoother", method="letkf", rho=1.05, k_ens=100, l_loc=12, amp_b=None, aint=5, smoother=True),
    dict(name="letkf_filter", method="letkf", rho=1.05, k_ens=100, l_loc=12, amp_b=None, aint=5, smoother=False),
]

def pos_obs(j):
    assert isinstance(j, int)
    assert P_OBS > 0
    return j * (N_MODEL // 2 // P_OBS)


