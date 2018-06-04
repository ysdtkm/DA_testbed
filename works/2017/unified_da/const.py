#!/usr/bin/env python

import numpy as np

N_MODEL = 20  # dimension of model variable
P_OBS = N_MODEL // 2  # dimension of observation

DT = 0.05
TMAX = 3
STEPS = int(TMAX / DT)
STEP_FREE = STEPS // 4
FCST_LT = 0

OERR = 0.01
FERR_INI = 10.0
AINT = 1
SEED = 10 ** 8 + 7

EXPLIST = [
    dict(name="fdvar-0_3", method="fdvar", rho=None, k_ens=1, l_loc=None, amp_b=0.3),
    dict(name="letkf", method="letkf", rho=1.2, k_ens=21, l_loc=10, amp_b=None),
]

def pos_obs(j):
    # return model grid i of j-th observation
    assert isinstance(j, int)
    assert P_OBS > 0
    return j * (N_MODEL // P_OBS)


