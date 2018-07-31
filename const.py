#!/usr/bin/env python

import numpy as np

N_MODEL = 40  # dimension of model variable
P_OBS = N_MODEL // 2 # dimension of observation

DT = 0.05
TMAX = 2
STEPS = int(TMAX / DT)
STEP_FREE = STEPS // 4
FCST_LT = 0

OERR = 0.5 ** 0.5
FERR_INI = 3.0
SEED = 10 ** 6 + 3

EXPLIST = [
    # dict(name="fdvar-0_5", method="fdvar", rho=None, k_ens=1, l_loc=None, amp_b=0.5, aint=2),
    dict(name="letkf_1", method="letkf", rho=1.05, k_ens=100, l_loc=12, amp_b=None, aint=1),
    # dict(name="letkf_3", method="letkf", rho=1.2**10, k_ens=21, l_loc=10, amp_b=None, aint=10),
    # dict(name="letkf_4", method="letkf", rho=1.2**30, k_ens=21, l_loc=10, amp_b=None, aint=30),
    # dict(name="letkf_5", method="letkf", rho=1.2**100, k_ens=21, l_loc=10, amp_b=None, aint=100),
]

def pos_obs(j):
    assert isinstance(j, int)
    assert P_OBS > 0
    return j * (N_MODEL // 2 // P_OBS)


