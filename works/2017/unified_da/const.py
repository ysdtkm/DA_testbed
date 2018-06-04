#!/usr/bin/env python

import numpy as np

N_MODEL = 20  # dimension of model variable
P_OBS = N_MODEL // 2  # dimension of observation

DT = 0.05
TMAX = 10
STEPS = int(TMAX / DT)
STEP_FREE = STEPS // 4
FCST_LT = 0

OERR = 0.5
FERR_INI = 10.0
AINT = 2
SEED = 10 ** 8 + 7

EXPLIST = [
    dict(name="letkf", method="letkf", rho=1.2, k_ens=21, l_loc=10),
]

def pos_obs(j):
    # return model grid i of j-th observation
    assert isinstance(j, int)
    assert P_OBS > 0
    return j * (N_MODEL // P_OBS)


