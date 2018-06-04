#!/usr/bin/env python

import numpy as np
from const import N_MODEL, P_OBS, pos_obs, OERR

class Scaler_obs:
    def __init__(self, val, type, position, sigma_r):
        assert isinstance(val, float)
        assert isinstance(type, str)
        assert isinstance(position, (int, float))
        assert isinstance(sigma_r, float)
        self.val = val
        self.type = type
        self.position = position
        self.sigma_r = sigma_r

def dist(i1, i2):
    assert isinstance(i1, int)
    assert isinstance(i2, int)
    d1 = abs(i1 - i2)
    dist = min(d1, N_MODEL - d1)
    return dist

def obs_within(i, l_loc):
    # O(p). This can be modified to O(1)
    assert isinstance(i, int)
    assert isinstance(l_loc, int)
    list_j = []
    for j in range(P_OBS):
        if dist(i, pos_obs(j)) <= l_loc:
            list_j.append(j)
    return list_j

def getr():
    # note: Non-diagonal element in R is ignored in main.exec_obs()
    r = np.identity(P_OBS) * OERR ** 2
    return r

def geth():
    h = np.zeros((P_OBS, N_MODEL))
    for j in range(P_OBS):
        i = pos_obs(j)
        h[j, i] = 1.0
    return h
