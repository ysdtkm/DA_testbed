#!/usr/bin/env python

import numpy as np
from const import N_MODEL, P_OBS, OERR

class Scaler_obs:
    def __init__(self, val, type, time, position, sigma_r):
        assert isinstance(val, float)
        assert isinstance(type, str)
        assert isinstance(time, (int, float))
        assert isinstance(position, (int, float))
        assert isinstance(sigma_r, float)
        self.val = val
        self.type = type
        self.time = time
        self.position = position
        self.sigma_r = sigma_r

def dist(i1, i2):
    assert isinstance(i1, int)
    assert isinstance(i2, int)
    d1 = abs(i1 - i2)
    dist = min(d1, N_MODEL - d1)
    return dist

def obs_within(i, l_loc, obs):
    # O(p). This can be modified to O(1)
    assert isinstance(i, int)
    assert isinstance(l_loc, int)
    list_j = []
    for j in range(P_OBS):
        if dist(i, obs[j].position) <= l_loc:
            list_j.append(j)
    return list_j

def geth(obs):
    assert obs.shape == (P_OBS,)
    assert isinstance(obs[0], Scaler_obs)
    h = np.zeros((P_OBS, N_MODEL))
    for j in range(P_OBS):
        i = obs[j].position
        assert isinstance(i, int)
        h[j, i] = 1.0
    return h

def getr(obs):
    assert obs.shape == (P_OBS,)
    assert isinstance(obs[0], Scaler_obs)
    r = np.zeros((P_OBS, P_OBS))
    for j in range(P_OBS):
        r[j, j] = obs[j].sigma_r ** 2
    return r
