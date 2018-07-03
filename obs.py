#!/usr/bin/env python

import numpy as np
from const import N_MODEL, OERR

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

def getr(obs):
    if len(obs) > 0:
        assert isinstance(obs[0], Scaler_obs)
    r = np.zeros((len(obs), len(obs)))
    for j in range(len(obs)):
        r[j, j] = obs[j].sigma_r ** 2
    return r

def get_background_obs(obs, fcst, t_end, aint):
    assert isinstance(obs, list)
    k_ens = fcst.shape[1]
    assert fcst.shape == (aint, k_ens, N_MODEL)
    yb_raw = np.empty((0, k_ens))
    for j in range(len(obs)):
        assert t_end - aint < obs[j].time <= t_end
        it = obs[j].time - t_end + aint - 1
        assert 0 <= it < aint
        yb_raw = np.concatenate((yb_raw[:, :], fcst[it, :, obs[j].position][np.newaxis, :]), axis=0)
    return yb_raw

def get_h_matrix(obs):
    assert isinstance(obs, list)
    p = len(obs)
    h = np.zeros((p, N_MODEL))
    times = set()
    for i, o in enumerate(obs):
        assert o.type == ""
        times.add(o.time)
        assert 0 <= o.position < N_MODEL
        h[i, o.position] = 1.0
    assert len(times) in [0, 1]  # synoptic
    return h

def get_yo(obs):
    p_obs = len(obs)
    yo = np.empty((p_obs, 1))
    for j in range(p_obs):
        yo[j, 0] = obs[j].val
    return yo

