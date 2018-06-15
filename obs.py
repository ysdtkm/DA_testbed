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

    def __str__(self):
        return f"val: {self.val}\ntype: {self.type}\nposition: {self.position}\nsigma_r: {self.sigma_r}"

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
    p_obs = len(obs)
    assert fcst.shape == (aint, k_ens, N_MODEL)
    yb_raw = np.empty((p_obs, k_ens))
    for j, o in enumerate(obs):
        assert t_end - aint < o.time <= t_end
        it = o.time - t_end + aint - 1
        assert 0 <= it < aint
        for i in range(k_ens):
            yb_raw[j, i] = generate_single_obs(fcst[it, i, :], o.position, 0.0, o.time).val
    return yb_raw

def generate_single_obs(x, position, sigma_r, time):
    # model must be a model of Dirren and Hakim
    assert x.shape == (N_MODEL,)
    assert isinstance(position, int)
    assert 0 <= position < N_MODEL // 2
    model_state = x[position] + x[position + N_MODEL // 2]
    obs = Scaler_obs(model_state + np.random.randn() * sigma_r, "Dirren", time, position, sigma_r)
    return obs

def test_single_obs():
    x = np.random.randn(N_MODEL)
    position = np.random.randint(N_MODEL // 2)
    sigma_r = 1.0
    time = 10
    obs = generate_single_obs(x, position, sigma_r, time)
    print(obs)

if __name__ == "__main__":
    pass
