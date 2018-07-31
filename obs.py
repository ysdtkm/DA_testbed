#!/usr/bin/env python

import numpy as np
from const import N_MODEL, OERR

class Scaler_obs:
    def __init__(self, val, type, time, position, sigma_r, dt_avg=1):
        assert isinstance(val, float)
        assert isinstance(type, str)
        assert isinstance(time, (int, float))
        assert isinstance(position, (int, float))
        assert isinstance(sigma_r, float)
        assert isinstance(dt_avg, int)
        self.val = val
        self.type = type
        self.time = time
        self.position = position
        self.sigma_r = sigma_r
        self.dt_avg = dt_avg  # observation is averaged (time - dt_avg, time]

    def __str__(self):
        return f"val: {self.val}\ntype: {self.type}\nposition: {self.position}\nsigma_r: {self.sigma_r}\ndt_avg: {self.dt_avg}"

def dist(i1, i2):
    # Only for the model of Dirren and Hakim
    assert isinstance(i1, int)
    assert isinstance(i2, int)
    assert N_MODEL % 2 == 0
    half = N_MODEL // 2
    i1m = i1 % half
    i2m = i2 % half
    d1 = abs(i1m - i2m)
    dist = min(d1, half - d1)
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
        assert t_end - aint < o.time - o.dt_avg + 1 <= t_end
        it = o.time - t_end + aint - 1
        assert 0 <= it < aint
        for i in range(k_ens):
            yb_raw[j, i] = generate_single_obs(fcst[it - o.dt_avg + 1:it + 1, i, :],
                o.position, 0.0, o.time, o.dt_avg).val
    return yb_raw

def generate_single_obs(x_in, position, sigma_r, time, dt_avg):
    # model must be a model of Dirren and Hakim
    assert x_in.shape == (dt_avg, N_MODEL)
    assert isinstance(position, int)
    assert 0 <= position < N_MODEL // 2
    x = np.mean(x_in, axis=0)
    model_state = x[position] + x[position + N_MODEL // 2]
    obs = Scaler_obs(model_state + np.random.randn() * sigma_r, "Dirren_averaged", time, position, sigma_r, dt_avg)
    return obs

if __name__ == "__main__":
    pass
