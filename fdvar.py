#!/usr/bin/env python

from functools import lru_cache, partial
from scipy.optimize import minimize
import numpy as np
from model import Model
from const import N_MODEL, DT
from obs import getr, get_background_obs

def fdvar(fcst_0, obs, sigma_b, t_end, aint):
    assert isinstance(fcst_0, np.ndarray)
    assert fcst_0.shape == (N_MODEL,)

    r_inv = np.linalg.inv(getr(obs))
    b_inv = np.linalg.inv(sigma_b ** 2 * static_b())

    first_guess = np.copy(fcst_0)
    cf = partial(fdvar_2j, fcst_0=fcst_0, obs=obs, r_inv=r_inv, b_inv=b_inv,
            t_end=t_end, aint=aint)
    opt = minimize(cf, first_guess, method="bfgs")
    # print(opt.message)
    anl_0 = opt.x
    anl_1 = np.copy(anl_0)
    for i in range(aint):
        anl_1 = Model().rk4(anl_1, DT)
    return anl_1

def fdvar_2j(anl_0, fcst_0, obs, r_inv, b_inv, t_end, aint):
    assert anl_0.shape == fcst_0.shape == (N_MODEL,)
    anl_0 = anl_0[:, np.newaxis]
    fcst_0 = fcst_0[:, np.newaxis]
    p_obs = len(obs)
    yo = np.empty((p_obs, 1))
    for j in range(p_obs):
        assert t_end - aint < obs[j].time <= t_end
        yo[j, 0] = obs[j].val
    traj = Model().rk4(anl_0[:, 0], DT)[np.newaxis, np.newaxis, :]
    for i in range(1, aint):
        next = Model().rk4(traj[i - 1, 0, :], DT)
        traj = np.concatenate((traj, next[np.newaxis, np.newaxis, :]), axis=0)
    assert traj.shape == (aint, 1, N_MODEL)
    yb_raw = get_background_obs(obs, traj, t_end, aint)
    twoj = np.dot((anl_0 - fcst_0).T, np.dot(b_inv, anl_0 - fcst_0)) + \
           np.dot((yb_raw - yo).T, np.dot(r_inv, yb_raw - yo))
    assert twoj.shape == (1, 1)
    return twoj[0, 0]

@lru_cache(maxsize=1)
def static_b():
    b = np.load("blob/mean_b_cov.npy")
    assert b.shape == (N_MODEL, N_MODEL)
    eigs = np.linalg.eigvalsh(b)
    assert np.all(eigs > 0.0)
    b /= np.max(eigs)
    return b

