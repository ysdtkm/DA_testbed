#!/usr/bin/env python

from scipy.optimize import fmin_bfgs
import numpy as np
from model import Model
from const import N_MODEL, AINT, DT
from obs import getr, get_background_obs

def fdvar(fcst_0, obs, sigma_b, t_end):
    assert isinstance(fcst_0, np.ndarray)
    assert fcst_0.shape == (N_MODEL,)

    first_guess = np.copy(fcst_0)
    anl_0 = fmin_bfgs(fdvar_2j, first_guess, args=(fcst_0, obs, sigma_b, t_end), disp=True)
    anl_1 = np.copy(anl_0)
    for i in range(AINT):
        anl_1 = Model().rk4(anl_1, DT)
    return anl_1

def fdvar_2j(anl_0, fcst_0, obs, sigma_b, t_end):
    assert anl_0.shape == fcst_0.shape == (N_MODEL,)
    assert isinstance(sigma_b, float)
    anl_0 = anl_0[:, np.newaxis]
    fcst_0 = fcst_0[:, np.newaxis]
    p_obs = len(obs)
    yo = np.empty((p_obs, 1))
    for j in range(p_obs):
        assert t_end - AINT < obs[j].time <= t_end
        yo[j, 0] = obs[j].val
    r = getr(obs)

    b = sigma_b ** 2 * static_b()
    traj = np.empty((AINT, 1, N_MODEL))
    traj[0, 0, :] = anl_0[:, 0]
    for i in range(1, AINT):
        traj[i, 0, :] = Model().rk4(traj[i - 1, 0, :], DT)
    yb_raw = get_background_obs(obs, traj, t_end)
    twoj = (anl_0 - fcst_0).T @ np.linalg.inv(b) @ (anl_0 - fcst_0) + \
           (yb_raw - yo).T @ np.linalg.inv(r) @ (yb_raw - yo)
    assert twoj.shape == (1, 1)
    return twoj[0, 0]

def static_b():
    return np.identity(N_MODEL)
