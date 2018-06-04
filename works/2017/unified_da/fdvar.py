#!/usr/bin/env python

from functools import lru_cache, partial
from scipy.optimize import minimize
import numpy as np
from model import Model
from const import N_MODEL, AINT, DT
from obs import getr, get_background_obs
import pickle
from autograd import value_and_grad

def fdvar(fcst_0, obs, sigma_b, t_end):
    assert isinstance(fcst_0, np.ndarray)
    assert fcst_0.shape == (N_MODEL,)

    first_guess = np.copy(fcst_0)
    cf = partial(fdvar_2j, fcst_0=fcst_0, obs=obs, sigma_b=sigma_b, t_end=t_end)
    cfg = value_and_grad(cf)
    opt = minimize(cfg, first_guess, method="bfgs", jac=True)
    print(opt)
    anl_0 = opt.x
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
    traj = (anl_0[:, :].T)[np.newaxis, :, :]
    for i in range(1, AINT):
        next = Model().rk4(traj[i - 1, 0, :], DT)
        traj = np.concatenate((traj, next[np.newaxis, np.newaxis, :]), axis=0)
    assert traj.shape == (AINT, 1, N_MODEL)
    yb_raw = get_background_obs(obs, traj, t_end)
    twoj = np.dot((anl_0 - fcst_0).T, np.dot(np.linalg.inv(b), anl_0 - fcst_0)) + \
           np.dot((yb_raw - yo).T, np.dot(np.linalg.inv(r), yb_raw - yo))
    assert twoj.shape == (1, 1)
    return twoj[0, 0]

def test_fdvar_2j():
    with open("args.pkl", "rb") as f:
        anl_0, fcst_0, obs, sigma_b, t_end = pickle.load(f)
    cf = partial(fdvar_2j, fcst_0=fcst_0, obs=obs, sigma_b=sigma_b, t_end=t_end)
    cfg = value_and_grad(cf)
    v, g = cfg(anl_0)
    print(v, g)

@lru_cache(maxsize=1)
def static_b():
    b = np.load("blob/mean_b_cov.npy")
    assert b.shape == (N_MODEL, N_MODEL)
    eigs = np.linalg.eigvalsh(b)
    assert np.all(eigs > 0.0)
    b /= np.max(eigs)
    return b

if __name__ == "__main__":
    test_fdvar_2j()
