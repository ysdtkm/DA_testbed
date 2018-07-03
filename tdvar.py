#!/usr/bin/env python

from functools import partial
import numpy as np
from scipy.optimize import minimize
from const import N_MODEL, DT
from fdvar import static_b
from model import Model
from obs import getr, get_background_obs, get_h_matrix

def tdvar(fcst, obs, sigma_b, t_anl):
    assert fcst.shape == (N_MODEL,)
    assert_synoptic(obs, t_anl)
    r_inv = np.linalg.inv(getr(obs))
    b_inv = np.linalg.inv(sigma_b ** 2 * static_b())
    first_guess = np.copy(fcst)
    cf = partial(tdvar_2j, fcst_in=fcst, obs=obs, r_inv=r_inv, b_inv=b_inv, t_anl=t_anl)
    opt = minimize(cf, first_guess, method="bfgs")
    anl = opt.x
    return anl

def assert_synoptic(obs, t_anl):
    for o in obs:
        assert o.time == t_anl

def tdvar_analytic(fcst, obs, sigma_b, t_anl):
    assert fcst.shape == (N_MODEL,)
    assert_synoptic(obs, t_anl)
    yb_raw = get_background_obs(obs, fcst[None, None, :], t_anl, aint=1)
    yo = get_yo(obs)
    d = yo - yb_raw
    r_inv = np.linalg.inv(getr(obs))
    b_inv = np.linalg.inv(sigma_b ** 2 * static_b())
    h = get_h_matrix(obs)
    delta_x = np.linalg.inv(b_inv + h.T @ r_inv @ h) @ h.T @ r_inv @ d
    anl = fcst + delta_x
    return anl

def get_yo(obs):
    p_obs = len(obs)
    yo = np.empty((p_obs, 1))
    for j in range(p_obs):
        yo[j, 0] = obs[j].val
    return yo

def tdvar_2j(anl_in, fcst_in, obs, r_inv, b_inv, t_anl):
    assert anl_in.shape == fcst_in.shape == (N_MODEL,)
    anl = anl_in[:, None]
    fcst = fcst_in[:, None]
    yo = get_yo(obs)
    yb_raw = get_background_obs(obs, anl_in[None, None, :], t_anl, aint=1)
    twoj = (anl - fcst).T @ b_inv @ (anl - fcst) + \
           (yb_raw - yo).T @ r_inv @ (yb_raw - yo)
    assert twoj.shape == (1, 1)
    return twoj[0, 0]

