#!/usr/bin/env python

from functools import partial
import numpy as np
from scipy.optimize import minimize
from const import N_MODEL, DT, static_b
from model import Model
from obs import getr, get_background_obs, get_h_matrix, get_yo

def tdvar(fcst, obs, sigma_b, t_anl, do_cvt=True):
    assert fcst.shape == (N_MODEL,)
    assert_synoptic(obs, t_anl)
    r_inv = np.linalg.inv(getr(obs))
    b_inv = np.linalg.inv(sigma_b ** 2 * static_b())
    if do_cvt:
        l = np.linalg.cholesky(sigma_b ** 2 * static_b())
        first_guess = np.zeros(N_MODEL)
        yo = get_yo(obs)
        yb = get_background_obs(obs, fcst[None, None, :], t_anl, aint=1)
        d = yo - yb
        h = get_h_matrix(obs)
        hl = h @ l
        cf = partial(tdvar_cvt_2j, d=d, hl=hl, r_inv=r_inv)
        opt = minimize(cf, first_guess, method="bfgs")
        anl = fcst[:, None] + l @ opt.x[:, None]
        anl = anl[:, 0]
    else:
        first_guess = np.copy(fcst)
        cf = partial(tdvar_2j, fcst_in=fcst, obs=obs, r_inv=r_inv, b_inv=b_inv, t_anl=t_anl)
        opt = minimize(cf, first_guess, method="bfgs")
        anl = opt.x
    return anl

def tdvar_2j(anl_in, fcst_in, obs, r_inv, b_inv, t_anl):
    assert anl_in.shape == fcst_in.shape == (N_MODEL,)
    anl = anl_in[:, None]
    fcst = fcst_in[:, None]
    yo = get_yo(obs)
    yb = get_background_obs(obs, anl_in[None, None, :], t_anl, aint=1)
    twoj = (anl - fcst).T @ b_inv @ (anl - fcst) + \
           (yb - yo).T @ r_inv @ (yb - yo)
    assert twoj.shape == (1, 1)
    return twoj[0, 0]

def tdvar_cvt_2j(anl_v_in, d, hl, r_inv):
    p = d.shape[0]
    assert anl_v_in.shape == (N_MODEL,)
    assert d.shape == (p, 1)
    assert hl.shape == (p, N_MODEL)
    assert r_inv.shape == (p, p)
    anl_v = anl_v_in[:, None]
    twoj = (anl_v.T @ anl_v) + (d - hl @ anl_v).T @ r_inv @ (d - hl @ anl_v)
    assert twoj.shape == (1, 1)
    return twoj[0, 0]

def assert_synoptic(obs, t_anl):
    for o in obs:
        assert o.time == t_anl

def tdvar_analytic(fcst, obs, sigma_b, t_anl):
    assert fcst.shape == (N_MODEL,)
    assert_synoptic(obs, t_anl)
    yb = get_background_obs(obs, fcst[None, None, :], t_anl, aint=1)
    yo = get_yo(obs)
    d = yo - yb
    r_inv = np.linalg.inv(getr(obs))
    b_inv = np.linalg.inv(sigma_b ** 2 * static_b())
    h = get_h_matrix(obs)
    delta_x = np.linalg.inv(b_inv + h.T @ r_inv @ h) @ h.T @ r_inv @ d
    anl = fcst[:, None] + delta_x
    assert anl.shape == (N_MODEL, 1)
    return anl[:, 0]

