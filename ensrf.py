#!/usr/bin/env python

import numpy as np
from scipy.linalg import sqrtm, inv
from const import N_MODEL
from obs import dist, getr, get_background_obs, Scaler_obs

def ensrf_all(fcst, obs, rho, t_end, aint):
    # 4D-EnSRF with no localization
    anl = apply_multiplicative_infl(fcst, rho)
    for o in obs:
        anl = ensrf_single(anl, o, t_end, aint)
    assert anl.shape == fcst.shape
    return anl[-1, :, :]

def ensrf_single(fcst, obs, t_end, aint):
    assert isinstance(obs, Scaler_obs)
    k_ens = fcst.shape[1]
    assert fcst.shape == (aint, k_ens, N_MODEL)
    assert isinstance(k_ens, int)
    assert isinstance(t_end, int)

    assert t_end - aint < obs.time <= t_end
    yo = np.empty((1, 1))
    yo[0, 0] = obs.val
    R = getr([obs])
    assert R.shape == (1, 1)
    yb_raw = get_background_obs([obs], fcst, t_end, aint)
    assert yb_raw.shape == (1, k_ens)

    X_raw = np.empty((aint * N_MODEL, k_ens))
    for t in range(aint):
        X_raw[t * N_MODEL:(t + 1) * N_MODEL, :] = fcst[t, :, :].T

    x_mean = np.mean(X_raw, axis=1)[:, None]
    X_ptb = X_raw - x_mean
    Ef = (k_ens - 1.0) ** (-0.5) * X_ptb
    yb_mean = np.mean(yb_raw, axis=1)[:, None]
    HEf = (k_ens - 1.0) ** (-0.5) * (yb_raw - yb_mean)
    K = Ef @ HEf.T @ inv(HEf @ HEf.T + R)
    xa_mean = x_mean + K @ (yo - yb_mean)
    alpha = (1.0 + (R[0, 0] / ((HEf @ HEf.T) + R[0, 0])) ** 0.5) ** (-1)
    Ea = (k_ens - 1.0) ** 0.5 * (Ef - alpha * K @ HEf)
    Xa = Ea + xa_mean
    assert Xa.shape == (aint * N_MODEL, k_ens)

    anl = np.empty((aint, k_ens, N_MODEL))
    for t in range(aint):
        anl[t, :, :] = Xa[t * N_MODEL:(t + 1) * N_MODEL, :].T
    return anl

def apply_multiplicative_infl(fcst, rho):
    aint, k_ens, _ = fcst.shape
    assert fcst.shape == (aint, k_ens, N_MODEL)
    ptb = fcst - np.mean(fcst, axis=1)[:, None, :]
    return np.mean(fcst, axis=1)[:, None, :] + ptb * rho ** 0.5

