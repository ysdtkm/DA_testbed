#!/usr/bin/env python

import numpy as np
from scipy.linalg import sqrtm
from const import N_MODEL
from obs import dist, getr, get_background_obs

def letkf(fcst, obs, rho, l_loc, t_end, aint, smoother):
    k_ens = fcst.shape[1]
    assert isinstance(obs, list)
    p_obs = len(obs)
    assert fcst.shape == (aint, k_ens, N_MODEL)
    assert isinstance(rho, float)
    assert isinstance(k_ens, int)
    assert isinstance(l_loc, (int, float))
    assert isinstance(t_end, int)

    i_mm = np.identity(k_ens)
    i_1m = np.ones((1, k_ens))

    yo = np.empty((p_obs, 1))
    for j in range(p_obs):
        assert t_end - aint < obs[j].time <= t_end
        yo[j, 0] = obs[j].val
    r = getr(obs)

    if smoother:
        xf_raw_t = np.transpose(fcst, (0, 2, 1))
        assert xf_raw_t.shape == (aint, N_MODEL, k_ens)
        xf_t = np.mean(xf_raw_t, axis=2)[:, :, None]
        xfpt_t = xf_raw_t - np.repeat(xf_t, k_ens, axis=2)
        xai = np.zeros((aint, k_ens, N_MODEL))
    else:
        xf_raw = fcst[-1, :, :].T
        xf = np.mean(xf_raw, axis=1)[:, np.newaxis]
        xfpt = xf_raw - xf @ i_1m
        xai = np.zeros((k_ens, N_MODEL))

    yb_raw = get_background_obs(obs, fcst, t_end, aint)
    yb = np.mean(yb_raw, axis=1)[:, np.newaxis]
    ybpt = yb_raw[:, :] - yb[:, :]

    for i in range(N_MODEL):
        # step 3
        ind   = obs_within(i, l_loc, obs)
        lw    = get_localization_weight(ind, i, l_loc, obs)
        yol   = yo[ind, :]
        ybl   = yb[ind, :]
        ybptl = ybpt[ind, :]
        rl    = r[ind, :][:, ind]

        # step 4-9
        cl    = ybptl.T @ (np.linalg.inv(rl) * lw)
        rho2  = rho if i >= N_MODEL // 2 else rho ** 5  # ttk
        pal   = np.linalg.inv(((k_ens - 1.0) / rho2) * i_mm + cl @ ybptl)
        waptl = np.real(sqrtm((k_ens - 1.0) * pal))
        wal   = pal @ cl @ (yol - ybl)

        s = np.s_[i:i + 1]
        if smoother:
            xfl_t = xf_t[:, s, :]
            xfptl_t = xfpt_t[:, s, :]
            for t in range(aint):
                xail = xfl_t[t, :, :] @ i_1m + xfptl_t[t, :, :] @ (wal @ i_1m + waptl)
                assert xail.shape == (1, k_ens)
                xai[t, :, i] = xail[0, :]
        else:
            xfl = xf[s, :]
            xfptl = xfpt[s, :]
            xail = xfl @ i_1m + xfptl @ (wal @ i_1m + waptl)
            assert xail.shape == (1, k_ens)
            xai[:, i] = xail[0, :]

    return xai

def get_localization_weight(ind, ic, length, obs):
    # O(dim^2)
    assert isinstance(ind, list)
    assert isinstance(ic, int)
    assert isinstance(length, (int, float))

    def gc99(r):
        r = abs(r)
        if r > 2:
            return 0
        elif r <= 1:
            return -1 / 4 * r ** 5 + 1 / 2 * r ** 4 + 5 / 8 * r ** 3 - 5 / 3 * r ** 2 + 1
        else:
            return 1 / 12 * r ** 5 - 1 / 2 * r ** 4 + 5 / 8 * r ** 3 + 5 / 3 * r ** 2 \
                   - 5 * r + 4 - 2 / 3 / r

    smooth = False
    dim = len(ind)
    lw = np.ones((dim, dim))
    for j, jg in enumerate(ind):
        di = dist(ic, obs[j].position)
        if smooth:
            co = gc99(di / length)
        else:
            co = 0.0 if di > length else 1.0
        lw[j, :] = co
        lw[:, j] = co
    return lw

def obs_within(i, l_loc, obs):
    # O(p). This can be modified to O(1)
    assert isinstance(i, int)
    assert isinstance(l_loc, int)
    list_j = []
    for j in range(len(obs)):
        if dist(i, obs[j].position) <= l_loc:
            list_j.append(j)
    return list_j

