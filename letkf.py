#!/usr/bin/env python

import numpy as np
from scipy.linalg import sqrtm
from const import N_MODEL
from obs import dist, getr, get_background_obs

def letkf(fcst, olist, rho, l_loc, t_end, aint):
    k_ens = fcst.shape[1]
    assert isinstance(olist, list)
    p_obs = len(olist)
    assert fcst.shape == (aint, k_ens, N_MODEL)
    assert isinstance(rho, float)
    assert isinstance(k_ens, int)
    assert isinstance(l_loc, (int, float))
    assert isinstance(t_end, int)

    i_mm = np.identity(k_ens)
    i_1m = np.ones((1, k_ens))

    yo = np.empty((p_obs, 1))
    for j in range(p_obs):
        assert t_end - aint < olist[j].time <= t_end
        yo[j, 0] = olist[j].val
    r = getr(olist)

    xf_raw = fcst[-1, :, :].T
    xf = np.mean(xf_raw, axis=1)[:, np.newaxis]
    xfpt = xf_raw - xf @ i_1m
    yb_raw = get_background_obs(olist, fcst, t_end, aint)
    yb = np.mean(yb_raw, axis=1)[:, np.newaxis]
    ybpt = yb_raw[:, :] - yb[:, :]

    xai = np.zeros((k_ens, N_MODEL))
    for i in range(N_MODEL):
        # step 3
        ind   = obs_within(i, l_loc, olist)
        lw    = get_localization_weight(ind, i, l_loc, olist)
        yol   = yo[ind, :]
        ybl   = yb[ind, :]
        ybptl = ybpt[ind, :]
        xfl   = xf[i:i + 1, :]
        xfptl = xfpt[i:i + 1, :]
        rl    = r[ind, :][:, ind]

        # step 4-9
        cl    = ybptl.T @ (np.linalg.inv(rl) * lw)
        pal   = np.linalg.inv(((k_ens - 1.0) / rho) * i_mm + cl @ ybptl)
        waptl = np.real(sqrtm((k_ens - 1.0) * pal))
        wal   = pal @ cl @ (yol - ybl)
        xail  = xfl @ i_1m + xfptl @ (wal @ i_1m + waptl)
        assert xail.shape == (1, k_ens)
        xai[:, i] = xail[0, :]

    return xai

def get_localization_weight(ind, ic, length, olist):
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
        di = dist(ic, olist[j].position)
        if smooth:
            co = gc99(di / length)
        else:
            co = 0.0 if di > length else 1.0
        lw[j, :] = co
        lw[:, j] = co
    return lw

def obs_within(i, l_loc, olist):
    # O(p). This can be modified to O(1)
    assert isinstance(i, int)
    assert isinstance(l_loc, int)
    list_j = []
    for j in range(len(olist)):
        if dist(i, olist[j].position) <= l_loc:
            list_j.append(j)
    return list_j

