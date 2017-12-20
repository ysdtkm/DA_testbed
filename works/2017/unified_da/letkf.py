#!/usr/bin/env python

import numpy as np
from scipy.linalg import sqrtm
from const import N_MODEL, P_OBS, pos_obs
import model


def letkf(fcst: np.ndarray, h_nda: np.ndarray, r_nda: np.ndarray, yo_nda: np.ndarray,
          rho: float, k_ens: int, l_loc: int) -> tuple:
    """
    :param fcst:         [k_ens, N_MODEL]
    :param h_nda:        [P_OBS, N_MODEL]
    :param r_nda:        [P_OBS, P_OBS]
    :param yo_nda:       [P_OBS, 1]
    :param rho:
    :param k_ens:
    :param l_loc:
    :return xa:          [N_MODEL, k_ens]
    :return xfpt:        [N_MODEL, N_MODEL]
    :return xapt:        [N_MODEL, N_MODEL]
    """

    h = np.asmatrix(h_nda)
    r = np.asmatrix(r_nda)
    yo = np.asmatrix(yo_nda)
    if fcst.shape[1] != N_MODEL:
        raise Exception

    i_mm = np.asmatrix(np.identity(k_ens))
    i_1m = np.asmatrix(np.ones((1, k_ens)))

    xfm = np.matrix(fcst[:, :]).T
    xf = np.mean(xfm, axis=1)
    xfpt = xfm - xf * i_1m
    ybpt = h * xfpt
    yb = h * xf

    xai = np.matrix(np.zeros((N_MODEL, k_ens)))

    for i in range(N_MODEL):
        # step 3
        ind   = model.obs_within(i, l_loc)
        lw    = get_localization_weight(ind, i, l_loc)
        yol   = yo[ind, :]
        ybl   = yb[ind, :]
        ybptl = ybpt[ind, :]
        xfl   = xf[i, :]
        xfptl = xfpt[i, :]
        rl    = r[ind, :][:, ind]

        # step 4-9
        cl    = ybptl.T * np.asmatrix(rl.I.A * lw)
        pal   = (((k_ens - 1.0) / rho) * i_mm + cl * ybptl).I
        waptl = np.asmatrix(np.real(sqrtm((k_ens - 1.0) * pal)))

        wal   = pal * cl * (yol - ybl)
        xail  = xfl * i_1m + xfptl * (wal * i_1m + waptl)
        xai[i, :] = xail[:, :]

    return np.real(xai.T.A)


def get_localization_weight(ind: list, ic: int, length: int):
    # O(dim^2)

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
        di = model.dist(ic, pos_obs(jg))
        if smooth:
            co = gc99(di / length)
        else:
            co = 0.0 if di > length else 1.0
        lw[j, :] = co
        lw[:, j] = co
    return lw

