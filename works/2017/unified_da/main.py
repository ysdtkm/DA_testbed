#!/usr/bin/env python

import letkf
import model
from model import Model
import numpy as np
from const import EXPLIST, DT, STEPS, STEP_FREE, N_MODEL, P_OBS, FERR_INI, AINT, SEED


def main():
    np.random.seed(SEED * 1)
    nature = exec_nature()

    np.random.seed(SEED * 3)
    obs = exec_obs(nature)

    for settings in EXPLIST:
        print("Analysis cycle: %s" % settings["name"])
        np.random.seed(SEED * 4)
        free = exec_free_run(settings)
        anl = exec_assim_cycle(settings, free, obs)


def exec_nature() -> np.ndarray:
    """
    :return all_true: [STEPS, N_MODEL]
    """
    all_true = np.empty((STEPS, N_MODEL))
    true = np.random.randn(N_MODEL) * FERR_INI

    # forward integration i-1 -> i
    for i in range(0, STEPS):
        true[:] = Model().rk4(true[:], DT)
        all_true[i, :] = true[:]
    np.save("data/true.npy", all_true)

    np.random.seed(SEED * 2)
    return all_true


def exec_obs(nature: np.ndarray) -> np.ndarray:
    """
    note: this method currently cannot handle non-diagonal element of R

    :param nature:   [STEPS, N_MODEL]
    :return all_obs: [STEPS, P_OBS]
    """
    all_obs = np.empty((STEPS, P_OBS))
    h = model.geth()
    r = model.getr()
    for i in range(0, STEPS):
        all_obs[i, :] = h.dot(nature[i, :]) + np.random.randn(P_OBS) * r.diagonal() ** 0.5

    np.save("data/obs.npy", all_obs)
    return all_obs


def exec_free_run(settings: dict) -> np.ndarray:
    """
    :param settings:
    :return free_run: [STEPS, k_ens, N_MODEL]
    """
    free_run = np.empty((STEPS, settings["k_ens"], N_MODEL))
    for m in range(0, settings["k_ens"]):
        free_run[0, m, :] = np.random.randn(N_MODEL) * FERR_INI
        for i in range(1, STEP_FREE):
            free_run[i, m, :] = Model().rk4(free_run[i - 1, m, :], DT)
    return free_run


def exec_assim_cycle(settings: dict, all_fcst: np.ndarray, all_obs: np.ndarray) -> np.ndarray:
    """
    :param settings:
    :param all_fcst:  [STEPS, k_ens, N_MODEL]
    :param all_obs:   [STEPS, P_OBS]
    :return all_fcst: [STEPS, k_ens, N_MODEL]
    """

    # prepare containers
    r = model.getr()
    h = model.geth()
    fcst = np.empty((settings["k_ens"], N_MODEL))
    all_back_cov = np.empty((STEPS, N_MODEL, N_MODEL))
    obs_used = np.empty((STEPS, P_OBS))
    obs_used[:, :] = np.nan

    # forecast-analysis cycle
    try:
        for i in range(STEP_FREE, STEPS):
            for m in range(0, settings["k_ens"]):
                fcst[m, :] = Model().rk4(all_fcst[i - 1, m, :], DT)

            if i % AINT == 0:
                obs_used[i, :] = all_obs[i, :]
                all_back_cov[i, :, :] = get_back_cov(fcst)
                fcst[:, :] = \
                    analyze_one_window(fcst, all_obs[i, :], h, r, settings)

            all_fcst[i, :, :] = fcst[:, :]

    except (np.linalg.LinAlgError, ValueError) as e:
        import traceback
        print("")
        print("ANALYSIS CYCLE DIVERGED: %s" % e)
        print("Settings: ", settings)
        print("This experiment is terminated (see error traceback below). Continue on next experiments.")
        print("")
        traceback.print_exc()
        print("")

    # save to files
    np.save("data/%s_obs.npy" % settings["name"], obs_used)
    np.save("data/%s_cycle.npy" % settings["name"], all_fcst)
    np.save("data/%s_bcov.npy" % settings["name"], all_back_cov)

    return all_fcst


def analyze_one_window(fcst: np.ndarray, obs: np.ndarray, h: np.ndarray, r: np.ndarray,
                       settings: dict) -> tuple:
    """
    :param  fcst:         [k_ens, N_MODEL]
    :param  obs:          [P_OBS]
    :param  h:            [P_OBS, N_MODEL]
    :param  r:            [P_OBS, P_OBS]
    :param  settings:
    :return anl:          [k_ens, N_MODEL]
    """

    anl = np.empty((settings["k_ens"], N_MODEL))
    yo = obs[:, np.newaxis]

    if settings["method"] == "letkf":
        anl[:, :] = letkf.letkf(fcst[:, :], h[:, :], r[:, :], yo[:, :],
                                settings["rho"], settings["k_ens"], settings["l_loc"])
    else:
        raise Exception("analysis method mis-specified: %s" % settings["method"])

    return anl[:, :]


def get_back_cov(ens: np.ndarray):
    """
    O(n^2)
    :param  ens:          [k_ens, N_MODEL]
    :return cov:          [N_MODEL, N_MODEL]
    """
    k_ens = ens.shape[0]
    ens_mean = np.mean(ens, axis=0)
    cov = np.empty((N_MODEL, N_MODEL))
    for i in range(N_MODEL):
        for j in range(N_MODEL):
            ptb_i = ens[:, i] - ens_mean[i]
            ptb_j = ens[:, j] - ens_mean[j]
            cov[i, j] = np.sum(ptb_i * ptb_j) / (k_ens - 1.0)
    return cov


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("process interrupted by keyboard.")
        import sys
        sys.exit(1)
