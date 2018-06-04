#!/usr/bin/env python

import letkf
import model
from model import Model
from obs import geth, getr, Scaler_obs
import numpy as np
from const import EXPLIST, DT, STEPS, STEP_FREE, N_MODEL, P_OBS, FERR_INI, AINT, SEED, pos_obs

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

def exec_nature():
    all_true = np.empty((STEPS, N_MODEL))
    true = np.random.randn(N_MODEL) * FERR_INI

    # forward integration i-1 -> i
    for i in range(0, STEPS):
        true[:] = Model().rk4(true[:], DT)
        all_true[i, :] = true[:]
    np.save("data/true.npy", all_true)

    np.random.seed(SEED * 2)
    return all_true

def exec_obs(nature):
    assert isinstance(nature, np.ndarray)
    assert nature.shape == (STEPS, N_MODEL)
    all_obs = np.empty((STEPS, P_OBS), dtype=object)
    h = geth()
    r = getr()
    for i in range(0, STEPS):
        obs = h.dot(nature[i, :]) + np.random.randn(P_OBS) * r.diagonal() ** 0.5
        for j in range(P_OBS):
            all_obs[i, j] = Scaler_obs(obs[j], "", pos_obs(j), r.diagonal()[j] ** 0.5)
    np.save("data/obs.npy", all_obs)
    return all_obs

def exec_free_run(settings):
    assert isinstance(settings, dict)
    free_run = np.empty((STEPS, settings["k_ens"], N_MODEL))
    for m in range(0, settings["k_ens"]):
        free_run[0, m, :] = np.random.randn(N_MODEL) * FERR_INI
        for i in range(1, STEP_FREE):
            free_run[i, m, :] = Model().rk4(free_run[i - 1, m, :], DT)
    return free_run

def exec_assim_cycle(settings, all_fcst, all_obs):
    assert isinstance(settings, dict)
    assert all_fcst.shape == (STEPS, settings["k_ens"], N_MODEL)
    assert all_obs.shape == (STEPS, P_OBS)

    # prepare containers
    r = getr()
    h = geth()
    fcst = np.empty((settings["k_ens"], N_MODEL))
    all_back_cov = np.empty((STEPS, N_MODEL, N_MODEL))

    # forecast-analysis cycle
    try:
        for i in range(STEP_FREE, STEPS):
            for m in range(0, settings["k_ens"]):
                fcst[m, :] = Model().rk4(all_fcst[i - 1, m, :], DT)
            if i % AINT == 0:
                all_back_cov[i, :, :] = get_back_cov(fcst)
                fcst[:, :] = analyze_one_window(fcst, all_obs[i, :], h, r, settings)
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
    np.save("data/%s_cycle.npy" % settings["name"], all_fcst)
    np.save("data/%s_bcov.npy" % settings["name"], all_back_cov)
    return all_fcst

def analyze_one_window(fcst, obs, h, r, settings):
    assert isinstance(settings, dict)
    assert fcst.shape == (settings["k_ens"], N_MODEL)
    assert obs.shape == (P_OBS,)
    assert h.shape == (P_OBS, N_MODEL)
    assert r.shape == (P_OBS, P_OBS)

    anl = np.empty((settings["k_ens"], N_MODEL))
    yo = np.empty((P_OBS, 1))
    for j in range(P_OBS):
        yo[j, 0] = obs[j].val
    if settings["method"] == "letkf":
        anl[:, :] = letkf.letkf(fcst[:, :], h[:, :], r[:, :], yo[:, :],
                                settings["rho"], settings["k_ens"], settings["l_loc"])
    else:
        raise Exception("analysis method mis-specified: %s" % settings["method"])
    return anl[:, :]

def get_back_cov(ens):
    k_ens = ens.shape[0]
    assert ens.shape == (k_ens, N_MODEL)
    ens_mean = np.mean(ens, axis=0)
    cov = np.empty((N_MODEL, N_MODEL))
    for i in range(N_MODEL):
        for j in range(N_MODEL):
            ptb_i = ens[:, i] - ens_mean[i]
            ptb_j = ens[:, j] - ens_mean[j]
            cov[i, j] = np.sum(ptb_i * ptb_j) / (k_ens - 1.0)
    return cov

if __name__ == "__main__":
    main()
