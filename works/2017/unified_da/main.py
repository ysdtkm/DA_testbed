#!/usr/bin/env python

import traceback
import model
from model import Model
from obs import Scaler_obs
import numpy as np
from const import EXPLIST, DT, STEPS, STEP_FREE, N_MODEL, P_OBS, OERR, FERR_INI, AINT, SEED, pos_obs
from da_system import Da_system

def main():
    np.random.seed(SEED * 1)
    nature = exec_nature()

    np.random.seed(SEED * 3)
    obs = exec_obs(nature)

    for settings in EXPLIST:
        print("Analysis cycle: %s" % settings["name"])
        np.random.seed(SEED * 4)
        fcst = init_background(settings)
        anl = exec_assim_cycle(settings, fcst, obs)

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
    all_obs = []
    for i in range(0, STEPS):
        obs_t = []
        for j in range(P_OBS):
            k = pos_obs(j)
            oval = nature[i, k] + np.random.randn(1)[0] * OERR
            obs_t.append(Scaler_obs(oval, "", i, pos_obs(j), OERR))
        all_obs.append(obs_t)
    np.save("data/obs.npy", np.array(all_obs))
    return all_obs

def init_background(settings):
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
    all_back_cov = np.empty((STEPS, N_MODEL, N_MODEL))
    da_sys = Da_system(settings)
    try:
        for i in range(STEP_FREE, STEPS):
            for m in range(0, settings["k_ens"]):
                all_fcst[i, m, :] = Model().rk4(all_fcst[i - 1, m, :], DT)
            if i % AINT == 0:
                all_back_cov[i, :, :] = get_back_cov(all_fcst[i, :, :])
                all_fcst[i, :, :] = da_sys.analyze_one_window(all_fcst[i, :, :], all_obs[i])
    except (np.linalg.LinAlgError, ValueError) as e:
        print("ANALYSIS CYCLE DIVERGED: %s" % e)
        print("Settings: ", settings)
        traceback.print_exc()
    np.save("data/%s_cycle.npy" % settings["name"], all_fcst)
    np.save("data/%s_bcov.npy" % settings["name"], all_back_cov)
    return all_fcst

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
