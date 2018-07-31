#!/usr/bin/env python

import sys
import traceback
from model import model_step
from obs import Scaler_obs, generate_single_obs
import numpy as np
from const import EXPLIST, DT, STEPS, STEP_FREE, N_MODEL, P_OBS, OERR, FERR_INI, SEED, pos_obs, DT_OBS, AINT
from da_system import Da_system
from tqdm import trange

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
    for i in range(STEPS):
        true[:] = model_step(true[:], DT)
        all_true[i, :] = true[:]
    np.save("data/true.npy", all_true)

    np.random.seed(SEED * 2)
    return all_true

def exec_obs(nature):
    assert isinstance(nature, np.ndarray)
    assert nature.shape == (STEPS, N_MODEL)
    all_obs = []
    for i in range(STEPS):
        obs_t = []
        if i - DT_OBS + 1 < 0 or i % DT_OBS != 0:
            all_obs.append([])
            continue
        for j in range(P_OBS):
            k = pos_obs(j)
            o = generate_single_obs(nature[i - DT_OBS + 1:i + 1, :], k, OERR, i, DT_OBS)
            obs_t.append(o)
        all_obs.append(obs_t)
    np.save("data/obs.npy", np.array(all_obs))
    return all_obs

def init_background(settings):
    assert isinstance(settings, dict)
    free_run = np.empty((STEPS, settings["k_ens"], N_MODEL))
    for m in range(0, settings["k_ens"]):
        free_run[0, m, :] = np.random.randn(N_MODEL) * FERR_INI
        for i in range(1, STEP_FREE):
            free_run[i, m, :] = model_step(free_run[i - 1, m, :], DT)
    return free_run

def exec_assim_cycle(settings, all_fcst, all_obs):
    assert isinstance(settings, dict)
    assert all_fcst.shape == (STEPS, settings["k_ens"], N_MODEL)
    assert len(all_obs) == STEPS
    all_back_cov = np.empty((STEPS, N_MODEL, N_MODEL))
    da_sys = Da_system(settings)
    aint = AINT
    try:
        for i in trange(STEP_FREE, STEPS, desc=settings["name"], ascii=True, disable=(not sys.stdout.isatty())):
            for m in range(settings["k_ens"]):
                fcst = np.copy(all_fcst[i - 1, m, :])
                all_fcst[i, m, :] = model_step(fcst, DT)
            if i % aint == 0:
                all_back_cov[i, :, :] = get_back_cov(all_fcst[i, :, :])
                s = np.s_[i - aint + 1:i + 1]
                if settings["smoother"]:
                    anl = da_sys.analyze_one_window(all_fcst[s, :, :], all_obs[s], i, aint, True)
                    all_fcst[s, :, :] = anl
                else:
                    all_fcst[i, :, :] = da_sys.analyze_one_window(all_fcst[s, :, :], all_obs[s], i, aint, False)
    except (np.linalg.LinAlgError, ValueError) as e:
        print("ANALYSIS CYCLE DIVERGED: %s" % e)
        print("Settings: ", settings)
        traceback.print_exc()
    np.save("data/%s_cycle.npy" % settings["name"], all_fcst)
    np.save("data/%s_bcov.npy" % settings["name"], all_back_cov)
    return all_fcst

def get_back_cov(ens):
    k_ens = ens.shape[0]
    if k_ens == 1:
        return np.zeros((N_MODEL, N_MODEL))
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
