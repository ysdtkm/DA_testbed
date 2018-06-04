#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import os
from const import N_MODEL, STEPS, OERR, EXPLIST, AINT

mpl.use('Agg')
import matplotlib.pyplot as plt

rmse_hash = {}

def plot_all():
    os.system("mkdir -p image/true")

    hist_true = np.load("data/true.npy")
    plot_true(hist_true)

    for exp in EXPLIST:
        name = exp["name"]
        k_ens = exp["k_ens"]
        os.system("mkdir -p image/%s" % name)

        hist_fcst = np.load("data/%s_cycle.npy" % name)
        back_cov = np.load("data/%s_bcov.npy" % name)

        plot_rmse_spread(hist_true, hist_fcst, name, k_ens)
        plot_time_value(hist_true, hist_fcst, name)
        plot_diff_ens_mean(hist_true, hist_fcst, name)
        plot_cov_corr(back_cov, name, k_ens)

def plot_rmse_spread(hist_true, hist_fcst, name, k_ens):
    # Error and Spread_square for each grid and time
    hist_fcst_mean = np.mean(hist_fcst, axis=1)
    hist_fcst_sprd2 = np.zeros((STEPS, N_MODEL))
    if k_ens > 1:
        for i in range(k_ens):
            hist_fcst_sprd2[:, :] = hist_fcst_sprd2[:, :] + \
                                    1.0 / (k_ens - 1.0) * (hist_fcst[:, i, :] ** 2 - hist_fcst_mean[:, :] ** 2)
    hist_err = hist_fcst_mean - hist_true

    global rmse_hash

    # MSE and Spread_square time series (grid average)
    mse_time = np.mean(hist_err[:, :] ** 2, axis=1)
    sprd2_time = np.mean(hist_fcst_sprd2[:, :], axis=1)

    # RMSE and Spread time average
    rmse = np.sqrt(np.mean(mse_time[STEPS // 2:]))
    sprd = np.sqrt(np.mean(sprd2_time[STEPS // 2:]))

    # RMSE-Spread time series
    plt.rcParams["font.size"] = 14
    plt.yscale('log')
    plt.plot(np.sqrt(mse_time), label="RMSE")
    if k_ens > 1:
        plt.plot(np.sqrt(sprd2_time), label="Spread")
    plt.axhline(y=OERR, label="obs error", alpha=0.5, color="black")
    plt.legend()
    plt.xlabel("timestep")
    plt.ylim(0.01, 10.0)
    plt.title("[%s] RMSE:%.3g Spread:%.3g" % (name, rmse, sprd))
    plt.savefig("./image/%s/time.pdf" % name)
    plt.clf()
    plt.close()

def plot_time_value(hist_true, hist_fcst, name):
    hist_fcst_mean = np.mean(hist_fcst, axis=1)

    plt.rcParams["font.size"] = 12
    fig, ax1 = plt.subplots(1)
    ax1.set_title(name)
    ax1.plot(hist_true[:, 0], label="true")
    ax1.plot(hist_fcst_mean[:, 0], label="DA cycle")
    ax1.set_ylabel("0th element")
    ax1.legend(loc="upper right")
    plt.xlabel("timestep")
    plt.savefig("./image/%s/%s.pdf" % (name, "val"))
    plt.clf()
    plt.close()

def plot_true(hist_true):
    cm = plt.imshow(hist_true, aspect="auto")
    cm.set_clim(-10, 15)
    cb = plt.colorbar(cm)

    plt.xlabel("grid")
    plt.ylabel("time [step]")
    plt.title("true")
    plt.savefig("./image/true/true.pdf")
    plt.close()

def plot_diff_ens_mean(hist_true, hist_fcst, name):
    hist_fcst_mean = np.mean(hist_fcst, axis=1)
    error = hist_fcst_mean - hist_true

    cm = plt.imshow(error, cmap=plt.cm.RdBu_r, aspect="auto")
    cm.set_clim(-1.0, 1.0)
    cb = plt.colorbar(cm)

    plt.xlabel("grid")
    plt.ylabel("time [step]")
    plt.title("analysis error")
    plt.savefig("./image/%s/error.pdf" % name)
    plt.close()

def plot_cov_corr(back_cov, name, k_ens):
    def reduce(back_cov):
        sum_cov = np.zeros((N_MODEL, N_MODEL))
        sum_corr = np.zeros((N_MODEL, N_MODEL))
        sum_corr2 = np.zeros((N_MODEL, N_MODEL))
        count = 0
        for t in range(STEPS // 2, STEPS):
            if t % AINT != 0:
                continue
            cov = back_cov[t, :, :]
            corr = cov_to_corr(cov)
            sum_cov += cov
            sum_corr += corr
            sum_corr2 += corr ** 2
            count += 1
        ave_cov = sum_cov / count
        ave_corr = sum_corr / count
        rms_corr = (sum_corr2 / count) ** 0.5
        return ave_cov, ave_corr, rms_corr

    def cov_to_corr(cov):
        corr = cov.copy()
        for i in range(N_MODEL):
            for j in range(N_MODEL):
                corr[i, j] /= np.sqrt(cov[i, i] * cov[j, j])
        return corr

    def plot_matrix(mat, out, title, cmax=None):
        plt.rcParams["font.size"] = 14
        fig, ax = plt.subplots(1)
        if cmax is None:
            cmax = np.max(np.abs(mat))
        cm = ax.imshow(mat, cmap=plt.cm.RdBu_r)
        cm.set_clim(-1.0 * cmax, cmax)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        plt.colorbar(cm)
        plt.title(title)
        plt.savefig(out)
        plt.close()

    def plot_corr_homo(ave_corr, rms_corr, out, k_ens):
        sum_ave_corr = np.zeros(N_MODEL)
        sum_rms_corr = np.zeros(N_MODEL)
        for i in range(N_MODEL):
            sum_ave_corr += np.roll(ave_corr[i], - i)
            sum_rms_corr += np.roll(rms_corr[i], - i)
        sum_ave_corr /= N_MODEL
        sum_rms_corr /= N_MODEL
        xr = np.linspace(-N_MODEL // 2, N_MODEL // 2 - 1, N_MODEL)
        plt.plot(xr, np.roll(sum_ave_corr, N_MODEL // 2), label="mean corr")
        plt.plot(xr, np.roll(sum_rms_corr, N_MODEL // 2), label="RMS corr")
        pseudo_rms_corr = 1.0 / (k_ens - 1) ** 0.5
        plt.axhline(y=0, alpha=0.5, color="black")
        plt.axhline(y=pseudo_rms_corr, alpha=0.5, color="red", label="RMS pseudo corr")
        plt.legend()
        plt.xlabel("distance")
        plt.title("spatial mean background correlation")
        plt.savefig(out)
        plt.close()

    ave_cov, ave_corr, rms_corr = reduce(back_cov)
    plot_matrix(ave_cov, "./image/%s/b_cov.pdf" % name, "background covariance")
    plot_matrix(ave_corr, "./image/%s/b_corr.pdf" % name, "mean background correlation", 1.0)
    plot_matrix(rms_corr, "./image/%s/b_rms_corr.pdf" % name, "RMS background correlation", 1.0)
    plot_corr_homo(ave_corr, rms_corr, "./image/%s/b_corr_homo.pdf" % name, k_ens)

if __name__ == "__main__":
    plot_all()
