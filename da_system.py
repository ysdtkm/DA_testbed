#!/usr/bin/env python

from itertools import chain
import numpy as np
from const import N_MODEL
from letkf import letkf
from ensrf import ensrf_all
from fdvar import fdvar
from tdvar import tdvar

class Da_system:
    def __init__(self, settings):
        assert isinstance(settings, dict)
        self.method = settings["method"]
        self.k_ens = settings["k_ens"]
        if self.k_ens == 1:
            self.amp_b = settings["amp_b"]
        else:
            self.rho = settings["rho"]
            self.l_loc = settings["l_loc"]

    def analyze_one_window(self, fcst, olist, t_anl, aint):
        assert fcst.shape == (aint, self.k_ens, N_MODEL)
        olist = list(chain.from_iterable(olist))
        if self.method == "letkf":
            assert self.k_ens > 1
            anl = letkf(fcst, olist, self.rho, self.l_loc, t_anl, aint)
        elif self.method == "ensrf":
            assert self.k_ens > 1
            anl = ensrf_all(fcst, olist, self.rho, t_anl, aint)
        elif self.method == "tdvar":
            assert self.k_ens == 1
            anl = tdvar(fcst[0, 0, :], olist, self.amp_b, t_anl)[None, :]
        elif self.method == "fdvar":
            assert self.k_ens == 1
            anl = fdvar(fcst[0, 0, :], olist, self.amp_b, t_anl, aint)[None, :]
        else:
            raise Exception(f"analysis method mis-specified: {self.settings['method']}")
        assert anl.shape == (self.k_ens, N_MODEL)
        return anl

