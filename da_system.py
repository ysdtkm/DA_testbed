#!/usr/bin/env python

from itertools import chain
import numpy as np
from const import N_MODEL
from letkf import letkf

class Da_system:
    def __init__(self, settings):
        assert isinstance(settings, dict)
        self.method = settings["method"]
        self.rho = settings["rho"]
        self.k_ens = settings["k_ens"]
        self.l_loc = settings["l_loc"]
        self.amp_b = settings["amp_b"]

    def analyze_one_window(self, fcst, obs, t_anl, aint, smoother=False):
        assert fcst.shape == (aint, self.k_ens, N_MODEL)
        obs = list(chain.from_iterable(obs))
        if self.method == "letkf":
            assert self.k_ens > 1
            anl = letkf(fcst, obs, self.rho, self.l_loc, t_anl, aint, smoother)
        else:
            raise Exception(f"analysis method mis-specified: {self.settings['method']}")
        assert anl.shape == ((aint, self.k_ens, N_MODEL) if smoother else (self.k_ens, N_MODEL))
        return anl

