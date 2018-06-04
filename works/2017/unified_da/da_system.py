#!/usr/bin/env python

from itertools import chain
import numpy as np
from const import N_MODEL, AINT
import letkf

class Da_system:
    def __init__(self, settings):
        assert isinstance(settings, dict)
        self.method = settings["method"]
        self.rho = settings["rho"]
        self.k_ens = settings["k_ens"]
        self.l_loc = settings["l_loc"]

    def analyze_one_window(self, fcst, obs, t_anl):
        assert fcst.shape == (AINT, self.k_ens, N_MODEL)
        if self.method == "letkf":
            assert self.k_ens > 1
            anl = letkf.letkf(fcst, list(chain.from_iterable(obs)), self.rho, self.l_loc, t_anl)
        elif self.method == "3dvar":
            assert self.k_ens == 1
            raise Exception("da_system.py: 4D-Var is not implemented")
        elif self.method == "4dvar":
            assert self.k_ens == 1
            raise Exception("da_system.py: 4D-Var is not implemented")
        else:
            raise Exception(f"analysis method mis-specified: {self.settings['method']}")
        assert anl.shape == (self.k_ens, N_MODEL)
        return anl

