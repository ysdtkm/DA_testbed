#!/usr/bin/env python

import numpy as np
from const import N_MODEL
import letkf

class Da_system:
    def __init__(self, settings):
        assert isinstance(settings, dict)
        # self.settings = settings
        self.method = settings["method"]
        self.rho = settings["rho"]
        self.k_ens = settings["k_ens"]
        self.l_loc = settings["l_loc"]

    def analyze_one_window(self, fcst, obs):
        assert fcst.shape == (self.k_ens, N_MODEL)
        if self.method == "letkf":
            assert self.k_ens > 1
            anl = letkf.letkf(fcst, obs, self.rho, self.l_loc)
        elif self.method == "3dvar":
            assert self.k_ens == 1
            raise Exception("da_system.py: 4D-Var is not implemented")
        elif self.method == "4dvar":
            assert self.k_ens == 1
            raise Exception("da_system.py: 4D-Var is not implemented")
        else:
            raise Exception(f"analysis method mis-specified: {self.settings['method']}")
        assert anl.shape == fcst.shape
        return anl[:, :]

