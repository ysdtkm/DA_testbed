#!/usr/bin/env python

import numpy as np
from const import N_MODEL, P_OBS
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
            anl = letkf.letkf(fcst, obs, self.rho, self.l_loc)
        else:
            raise Exception("analysis method mis-specified: %s" % self.settings["method"])
        assert anl.shape == fcst.shape
        return anl[:, :]

