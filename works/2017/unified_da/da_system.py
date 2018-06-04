#!/usr/bin/env python

import numpy as np
from const import N_MODEL, P_OBS
import letkf

class Da_system:
    def __init__(self):
        pass

    def analyze_one_window(self, fcst, obs, settings):
        assert isinstance(settings, dict)
        assert fcst.shape == (settings["k_ens"], N_MODEL)
        assert obs.shape == (P_OBS,)

        if settings["method"] == "letkf":
            anl = letkf.letkf(fcst, obs, settings["rho"], settings["k_ens"], settings["l_loc"])
            assert anl.shape == (settings["k_ens"], N_MODEL)
        else:
            raise Exception("analysis method mis-specified: %s" % settings["method"])
        return anl[:, :]

