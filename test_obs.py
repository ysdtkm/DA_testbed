#!/usr/bin/env python

import unittest
import numpy as np
from const import N_MODEL
from obs import generate_single_obs, Scaler_obs

class TestObs(unittest.TestCase):
    def test_single_obs(self):
        dt_obs = 2
        x = np.random.randn(dt_obs, N_MODEL)
        position = np.random.randint(N_MODEL // 2)
        sigma_r = 1.0
        time = 10
        obs = generate_single_obs(x, position, sigma_r, time, dt_obs)
        self.assertIsInstance(obs, Scaler_obs)

