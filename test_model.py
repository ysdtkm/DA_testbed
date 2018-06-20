#!/usr/bin/env python

import unittest
import numpy as np
from const import N_MODEL
from model import Model

class TestModel(unittest.TestCase):
    def test_model(self):
        x = np.random.randn(N_MODEL)
        dt = 0.01
        m = Model()
        for i in range(100):
            x = m.rk4(x, dt)
        self.assertEqual(x.shape, (N_MODEL,))
