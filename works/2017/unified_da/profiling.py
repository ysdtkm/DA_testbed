#!/usr/bin/env python

import main, letkf
import line_profiler

pr = line_profiler.LineProfiler()
pr.add_function(letkf.letkf)
pr.runcall(main.main)
pr.print_stats()

