# DA_testbed for quick prototyping

## How to use
* This repository should only have core features. Each experiments should be conducted after folking/branching this repository.

```bash
make
evince image/merged.pdf
```

## Dependency of modules
<img src="documentation/graph.png">

* main: run nature run, observation, and DA cycle. Output is saved at **data/**
* da_system: wrapper for DA algorithms. One DA window can take observations at t in (t_s, t_e].
* letkf: LETKF core programs
* fdvar: 4D-Var core programs
* model: model integration and TLM
* obs: observation generation and xb -> yb transform.
    * class Scaler_obs allows flexible observation networks, localization, and 4D-implementation
* plot: all visualization functions. Read **data/** and write **image/**
* const: Run-time settings

