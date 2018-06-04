# unified_da

## How to use
```bash
make
evince image/merged.pdf
```

## Dependency of modules
<img src="documentation/graph.png">

* main: run nature run, observation, and DA cycle. Output is saved at data/
* letkf: LETKF core programs
* model: model dependent functions including observation
* plot: all visualization functions. Read data/ and write image/
* const: Run-time settings

## Todo
* Better modularbility by OOP
