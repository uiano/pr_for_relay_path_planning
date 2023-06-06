# probabilistic_roadmaps_for_relay_path_planning
This repository contains an implementation of the PRFI algorithm as well as the code to reproduce the numerical experiments of the paper "Probabilistic Roadmaps for Flying Relay Path Planning" by Pham Q. Viet and Daniel Romero.

After cloning the repository, do the following:

```
cd gsim
git submodule init
git submodule update
cd ..
bash gsim/install.sh

cd common
python grid_utilities_setup.py build
python grid_utilities_setup.py install
cd ..
```
You may need to install a compiler and some Python packages.

To run the simulations, type

```
python run_experiment.py M
```

where M is 1034 for Fig. 3 and Fig. 4, 1005 for Fig. 5. 

The code of the experiments can be found in experiments/paper_experiments.py. 

More information on the simulation environment [here](https://github.com/fachu000/GSim-Python).

