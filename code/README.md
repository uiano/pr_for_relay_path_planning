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

Other Python packages may be required. To run the simulations, type

```
python run_experiment.py <experiment_id>
```

where `experiment_id` is the ID of an experiment, e.g. 6725, 6739, etc.
