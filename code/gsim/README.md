# GSim-Python

General-purpose simulation environment for Python.

To install GSim into your git repository, cd to the folder of your repository and type

```
$ git submodule add https://github.com/fachu000/GSim-Python.git ./gsim
$ bash gsim/install.sh
```

Create your experiments as functions in a module in the folder
Experiments. There is an example of such a module there.

To run experiment 1002 in experiments.example_experiments, type

```bash
$ python run_experiment.py 1002
```

Then you can create a new experiment file in the folder Experiments with the same structure as `example_experiments`. Remember to set the variable `module_name` in `gsim_conf.py` accordingly.

The full potential of GSim is exploited when the experiment functions
return GFigures rather than directly plotting figures. GFigures are
stored and can be plotted and edited afterwards without having to run
again the experiment.

For example, to see the figures created last time experiment 1002 was
run but without running it again, type

```
$ python run_experiment.py -p 1002
```

GFigure offers a neater interface than matplotlib, whose goal was to
resemble the interface of MATLAB. Figures are collections of
subfigures, subfigures are collections of curves. Simple to
understand. See the description at the top of `gsim.gfigure` for a small tutorial on
how to use GFigure.

Pull requests are welcome!!

## Cloning a repository that contains GSim

Fast way:
```
$ git clone --recurse-submodules <url_of_the_repo>
$ cd <folder_of_the_repo>
$ bash gsim/install.sh
```

If you have already cloned the repository and forgot the `--recurse-submodules`, just cd to the main folder of the cloned repo and type
```
$ cd gsim
$ git submodule init
$ git submodule update
$ cd ..
$ bash gsim/install.sh
```

## Visual Studio Code

For debugging, it is comfortable to use the following `launch.json`:

```js
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "gsim",
            "type": "python",
            "program": "${workspaceFolder}/run_experiment.py",
            "args": [
                "1001"
            ],
            "request": "launch",
            "justMyCode": false,
        }
    ]
}
```

Remember to replace `1001` with the number of the experiment that you want to run.
