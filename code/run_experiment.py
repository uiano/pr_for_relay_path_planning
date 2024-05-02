# Python 3.6
#

import sys
import os
from IPython.core.debugger import set_trace
import importlib


def initialize():
    if not os.path.exists("./gsim"):
        # If this behavior were changed, the output file storage functionality should be
        # modified accordingly.
        print(
            "Error: `run_experiment` must be invoked from the folder where it is defined"
        )
        quit()

    sys.path.insert(1, './gsim/')


initialize()

import gsim
import gsim_conf


def load_modules():
    print("Loading modules...", end=' ')
    module = importlib.import_module(gsim_conf.module_name)
    ExperimentSet = getattr(module, "ExperimentSet")
    print("done")
    return ExperimentSet


########################################################################

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description=
        "Run the experiment function with index <experiment_index> in the module specified in gsim_conf.py."
    )
    parser.add_argument('experiment_index')
    parser.add_argument(
        'experiment_args',
        nargs='*',
        default=[],
        help='list of arguments passed to the experiment function')
    parser.add_argument(
        '-p',
        '--plot_only',
        help=
        'plot the results stored from the last execution (the experiment will not be run).',
        action="store_true")
    parser.add_argument('-e',
                        '--export',
                        help='export the figures as PDF.',
                        action="store_true")
    parser.add_argument(
        '-i',
        '--inspect',
        help=
        'load the stored figures and open a pdb prompt to inspect the GFigure objects.',
        action="store_true")

    parser.add_argument('-g', '--gpu', help='Select the GPU.', default=None)

    args, unknown_args = parser.parse_known_args()
    ExperimentSet = load_modules()
    if len(unknown_args):
        print('WARNING: The following arguments were not recognized:')
        print(unknown_args)
    if len(args.experiment_args):
        if args.plot_only:
            print(
                "WARNING: the following experiment arguments were specified but will not be passed to the experiment since it will not be run."
            )
        else:
            print("The following arguments will be passed to the experiment:")
        print(args.experiment_args)

    if args.plot_only:
        ExperimentSet.plot_only(args.experiment_index,
                                save_pdf=args.export,
                                inspect=args.inspect)
    else:
        if args.gpu is not None:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        ExperimentSet.run_experiment(args.experiment_index,
                                     args.experiment_args,
                                     save_pdf=args.export,
                                     inspect=args.inspect)

    def set_permisions_recursively(folder):
        for root, dirs, files in os.walk(folder):
            for d in dirs:
                os.chmod(os.path.join(root, d), 16895)
            for f in files:
                os.chmod(os.path.join(root, f), 33279)

    try:
        set_permisions_recursively(gsim.OUTPUT_DATA_FOLDER)
    except Exception as e:
        print(e)
        print(
            f"The permisions of the files and folders in {gsim.OUTPUT_DATA_FOLDER} could not be properly set, possibly because of your operating system. Please do not use the script sync_data until this issue is fixed"
        )
else:
    ExperimentSet = load_modules()
###### Python API ######
""" 
To run an experiment from the iPython shell:

%load_ext autoreload
%autoreload 2
from run_experiment import run
run(1001)

"""


def run(name, *args, **kwargs):
    ExperimentSet.run_experiment(name, *args, **kwargs)
