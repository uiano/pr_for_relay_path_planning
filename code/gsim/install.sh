#!/bin/bash


function ignore_file {
    # File added to .gitignore if it is not already ignored. We also check
    # whether its name is already in .gitignore since the file may be tracked. 
    git ls-files . --ignored --exclude-standard --others | grep -q $1
    IS_NOT_IGNORED=$?
    more ".gitignore" | grep -q $1  
    IS_NOT_IN_GITIGNORE=$?  
    if [ $IS_NOT_IGNORED -eq 0 ] || [ $IS_NOT_IN_GITIGNORE -eq 0 ] ;    
    then
        echo "The file $1 is already ignored by .gitignore";
    else
        echo "Adding $1 to .gitignore...";
        echo -e "\n\n# Added by GSim\n$1" >> .gitignore
    fi
}

if [ ! -f "gsim/install.sh" ]
then
    echo "ERROR: You must enter the root folder of your repository before executing this file."
    echo " Synopsis:"
    echo "    $ bash gsim/install.sh  "
    exit 1
fi

INSTALLATION_FOLDER="installation_files"

GSIM_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/"

# ACK: https://patorjk.com/software/taag
echo ' '
echo '-----------------------------------------------------------------'
echo '   ___________ _                 ____        __  __              '
echo '  / ____/ ___/(_)___ ___        / __ \__  __/ /_/ /_  ____  ____ '
echo ' / / __ \__ \/ / __ `__ \______/ /_/ / / / / __/ __ \/ __ \/ __ \ '
echo '/ /_/ /___/ / / / / / / /_____/ ____/ /_/ / /_/ / / / /_/ / / / /'
echo '\____//____/_/_/ /_/ /_/     /_/    \__, /\__/_/ /_/\____/_/ /_/ '
echo '                                   /____/                        '
echo '                                                                 '
echo "             https://github.com/fachu000/GSim-Python             "
echo '-----------------------------------------------------------------'
echo ' '
echo "Welcome to GSim-Python"

echo " "
echo "Installing GSim-Python..."

# The following sets git to sync the submodules when doing `git pull`
git config submodule.recurse true

# run_experiment.py
RUN_EXPERIMENT="run_experiment.py"
if [ ! -f $RUN_EXPERIMENT ]
then
    echo "Copying "$GSIM_DIR$INSTALLATION_FOLDER"/run_experiment.py as "$RUN_EXPERIMENT
    cp -n $GSIM_DIR$INSTALLATION_FOLDER"/run_experiment.py" $RUN_EXPERIMENT
    ignore_file $RUN_EXPERIMENT
else
    echo "File $RUN_EXPERIMENT already exists."
fi


# gsim_conf.py
#
# This file is intended to be ignored since it may be used to contain
# user-specific settings. 
#
# If a file with name $GSIM_CONF_PROJ exists in the root folder of the parent
# repository, it is used to create the `gsim_conf.py` file. This is useful in
# the case where we want to distribute a repository -- the user must be able to
# run an experiment from the desired experiment file without modifying the
# variable `module_name` in gsim_conf.py, as it would be required in case that
# we copied directly from $GSIM_CONF_DEF. 
# 
# If no such a file exists, then we use the default one $GSIM_CONF_DEF. 
#
# that is that the user needs to run this script also when cloning. This means
# that when we distribute a repo, the user will also need to change the line
# that selects the experiment module in gsim_conf.py. 

if [ ! -f "gsim_conf.py" ]
then
    GSIM_CONF_PROJ=".gsim_conf_default.py" # Project specific
    GSIM_CONF_DEF=$GSIM_DIR$INSTALLATION_FOLDER"/gsim_conf-base.py" # GSim template
    if [ ! -f $GSIM_CONF_PROJ ];
    then 
        echo "No previous default configuration file $GSIM_CONF_PROJ found."
        GSIM_CONF_BASE=$GSIM_CONF_DEF
    else
        echo "Default configuration file $GSIM_CONF_PROJ found."
        GSIM_CONF_BASE=$GSIM_CONF_PROJ        
    fi

    echo "Copying file $GSIM_CONF_BASE to gsim_conf.py"
    cp -n $GSIM_CONF_BASE "gsim_conf.py"
    ignore_file "gsim_conf.py"
else
    echo "File gsim_conf.py already exists."
fi

EXAMPLE_EXPERIMENTS="experiments/example_experiments.py"
if [ ! -f $EXAMPLE_EXPERIMENTS ]
then
    mkdir -p experiments
    echo "Copying "$GSIM_DIR$INSTALLATION_FOLDER"/example_experiments.py to"$EXAMPLE_EXPERIMENTS
    cp -n $GSIM_DIR$INSTALLATION_FOLDER"/example_experiments.py" $EXAMPLE_EXPERIMENTS
else
    echo "File $EXAMPLE_EXPERIMENTS already exists."
fi

echo -e "Done.\n"

echo "You can now run your experiment as"
echo "$ python run_experiment.py <experiment_number>"
echo ""