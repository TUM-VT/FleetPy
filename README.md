# FleetPy
Simulation framework to model and control the tasks of vehicle fleets (routing, user-assignment, charging, ...)

## Features

* Agent-based simulations of dynamic vehicle routing problems

* Time- and event-based user-operator interaction models

* Simulation contains information flow, i.e. request – offer – accept/reject

* Various request-acceptance models, e.g. ‘always accept’, ‘accept if hard constraints are satisfied’, ‘probabilistic sensitivity on offer parameters’, or even ‘probabilistic based on mode-choice model’ that considers also other mode alternatives

* Several fleet control strategies to assign requests, pool and assign requests, reposition vehicles, re-charge electric vehicles

* Different underlying routing algorithms, choice can be made based on trade-off between memory requirement and computation time

* Modular approach enables combination of different request models, network representations, fleet control strategies


## Dependency installation

### Conda workflow for virtual environment
Prerequisite: anaconda installed; Example in this instruction: 
**Anaconda version: 201903; conda version: 4.9.2**

After installing Anaconda, we could open the *Anaconda Prompt* to execute the following codes. First, let's change the working directory.
```
cd <working_directory>
```

#### Control the channel priority
It is strongly recommended by [GeoPandas](https://geopandas.org/install.html "geopandas_installation") to either install everything from the defaults channel, or everything from the *conda-forge* channel. Ending up with a mixture of packages from both channels for the dependencies of *GeoPandas* can lead to importing problems.

In this instruction, we choose the channel *conda-forge*.To achieve this, first, we could add the channel *conda-forge* by
```
conda config --env --add channels conda-forge
```

You should check all your channels by:

```
conda config --show channels
```

If *conda-forge* is not on top of your list, please add the channel once more to put it on top of the list.


To restrict the channel, use the following code:
```
conda config --env --set channel_priority strict
```

Which basically installs packages with same names strictly by channel priority, and as it is on top of the list, the higher channel is the *conda-forge*. To check more detailed information about the channel priority, input the following code:

```
conda config --describe channel_priority
```

#### Create the virtual environment
Create a new virtual environment in an *Anaconda Prompt* with all the packages listed in `environment.yml`. The default name of the environment is called `fleetpy`, which you could also change in the first line of `environment.yml` by changing the `name` variable.

```
conda env create -f environment.yml
```

Check out the created virtual environment.

```
conda activate <new_env>
```

Now check the installed packages :wink:

```
conda list
```

Everything is set up! :thumbsup: Now you could run your first simulation!

### C++ Router
We recommend to use the C++ router unless your network is small enough to preprocess the complete travel time matrix.
If you want to use the C++ router, you need to have a C++ compiler and Cython set up on your computer.

Moreover, you need to compile the module on your system. Please go to 

```
cd FleetPy/src/routing/cpp_router
```

Next, you can install the C++ router:

```
python setup.py build_ext --inplace
```

<!-- waiting for Roman and Yunfei to supplement -->

### Optimizer

* Gurobi:
Set gurobi channel on top of your channel list by twice calling
```
conda config --add channels http://conda.anaconda.org/gurobi
```
Install gurobi package by
```
conda install gurobi
```
Free academic licenses of Gurobi can be acquired. See https://www.gurobi.com/academia/academic-program-and-licenses/ for more details in installation instructions.

<!-- waiting for Yunfei to supplement; check the packages gurobi and cplex -->


## Data Preparation and Study Setup
For now, you can inspect the data structures and files in the examples provided in github:
* FleetPy/data
* FleetPy/studies

More detailed descriptions of the data structure, preprocessing steps, and result data will be provided in the next versions.
Additionally, a GUI to set up scenarios (with choice of submodules and data) is planned for the future.

In general, you can save your data and study definitions in the mentioned directories. These are included in .gitignore.

<!-- ... (prepare study by config.csv and scenarios.csv) -->
<!-- ... (necessary modules for preprocessing can be installed by "pip3 install -r requirements_with_pp.txt") -->

<!-- more detailed description to follow -->

## Usage

### Simulation
You can test the example scenarios provided in the github repository by calling

```
python3 run_examples.py
```

You should now have a directory containing several simulated scenarios. In each of the scenario directories, you will have a summary of the configuration, a simulation log file and several output files.
* FleetPy/studies/example_study/results
The output files will be described in more detail in future versions; a very brief description for now:
* 1_user_stats.csv contains user data for every single request
* 2_0_op-stats.csv contains vehicle data for every task of the fleet vehicles of operator 0

By default, a standard evaluation aggregating several results is performed after every simulation.
* standard_eval.csv
Additionally, you can create a few temporal evaluations by calling
```
python3 FleetPy/src/evaluation/temporal.py scenario_result_directory
```


When you want to run your own scenarios, please create a copy of this script file, denote it 'run_private_[XYZ].py' with any study indicator XYZ, and modify the part following

```
if __name__ == '__main__'
```

'run_private\*.py' is included in .gitignore.


### Watch Replay
If you want to see a visualization of an already simulated scenario, you can call

```
python3 replay_pyplot.py scenario_result_directory sim_seconds_per_real_second
```

The start time of the replay can be given as an additional optional input parameter.

<!-- waiting for GUI Scenario Creator for further information -->


## Tested on

Windows 10 Pro x64
Chrome 79.0.3945
Python 3.7