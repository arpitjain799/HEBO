# Bayesian Optimisation Framework

Only tested on Ubuntu 18.04 and python 3.8!

## Installation

Clone the repository

> git clone ssh://git@gitlab-uk.rnd.huawei.com:2222/ai-uk-team/reinforcement_learning_london/combopt.git

Change the working directory

> cd combopt/

Download all submodules

> git submodule update --init --recursive

[Optional] If you plan to use the Antibody design task, install AbsolutNoLib by following the instructions from https://github.com/csi-greifflab/Absolut

Create a virtual environment and activate it

> conda create -n bof python=3.8
> 
> conda activate bof

Install pytorch 1.10.0 with cu113 by following instructions at https://pytorch.org/get-started/locally/

Install remaining requirements with 

> pip install -r requirements.txt

Install the package itself

> pip install .

## Implemented Tasks

### Synthetic
- 21 SFU test functions
- Pest Control
- Random TSP

### Real-world
- Antibody Design (Needs to download `Absolut!` ~ 39GB)  #TODO: indicate how to install absolute
- RNA Inverse Folding
- EDA Sequence Optimisation (AIG sequence optimisation)
- EDA Sequence and Parameter Optimisation (AIG sequence and parameter optimisation)
- MIG Sequence Optimisation
- Bayesmark hyperparameter tuning tasks
- SVM hyperparameter tuning: `svm_opt`
- XG-Boost hyperparameter tuning

### How to access a task

All tasks are accessible via the `task_factory` function. Below we show how to obtain the `task`  and `search_space` class for the Antibody Design task.
```
from comb_opt.factory import task_factory

task, search_space = task_factory(task_name='antibody_design')
```
## Implemented Solvers

### BO baselines
- BOCS
- COMBO
- CoCaBO
- BOSS
- Casmopolitan
- BOiLS

### Non-BO baselines
- Random Search
- Local Search
- Simulated Annealing
- Genetic Algorithm
- Multi-Armed Bandit

## Typical use-case example
```
import torch

from comb_opt.factory import task_factory
from comb_opt.optimizers import Casmopolitan

if __name__ == '__main__':
    task, search_space = task_factory(task_name='antibody_design', dtype=torch.float32)
    optimizer = Casmopolitan(search_space, n_init=20, dtype=torch.float32, device=torch.device('cuda'))

    for i in range(100):
        x = optimizer.suggest(1)
        y = task(x)
        optimizer.observe(x, y)
        print(f'Iteration {i+1:3d}/{100:3d} - f(x) = {y:.3f} - f(x*) = {optimizer.best_y:.3f}')
```

## Contribute

### Adding a task

- The task should be a class inheriting [TaskBase](./comb_opt/tasks/task_base.py)
- Create a folder containing the code of the new task in the [tasks folder](./comb_opt/tasks).
- The folder can also contain a README providing a description of the task, associated search_space, dependencies 
- Add the new task to the [factory script](./comb_opt/factory.py) (prefer local import of the new task class if it
depends on packages not listed in [requirements.txt](./requirements.txt)).
- Add reference to the task in the present README.md in the dedicated section.

### Adding an optimizer

\#TODO