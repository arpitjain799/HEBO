import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(os.path.realpath(__file__)).parent.parent.parent))

from comb_opt.factory import task_factory
from comb_opt.optimizers import RandomSearch, LocalSearch, SimulatedAnnealing, PymooGeneticAlgorithm, BOCS, BOSS, COMBO, \
    CoCaBO, Casmopolitan, BOiLS
from comb_opt.utils.experiment_utils import run_experiment

if __name__ == '__main__':
    task_name = 'bayesmark'
    dtype = torch.float32
    bo_n_init = 20
    bo_device = torch.device('cuda:1')
    max_num_iter = 200
    random_seeds = [42, 43, 44, 45, 46]

    for model_name in ["MLP-adam", "lasso", "linear"]:
        for database_id in ["boston", "diabetes"]:
            task_kwargs = {'model_name': model_name, "metric": "mse", "database_id": database_id}

            task, search_space = task_factory(task_name, dtype, **task_kwargs)

            rs_optim = RandomSearch(search_space=search_space, dtype=dtype)
            ls_optim = LocalSearch(search_space=search_space, dtype=dtype)
            sa_optim = SimulatedAnnealing(search_space=search_space, dtype=dtype)
            casmopolitan = Casmopolitan(search_space=search_space, n_init=bo_n_init, dtype=dtype, device=bo_device)

            optimizers = [
                # casmopolitan,
                rs_optim,
                # ls_optim,
                # sa_optim,
            ]

            run_experiment(task=task, optimizers=optimizers, random_seeds=random_seeds, max_num_iter=max_num_iter,
                           very_verbose=True)
