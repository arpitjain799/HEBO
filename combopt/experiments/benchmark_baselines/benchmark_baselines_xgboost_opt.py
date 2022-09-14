import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(os.path.realpath(__file__)).parent.parent.parent))

from comb_opt.factory import task_factory
from comb_opt.optimizers import RandomSearch, LocalSearch, SimulatedAnnealing, Casmopolitan, CoCaBO
from comb_opt.utils.experiment_utils import run_experiment

if __name__ == '__main__':
    task_name = 'xgboost_opt'
    dataset_id = "mnist"
    dtype = torch.float32

    task, search_space = task_factory(task_name, dtype=dtype, dataset_id=dataset_id)
    bo_n_init = 20
    bo_device = torch.device('cuda:2')
    max_num_iter = 200
    random_seeds = [42, 43, 44, 45, 46]

    rs_optim = RandomSearch(search_space=search_space, dtype=dtype)
    ls_optim = LocalSearch(search_space=search_space, dtype=dtype)
    sa_optim = SimulatedAnnealing(search_space=search_space, dtype=dtype)
    casmopolitan = Casmopolitan(search_space=search_space, n_init=bo_n_init, dtype=dtype, device=bo_device)
    cocabo = CoCaBO(search_space=search_space, n_init=bo_n_init)

    optimizers = [
        # casmopolitan,
        # rs_optim,
        # ls_optim,
        # sa_optim,
        cocabo
    ]

    run_experiment(task=task, optimizers=optimizers, random_seeds=random_seeds, max_num_iter=max_num_iter,
                   very_verbose=False)
