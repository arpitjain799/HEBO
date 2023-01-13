import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(os.path.realpath(__file__)).parent.parent.parent))

from comb_opt.factory import task_factory
from comb_opt.optimizers import RandomSearch, LocalSearch, SimulatedAnnealing, Casmopolitan, \
    CoCaBO, GeneticAlgorithm
from comb_opt.utils.experiment_utils import run_experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True, description='BOF - Mix And Match - RNA folding')
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="Seeds to run")
    args = parser.parse_args()

    task_name = 'aig_optimization_hyp'
    task_kwargs = {'designs_group_id': "sin", "operator_space_id": "basic", "objective": "both",
                   "seq_operators_pattern_id": "basic_w_post_map"}
    dtype = torch.float32
    task, search_space = task_factory('aig_optimization_hyp', dtype, **task_kwargs)

    bo_n_init = 20
    bo_device = torch.device('cuda:2')
    max_num_iter = 200
    random_seeds = args.seeds
    rs_optim = RandomSearch(search_space=search_space, dtype=dtype)
    ls_optim = LocalSearch(search_space=search_space, dtype=dtype)
    sa_optim = SimulatedAnnealing(search_space=search_space, dtype=dtype)
    ga_optim = GeneticAlgorithm(search_space=search_space, dtype=dtype)
    casmopolitan = Casmopolitan(search_space=search_space, n_init=bo_n_init, dtype=dtype, device=bo_device)
    cocabo = CoCaBO(search_space=search_space, n_init=bo_n_init)

    optimizers = [
        # casmopolitan,
        # cocabo,
        # rs_optim,
        # ls_optim,
        # sa_optim,
        ga_optim,
    ]

    run_experiment(task=task, optimizers=optimizers, random_seeds=random_seeds, max_num_iter=max_num_iter,
                   very_verbose=0)
