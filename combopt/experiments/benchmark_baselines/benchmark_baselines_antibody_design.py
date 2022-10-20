import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(os.path.realpath(__file__)).parent.parent.parent))

from comb_opt.factory import task_factory
from comb_opt.optimizers import RandomSearch, LocalSearch, SimulatedAnnealing, BOCS, BOSS, COMBO, \
    BOiLS, MultiArmedBandit, GeneticAlgorithm, Casmopolitan
from comb_opt.utils.experiment_utils import run_experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True, description='BOF - Baselines - Antibody design')
    parser.add_argument("--device_id", type=int, default=0, help="Cuda device id (cpu is used if id is negative)")
    parser.add_argument("--absolut_dir", type=str, default=None, help="Path to AbsolutNoLib")

    args = parser.parse_args()

    task_name = 'antibody_design'
    task_kwargs = {'num_cpus': 5, 'first_cpu': 0, 'absolut_dir': args.absolut_dir}
    dtype = torch.float32
    task, search_space = task_factory(task_name, dtype, **task_kwargs)

    bo_n_init = 20
    if args.device_id >= 0 and torch.cuda.is_available():
        bo_device = torch.device(f'cuda:{args.device_id}')
    else:
        bo_device = torch.device("cpu")

    max_num_iter = 200
    random_seeds = [42, 43, 44, 45, 46]

    rs_optim = RandomSearch(search_space=search_space, dtype=dtype)
    ls_optim = LocalSearch(search_space=search_space, dtype=dtype)
    sa_optim = SimulatedAnnealing(search_space=search_space, dtype=dtype)
    ga_optim = GeneticAlgorithm(search_space=search_space, dtype=dtype)
    bocs = BOCS(search_space=search_space, n_init=bo_n_init, dtype=dtype, device=bo_device)
    boss = BOSS(search_space=search_space, n_init=bo_n_init, dtype=dtype, device=bo_device)
    combo = COMBO(search_space=search_space, n_init=bo_n_init, dtype=dtype, device=bo_device)
    boils = BOiLS(search_space=search_space, n_init=bo_n_init, model_max_batch_size=50, dtype=dtype, device=bo_device)
    casmopolitan = Casmopolitan(search_space=search_space, n_init=bo_n_init, dtype=dtype, device=bo_device)
    mab_optim = MultiArmedBandit(search_space=search_space, batch_size=1, max_n_iter=200, noisy_black_box=False,
                                 dtype=dtype)

    optimizers = [
        # casmopolitan,
        # boss,
        # boils,
        # combo,
        # bocs,
        # mab_optim,
        # rs_optim,
        # ls_optim,
        # sa_optim,
        # ga_optim
    ]

    run_experiment(task=task, optimizers=optimizers, random_seeds=random_seeds, max_num_iter=max_num_iter,
                   very_verbose=True)
