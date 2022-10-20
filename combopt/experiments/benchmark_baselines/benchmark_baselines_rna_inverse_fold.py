import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(os.path.realpath(__file__)).parent.parent.parent))

from comb_opt.factory import task_factory
from comb_opt.optimizers import RandomSearch, LocalSearch, SimulatedAnnealing, BOCS, BOSS, COMBO, \
    Casmopolitan, BOiLS, MultiArmedBandit, GeneticAlgorithm
from comb_opt.utils.experiment_utils import run_experiment

if __name__ == '__main__':
    task_name = 'rna_inverse_fold'
    task_kwargs = {'target': 65}
    dtype = torch.float32

    task, search_space = task_factory(task_name, dtype, **task_kwargs)

    bo_n_init = 20
    bo_device = torch.device('cuda:1')
    max_num_iter = 200
    random_seeds = [42, 43, 44, 45, 46]

    rs_optim = RandomSearch(search_space=search_space, dtype=dtype)
    ls_optim = LocalSearch(search_space=search_space, dtype=dtype)
    sa_optim = SimulatedAnnealing(search_space=search_space, dtype=dtype)
    ga_optim = GeneticAlgorithm(search_space=search_space, dtype=dtype)
    bocs = BOCS(search_space=search_space, n_init=bo_n_init, dtype=dtype, device=bo_device)
    boss = BOSS(search_space=search_space, n_init=bo_n_init, model_max_batch_size=50, dtype=dtype, device=bo_device)
    combo = COMBO(search_space=search_space, n_init=bo_n_init, dtype=dtype, device=bo_device)
    casmopolitan = Casmopolitan(search_space=search_space, n_init=bo_n_init, dtype=dtype, device=bo_device)
    boils = BOiLS(search_space=search_space, n_init=bo_n_init, model_max_batch_size=50, dtype=dtype, device=bo_device)
    mab_optim = MultiArmedBandit(search_space=search_space, batch_size=1, max_n_iter=200, noisy_black_box=False,
                                 dtype=dtype)

    optimizers = [boss, boils, casmopolitan, combo, bocs, rs_optim, ls_optim, sa_optim, ga_optim, mab_optim]

    run_experiment(task=task, optimizers=optimizers, random_seeds=random_seeds, max_num_iter=max_num_iter,
                   very_verbose=False)
