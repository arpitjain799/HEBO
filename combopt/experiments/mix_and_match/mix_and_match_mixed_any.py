import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(os.path.realpath(__file__)).parent.parent.parent))

from comb_opt.factory import task_factory
from comb_opt.utils.experiment_utils import run_experiment
from comb_opt.utils.general_utils import log

from comb_opt.optimizers import COMBO, BOCS, BOSS, BOiLS, Casmopolitan, CoCaBO
from comb_opt.optimizers.mix_and_match.gp_diff_ker_ga_acq_optim import GpDiffusionGaAcqOptim
from comb_opt.optimizers.mix_and_match.gp_diff_ker_is_acq_optim import GpDiffusionIsAcqOptim
from comb_opt.optimizers.mix_and_match.gp_diff_ker_sa_acq_optim import GpDiffusionSaAcqOptim
from comb_opt.optimizers.mix_and_match.gp_o_ker_ga_acq_optim import GpOGaAcqOptim
from comb_opt.optimizers.mix_and_match.gp_o_ker_is_acq_optim import GpOIsAcqOptim
from comb_opt.optimizers.mix_and_match.gp_o_ker_ls_acq_optim import GpOLsAcqOptim
from comb_opt.optimizers.mix_and_match.gp_o_ker_sa_acq_optim import GpOSaAcqOptim
from comb_opt.optimizers.mix_and_match.gp_ssk_ker_ls_acq_optim import GpSskLsAcqOptim
from comb_opt.optimizers.mix_and_match.gp_ssk_ker_sa_acq_optim import GpSskSaAcqOptim
from comb_opt.optimizers.mix_and_match.gp_to_ker_ls_acq_optim import GpToLsAcqOptim
from comb_opt.optimizers.mix_and_match.gp_to_ker_ga_acq_optim import GpToGaAcqOptim
from comb_opt.optimizers.mix_and_match.gp_to_ker_sa_acq_optim import GpToSaAcqOptim
from comb_opt.optimizers.mix_and_match.lr_ga_acq_optim import LrGaAcqOptim
from comb_opt.optimizers.mix_and_match.lr_is_acq_optim import LrIsAcqOptim
from comb_opt.optimizers.mix_and_match.lr_ls_acq_optim import LrLsAcqOptim

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True, description='BOF - Mix And Match')
    parser.add_argument("--device_id", type=int, default=0, help="Cuda device id (cpu is used if id is negative)")
    parser.add_argument("--use_tr", action="store_true", help="Whether to use Trust-Region based methods")
    parser.add_argument("--task_name", type=str, required=True, help="Name of the task")
    parser.add_argument("--optimizers_ids", type=str, nargs="+", required=True, help="Name of the methods to run")
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="Seeds to run")

    args = parser.parse_args()

    task_name = args.task_name
    if task_name == "ackley-53":
        task_name = "ackley"
        num_dims = [50, 3]
        variable_type = ['nominal', 'num']
        num_categories = [2, None]
        task_name_suffix = " 50-nom-2 3-num"
        lb = np.zeros(53)
        lb[-3:] = -1
        task_kwargs = dict(num_dims=num_dims, variable_type=variable_type, num_categories=num_categories,
                           task_name_suffix=task_name_suffix, lb=lb, ub=1)
    elif task_name == 'xgboost_opt':
        dataset_id = "mnist"
        task_kwargs = dict(dataset_id=dataset_id)
    elif task_name == 'aig_optimization_hyp':
        task_kwargs = {'designs_group_id': "sin", "operator_space_id": "basic", "objective": "both",
                       "seq_operators_pattern_id": "basic_w_post_map"}
    elif task_name == 'svm_opt':
        task_kwargs = dict()
    else:
        raise ValueError(task_name)
    dtype = torch.float32

    task, search_space = task_factory(task_name, dtype, **task_kwargs)

    bo_n_init = 20
    if args.device_id >= 0 and torch.cuda.is_available():
        bo_device = torch.device(f'cuda:{args.device_id}')
    else:
        bo_device = torch.device("cpu")

    max_num_iter = 200
    random_seeds = args.seeds

    use_tr = args.use_tr

    opt_kwargs = dict(dtype=dtype, device=bo_device, use_tr=use_tr)

    gp_to_ga = GpToGaAcqOptim(search_space, bo_n_init, **opt_kwargs)
    gp_to_sa = GpToSaAcqOptim(search_space, bo_n_init, **opt_kwargs)
    # gp_to_ls = GpToLsAcqOptim(search_space, bo_n_init, **opt_kwargs)
    gp_to_is = Casmopolitan(search_space, bo_n_init, **opt_kwargs)
    gp_to_mab = CoCaBO(search_space=search_space, n_init=bo_n_init, model_cat_kernel_name='transformed_overlap', **opt_kwargs)

    gp_o_ga = GpOGaAcqOptim(search_space=search_space, n_init=bo_n_init, **opt_kwargs)
    gp_o_sa = GpOSaAcqOptim(search_space=search_space, n_init=bo_n_init, **opt_kwargs)
    # gp_o_ls = GpOLsAcqOptim(search_space=search_space, n_init=bo_n_init, **opt_kwargs)
    gp_o_is = GpOIsAcqOptim(search_space=search_space, n_init=bo_n_init, **opt_kwargs)
    gp_o_mab = CoCaBO(search_space=search_space, n_init=bo_n_init, **opt_kwargs)

    if search_space.num_nominal + search_space.num_ordinal == search_space.num_params:
        gp_diffusion_ga = GpDiffusionGaAcqOptim(search_space=search_space, n_init=bo_n_init, **opt_kwargs)
        gp_diffusion_sa = GpDiffusionSaAcqOptim(search_space=search_space, n_init=bo_n_init, **opt_kwargs)
        gp_diffusion_is = GpDiffusionIsAcqOptim(search_space=search_space, n_init=bo_n_init, **opt_kwargs)
        # gp_diffusion_ls = COMBO(search_space=search_space, n_init=bo_n_init, **opt_kwargs)


    selected_optimizers = []
    for opt_id in args.optimizers_ids:
        opt = globals().get(opt_id, None)
        if opt is None:
            log(f"Optimizer {opt_id} is None")
            continue
        selected_optimizers.append(opt)

    run_experiment(task=task, optimizers=selected_optimizers, random_seeds=random_seeds, max_num_iter=max_num_iter,
                   very_verbose=False)
