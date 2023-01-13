import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(os.path.realpath(__file__)).parent.parent))

from comb_opt.factory import task_factory
from comb_opt.utils.experiment_utils import run_experiment
from comb_opt.utils.general_utils import log

from comb_opt.optimizers import COMBO, BOCS, BOSS, BOiLS, Casmopolitan
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
    parser = argparse.ArgumentParser(add_help=True, description='BOF - Combinatorial Tasks')
    parser.add_argument("--device_id", type=int, default=0, help="Cuda device id (cpu is used if id is negative)")
    parser.add_argument("--use_tr", action="store_true", help="Whether to use Trust-Region based methods")
    parser.add_argument("--task_name", type=str, required=True, help="Name of the task")
    parser.add_argument("--optimizers_ids", type=str, nargs="+", required=True, help="Name of the methods to run")
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="Seeds to run")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")

    args = parser.parse_args()

    task_name = args.task_name
    if task_name == "rna_inverse_fold":
        task_kwargs = {'target': 65}
    elif "ackley" in task_name or "levy" in task_name:
        task_kwargs = None
        n_cats = 11
        for synth in ["ackley", "levy"]:
            if synth not in task_name:
                continue
            if task_name == synth:
                dim = 20
                task_name_suffix = None
            else:
                dim = int(task_name.split("-")[-1])
                assert task_name == f"{synth}-{dim}"
                task_name_suffix = f"{dim}-nom-{n_cats}"
            task_kwargs = {'num_dims': dim, 'variable_type': 'nominal', 'num_categories': n_cats,
                           "task_name_suffix": task_name_suffix}
            task_name = synth
        assert task_kwargs is not None
    elif task_name == 'mig_optimization':
        task_kwargs = {'ntk_name': "sqrt", "objective": "both"}
    elif task_name == 'aig_optimization':
        task_kwargs = {'designs_group_id': "sin", "operator_space_id": "basic", "objective": "both"}
    elif task_name == 'antibody_design':
        task_kwargs = {'num_cpus': 5, 'first_cpu': 0, 'absolut_dir': None}
    elif task_name == 'pest':
        task_kwargs = {}
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
    gp_to_ls = GpToLsAcqOptim(search_space, bo_n_init, **opt_kwargs)
    gp_to_is = Casmopolitan(search_space, bo_n_init, **opt_kwargs)

    gp_o_ga = GpOGaAcqOptim(search_space=search_space, n_init=bo_n_init, **opt_kwargs)
    gp_o_sa = GpOSaAcqOptim(search_space=search_space, n_init=bo_n_init, **opt_kwargs)
    gp_o_ls = GpOLsAcqOptim(search_space=search_space, n_init=bo_n_init, **opt_kwargs)
    gp_o_is = GpOIsAcqOptim(search_space=search_space, n_init=bo_n_init, **opt_kwargs)

    gp_diffusion_ga = GpDiffusionGaAcqOptim(search_space, bo_n_init, **opt_kwargs)
    gp_diffusion_sa = GpDiffusionSaAcqOptim(search_space, bo_n_init, **opt_kwargs)
    gp_diffusion_is = GpDiffusionIsAcqOptim(search_space, bo_n_init, **opt_kwargs)
    gp_diffusion_ls = COMBO(search_space, bo_n_init, **opt_kwargs)

    gp_ssk_sa = GpSskSaAcqOptim(search_space, bo_n_init, model_max_batch_size=50, **opt_kwargs)
    gp_ssk_ls = GpSskLsAcqOptim(search_space, bo_n_init, model_max_batch_size=50, **opt_kwargs)
    gp_ssk_ga = BOSS(search_space, bo_n_init, model_max_batch_size=50, **opt_kwargs)
    gp_ssk_is = BOiLS(search_space, bo_n_init, model_max_batch_size=50, **opt_kwargs)

    lr_sparse_hs_ga = LrGaAcqOptim(search_space, bo_n_init, **opt_kwargs)
    lr_sparse_hs_ls = LrLsAcqOptim(search_space, bo_n_init, **opt_kwargs)
    lr_sparse_hs_is = LrIsAcqOptim(search_space, bo_n_init, **opt_kwargs)
    lr_sparse_hs_sa = BOCS(search_space, bo_n_init, **opt_kwargs)

    selected_optimizers = []
    for opt_id in args.optimizers_ids:
        opt = globals().get(opt_id, None)
        if opt is None:
            log(f"Optimizer {opt_id} is None")
            continue
        selected_optimizers.append(opt)

    run_experiment(task=task, optimizers=selected_optimizers, random_seeds=random_seeds, max_num_iter=max_num_iter,
                   very_verbose=args.verbose > 1)
