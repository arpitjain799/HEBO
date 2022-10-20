import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(os.path.realpath(__file__)).parent.parent.parent))

from comb_opt.factory import task_factory
from comb_opt.utils.experiment_utils import run_experiment

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
    parser = argparse.ArgumentParser(add_help=True, description='BOF - Mix And Match - Antibody design')
    parser.add_argument("--device_id", type=int, default=0, help="Cuda device id (cpu is used if id is negative)")
    parser.add_argument("--absolut_dir", type=str, default=None, help="Path to AbsolutNoLib")
    parser.add_argument("--use_tr", action="store_true", help="Whether to use Trust-Region based methods")

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

    use_tr = args.use_tr

    opt_kwargs = dict(dtype=dtype, device=bo_device, use_tr=use_tr)

    gp_to_ga = GpToGaAcqOptim(search_space, bo_n_init, **opt_kwargs)
    gp_to_sa = GpToSaAcqOptim(search_space, bo_n_init, **opt_kwargs)
    gp_to_exhaustive_ls = GpToLsAcqOptim(search_space, bo_n_init, **opt_kwargs)
    gp_to_is = None
    if not use_tr:
        gp_to_is = Casmopolitan(search_space, bo_n_init, **opt_kwargs)

    gp_o_ga = GpOGaAcqOptim(search_space=search_space, n_init=bo_n_init, **opt_kwargs)
    gp_o_sa = GpOSaAcqOptim(search_space=search_space, n_init=bo_n_init, **opt_kwargs)
    gp_o_ls = GpOLsAcqOptim(search_space=search_space, n_init=bo_n_init, **opt_kwargs)
    gp_o_is = GpOIsAcqOptim(search_space=search_space, n_init=bo_n_init, **opt_kwargs)

    gp_diffusion_ga = GpDiffusionGaAcqOptim(search_space, bo_n_init, **opt_kwargs)
    gp_diffusion_sa = GpDiffusionSaAcqOptim(search_space, bo_n_init, **opt_kwargs)
    gp_diffusion_is = GpDiffusionIsAcqOptim(search_space, bo_n_init, **opt_kwargs)
    gp_diffusion_ls = None
    if use_tr:
        gp_diffusion_ls = COMBO(search_space, bo_n_init, **opt_kwargs)

    gp_ssk_sa = GpSskSaAcqOptim(search_space, bo_n_init, model_max_batch_size=50, **opt_kwargs)
    gp_ssk_ls = GpSskLsAcqOptim(search_space, bo_n_init, model_max_batch_size=50, **opt_kwargs)
    gp_ssk_ga = None
    if use_tr:
        gp_ssk_ga = BOSS(search_space, bo_n_init, model_max_batch_size=50, **opt_kwargs)
    gp_ssk_is = None
    if not use_tr:
        gp_ssk_is = BOiLS(search_space, bo_n_init, model_max_batch_size=50, **opt_kwargs)

    lr_sparse_hs_ga = LrGaAcqOptim(search_space, bo_n_init, **opt_kwargs)
    lr_sparse_hs_ls = LrLsAcqOptim(search_space, bo_n_init, **opt_kwargs)
    lr_sparse_hs_is = LrIsAcqOptim(search_space, bo_n_init, **opt_kwargs)
    lr_sparse_hs_sa = None
    if use_tr:
        lr_sparse_hs_sa = BOCS(search_space, bo_n_init, **opt_kwargs)

    optimizers = [
        # gp_ssk_sa,
        # gp_ssk_ls,
        # gp_diffusion_ga,
        # gp_diffusion_sa,
        # gp_diffusion_is,
        # gp_to_ga,
        # gp_to_sa,
        # gp_o_ga,
        # gp_o_sa,
        # gp_o_ls,
        # gp_o_is,
        # lr_sparse_hs_ga,
        # lr_sparse_hs_ls,
        # lr_sparse_hs_is
    ]

    optional_opts = [
        # gp_to_is,
        # gp_diffusion_ls,
        # lr_sparse_hs_sa,
        # gp_ssk_ga,
        gp_ssk_is,
    ]

    for opt in optional_opts:
        if opt is not None:
            optimizers.append(opt)

    run_experiment(task=task, optimizers=optimizers, random_seeds=random_seeds, max_num_iter=max_num_iter,
                   very_verbose=True)
